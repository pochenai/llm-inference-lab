use super::state::Qwen3_5ExecutionView;
use anyhow::{ensure, Result};
use cuda_async::{
    cuda_graph::CudaGraph,
    device_operation::{DeviceOp, ExecutionContext},
};
use cutile::{
    cuda_core::{memcpy_htod_async, DType, PinnedHostBuffer},
    tensor::Tensor,
};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

pub(super) type ComputeOutput = (Tensor<f32>, Tensor<i32>);
pub(super) enum Execution<'a, 'b, T: DType> {
    Eager(&'a mut Qwen3_5ExecutionView<'b, T>),
    Graph(&'a GraphData<T>),
}

pub(super) struct GraphCacheStorage<T: DType> {
    pub layers: Vec<(Arc<Tensor<T>>, Arc<Tensor<T>>)>,
    pub table: Arc<Tensor<i32>>,
}

pub(super) struct GraphData<T: DType> {
    pub ids: Tensor<u32>,
    /// [position, table offset, conv_0, recurrent_0, ...]. Addresses change per request.
    pub metadata: Tensor<u64>,
    pub cache: GraphCacheStorage<T>,
}

pub(super) struct DecodeGraph<T: DType> {
    pub key: usize,
    pub data: GraphData<T>,
    pub graph: CudaGraph<Result<ComputeOutput>>,
    pub logits: Arc<Tensor<f32>>,
    pub token: Arc<Tensor<i32>>,
    pub host_metadata: PinnedHostBuffer<u64>,
    pub host_id: PinnedHostBuffer<u32>,
}

impl<T: DType> DecodeGraph<T> {
    /// Caller has the matching pending KV write view and retains every addressed state.
    pub unsafe fn replay(
        self: &Arc<Self>,
        token: u32,
        metadata: &[u64],
        exec: &ExecutionContext,
    ) -> Result<()> {
        ensure!(
            exec.context().cu_ctx() == self.data.ids.context().cu_ctx(),
            "graph replay CUDA context mismatch"
        );
        ensure!(
            metadata.len() == self.data.metadata.size(),
            "graph metadata size mismatch"
        );
        exec.retain(self.clone())?;
        std::ptr::copy_nonoverlapping(
            metadata.as_ptr(),
            self.host_metadata.as_ptr().cast_mut(),
            metadata.len(),
        );
        self.host_id.as_ptr().cast_mut().write(token);
        memcpy_htod_async(
            self.data.metadata.device_pointer().cu_deviceptr(),
            self.host_metadata.as_ptr(),
            metadata.len(),
            exec.get_cuda_stream(),
        )?;
        memcpy_htod_async(
            self.data.ids.device_pointer().cu_deviceptr(),
            self.host_id.as_ptr(),
            1,
            exec.get_cuda_stream(),
        )?;
        self.graph.launch().execute(exec)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct CudaGraphMetrics {
    pub captures: u64,
    pub replays: u64,
    pub cached_graphs: usize,
}

pub(super) struct DecodeGraphs<T: DType> {
    entries: Mutex<Vec<Arc<DecodeGraph<T>>>>,
    captures: AtomicU64,
    replays: AtomicU64,
}
impl<T: DType> Default for DecodeGraphs<T> {
    fn default() -> Self {
        Self {
            entries: Mutex::default(),
            captures: AtomicU64::new(0),
            replays: AtomicU64::new(0),
        }
    }
}
impl<T: DType> DecodeGraphs<T> {
    fn idle(entry: &Arc<DecodeGraph<T>>) -> bool {
        Arc::strong_count(entry) == 1
            && Arc::strong_count(&entry.logits) == 1
            && Arc::strong_count(&entry.token) == 1
    }
    pub fn acquire(
        &self,
        key: usize,
        build: impl FnOnce() -> Result<DecodeGraph<T>>,
    ) -> Result<Arc<DecodeGraph<T>>> {
        {
            let entries = self.entries.lock().unwrap();
            if let Some(entry) = entries.iter().find(|e| e.key == key && Self::idle(e)) {
                return Ok(entry.clone());
            }
        }
        let entry = Arc::new(build()?);
        self.captures.fetch_add(1, Ordering::Relaxed);
        let mut entries = self.entries.lock().unwrap();
        if entries.len() == 4 {
            if let Some(index) = entries.iter().position(Self::idle) {
                entries.swap_remove(index);
            }
        }
        if entries.len() < 4 {
            entries.push(entry.clone());
        }
        Ok(entry)
    }
    pub fn replayed(&self) {
        self.replays.fetch_add(1, Ordering::Relaxed);
    }
    pub fn metrics(&self) -> CudaGraphMetrics {
        CudaGraphMetrics {
            captures: self.captures.load(Ordering::Relaxed),
            replays: self.replays.load(Ordering::Relaxed),
            cached_graphs: self.entries.lock().unwrap().len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::model::{
        submission::{tests::Gate, SubmittedGpuWorkWithCompletion},
        ModelError,
    };
    use cuda_async::{
        device_operation::{value, with_context},
        scheduling_policies::{SchedulingPolicy, SingleStream},
    };
    use cutile::api;
    use futures::{poll, FutureExt};

    #[tokio::test]
    async fn gated_replays_and_escaped_outputs_exclude_graph_reuse() -> Result<()> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let context = cutile::cuda_core::Context::new(0)?;
        let policy: Arc<dyn SchedulingPolicy> = Arc::new(SingleStream::new(&context)?);
        let stream = policy.next_stream()?;
        let graphs = DecodeGraphs::<f32>::default();
        let build = || -> Result<DecodeGraph<f32>> {
            let (data, logits, token) = with_context(|exec| {
                value((|| unsafe {
                    Ok::<_, anyhow::Error>((
                        GraphData {
                            ids: api::zeros::<u32>(&[1]).execute(exec)?,
                            metadata: api::zeros::<u64>(&[2]).execute(exec)?,
                            cache: GraphCacheStorage {
                                layers: vec![],
                                table: Arc::new(api::zeros::<i32>(&[1, 1]).execute(exec)?),
                            },
                        },
                        api::zeros::<f32>(&[1]).execute(exec)?,
                        api::zeros::<i32>(&[1]).execute(exec)?,
                    ))
                })())
            })
            .sync_on(&stream)??;
            // Empty compute is sufficient: the test gates the real metadata DMA and launch.
            let mut graph = CudaGraph::capture(stream.clone(), value(Ok((logits, token))))?;
            let (logits, token) = graph.take_output().unwrap()?;
            Ok(DecodeGraph {
                key: 0,
                data,
                graph,
                logits: Arc::new(logits),
                token: Arc::new(token),
                host_metadata: PinnedHostBuffer::zeroed(&context, 2)?,
                host_id: PinnedHostBuffer::zeroed(&context, 1)?,
            })
        };
        // Capture, allocation and completion initialization all precede the gates.
        let a = graphs.acquire(0, build)?;
        let b = graphs.acquire(0, build)?;
        drop((a, b));
        SubmittedGpuWorkWithCompletion::submit(&policy, |_| ())
            .await?
            .gpu_completion
            .await?;
        let first = graphs.acquire(0, || anyhow::bail!("expected warmed entry"))?;
        let first_output = first.logits.clone();
        let first_gate = Gate::arm(&stream)?;
        let first_expired = first_gate.expired.clone();
        let work = SubmittedGpuWorkWithCompletion::submit(&policy, move |exec| {
            unsafe { first.replay(7, &[2, 0], exec) }
                .map_err(|e| ModelError::Execution(e.to_string()))
        })
        .await?;
        work.output?;
        let mut first_done = work.gpu_completion.boxed();
        assert!(poll!(&mut first_done).is_pending());
        let second = graphs.acquire(0, || anyhow::bail!("expected second warmed entry"))?;
        assert!(!Arc::ptr_eq(&second.logits, &first_output));
        let second_gate = Gate::arm(&stream)?;
        let second_expired = second_gate.expired.clone();
        let work = SubmittedGpuWorkWithCompletion::submit(&policy, move |exec| {
            unsafe { second.replay(9, &[3, 0], exec) }
                .map_err(|e| ModelError::Execution(e.to_string()))
        })
        .await?;
        work.output?;
        let mut second_done = work.gpu_completion.boxed();
        assert!(poll!(&mut second_done).is_pending());
        assert!(graphs
            .acquire(0, || anyhow::bail!("both entries live"))
            .is_err());
        drop(first_gate);
        first_done.await?;
        assert!(
            !first_expired.load(Ordering::Acquire),
            "first GPU gate expired"
        );
        assert!(poll!(&mut second_done).is_pending());
        assert!(graphs
            .acquire(0, || anyhow::bail!("escaped output still live"))
            .is_err());
        drop(first_output);
        let available = graphs.acquire(0, || anyhow::bail!("completed entry must be reusable"))?;
        drop(available);
        drop(second_gate);
        second_done.await?;
        assert!(
            !second_expired.load(Ordering::Acquire),
            "second GPU gate expired"
        );
        assert_eq!(graphs.metrics().captures, 2);
        Ok(())
    }
}
