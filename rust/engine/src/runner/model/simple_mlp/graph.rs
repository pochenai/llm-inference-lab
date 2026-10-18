use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

use cuda_async::{
    cuda_graph::CudaGraph,
    device_buffer::DeviceAllocation,
    device_operation::{value, DeviceOp, ExecutionContext},
};
use cutile::{
    api,
    cuda_core::{memcpy_htod_async, Stream},
    tensor::Tensor,
    tile_kernel::ToHostVecOp,
};

use super::{workspace::Buffers, ModelError, INPUT_DIM};

const MAX_IDLE_GRAPHS: usize = 4;

/// Addresses and launch geometry baked into a graph; request data is not a key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct GraphKey {
    pub device: usize,
    pub rows: usize,
    pub mapping_words: usize,
    pub payload_words: usize,
    pub cache: [usize; 3],
    pub weights: [usize; 6],
}

/// Private fixed buffers. Capture owners never include a request's KV state lease.
pub(super) struct GraphInstance {
    pub key: GraphKey,
    pub input: Arc<Tensor<i32>>,
    pub buffers: Buffers<Tensor<f32>>,
    pub graph: CudaGraph<()>,
}

#[derive(Default)]
pub(super) struct GraphPool {
    idle: Mutex<Vec<GraphInstance>>,
    captures: AtomicUsize,
    replays: AtomicUsize,
}

impl GraphPool {
    /// Called outside DeviceOp execution: capture acquires its own execution lock.
    pub fn acquire(
        self: &Arc<Self>,
        key: GraphKey,
        build: impl FnOnce() -> Result<GraphInstance, ModelError>,
    ) -> Result<Arc<GraphLease>, ModelError> {
        let cached = {
            let mut idle = self.idle.lock().unwrap();
            idle.iter()
                .position(|entry| entry.key == key)
                .map(|i| idle.swap_remove(i))
        };
        let instance = match cached {
            Some(instance) => instance,
            None => {
                let instance = build()?;
                self.captures.fetch_add(1, Ordering::Relaxed);
                instance
            }
        };
        Ok(Arc::new(GraphLease {
            instance: Some(instance),
            pool: self.clone(),
        }))
    }

    #[cfg(test)]
    pub fn stats(&self) -> (usize, usize) {
        (
            self.captures.load(Ordering::Relaxed),
            self.replays.load(Ordering::Relaxed),
        )
    }
}

/// One in-flight batch exclusively owns this graph and all its writable buffers.
pub(super) struct GraphLease {
    instance: Option<GraphInstance>,
    pool: Arc<GraphPool>,
}

impl GraphLease {
    /// Caller owns exec; upload, replay, and pageable readback use its same stream.
    pub unsafe fn execute(
        self: &Arc<Self>,
        payload: Arc<Vec<i32>>,
        exec: &ExecutionContext,
    ) -> Result<Vec<f32>, ModelError> {
        let instance = self.instance.as_ref().unwrap();
        if payload.len() != instance.key.payload_words
            || exec.get_device_id() != instance.key.device
        {
            return Err(ModelError::InvalidInput(
                "graph payload shape or device differs".into(),
            ));
        }
        exec.device()
            .bind_to_thread()
            .map_err(cuda_async::error::DeviceError::from)?;
        exec.retain(self.clone())?;
        exec.retain(payload.clone())?;
        memcpy_htod_async(
            instance.input.device_pointer().cu_deviceptr(),
            payload.as_ptr(),
            payload.len(),
            exec.get_cuda_stream(),
        )
        .map_err(cuda_async::error::DeviceError::from)?;
        instance.graph.launch().execute(exec)?;
        self.pool.replays.fetch_add(1, Ordering::Relaxed);
        // Read via the same owned-Tensor to_host_vec implementation as eager. This
        // alias also owns the checkout, so its allocation cannot be recycled early.
        let output = Tensor::<f32>::from_foreign(
            self.clone(),
            instance.buffers.output.shape().to_vec(),
            instance.buffers.output.strides().to_vec(),
        );
        Ok(value(output).to_host_vec().execute(exec)?)
    }
}

// SAFETY: this lease owns a fixed output allocation and stable device/size until drop.
unsafe impl DeviceAllocation for GraphLease {
    fn device_ptr(&self) -> u64 {
        self.instance
            .as_ref()
            .unwrap()
            .buffers
            .output
            .device_pointer()
            .cu_deviceptr()
    }
    fn len_bytes(&self) -> usize {
        self.instance.as_ref().unwrap().buffers.output.num_bytes()
    }
    fn device_id(&self) -> usize {
        self.instance.as_ref().unwrap().key.device
    }
}

impl Drop for GraphLease {
    fn drop(&mut self) {
        let mut idle = self.pool.idle.lock().unwrap();
        if idle.len() == MAX_IDLE_GRAPHS {
            idle.remove(0);
        }
        idle.push(self.instance.take().unwrap());
    }
}

pub(super) fn allocate(
    rows: usize,
    stream: &Arc<Stream>,
) -> Result<Buffers<Tensor<f32>>, ModelError> {
    Ok(Buffers {
        x: api::zeros(&[rows, INPUT_DIM]).sync_on(stream)?,
        q: api::zeros(&[rows, INPUT_DIM]).sync_on(stream)?,
        attention: api::zeros(&[rows, INPUT_DIM]).sync_on(stream)?,
        residual: api::zeros(&[rows, INPUT_DIM]).sync_on(stream)?,
        hidden: api::zeros(&[rows, INPUT_DIM]).sync_on(stream)?,
        output: api::zeros(&[rows]).sync_on(stream)?,
    })
}
