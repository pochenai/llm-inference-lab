use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

use cuda_async::{
    cuda_graph::CudaGraph,
    device_operation::{DeviceOp, ExecutionContext},
};
use cutile::{
    cuda_core::{memcpy_htod_async, DType, PinnedHostBuffer},
    tensor::Tensor,
};

use super::{metrics::CudaGraphMetrics, workspace::Buffers};
use crate::ModelError;

const MAX_CACHED_GRAPHS: usize = 8;

/// Storage identity and fixed launch geometry; token values/positions are GPU metadata.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct DecodeGraphKey {
    pub(super) table: usize,
    pub(super) batch: usize,
    pub(super) splits: usize,
}

/// Storage only: no KVContext, request lease or previous token owner is captured.
pub(super) struct DecodeCache<T: DType> {
    pub(super) layers: Vec<(Arc<Tensor<T>>, Arc<Tensor<T>>)>,
    pub(super) table: Arc<Tensor<i32>>,
}

pub(super) struct DecodeGraph<T: DType> {
    pub(super) key: DecodeGraphKey,
    pub(super) metadata: Tensor<i32>,
    pub(super) sources: Tensor<u64>,
    pub(super) host_metadata: PinnedHostBuffer<i32>,
    pub(super) host_sources: PinnedHostBuffer<u64>,
    pub(super) output: Arc<Tensor<i32>>,
    pub(super) logits: Arc<Tensor<f32>>,
    pub(super) graph: CudaGraph<()>,
    pub(super) buffers: Buffers<T>,
    pub(super) _cache: DecodeCache<T>,
}

impl<T: DType> DecodeGraph<T> {
    /// Both GPU completion and every escaped prediction/readback must release the instance.
    fn idle(this: &Arc<Self>) -> bool {
        Arc::strong_count(this) == 1 && Arc::strong_count(&this.output) == 1
    }

    /// Upload current addresses/positions before replay on this submission's ordered stream.
    pub(super) unsafe fn replay(
        self: &Arc<Self>,
        metadata: &[i32],
        sources: &[u64],
        exec: &ExecutionContext,
    ) -> Result<(), ModelError> {
        exec.retain(self.clone())?;
        // Exclusive checkout protects pinned staging while uploads are in flight.
        std::ptr::copy_nonoverlapping(
            metadata.as_ptr(),
            self.host_metadata.as_ptr().cast_mut(),
            metadata.len(),
        );
        std::ptr::copy_nonoverlapping(
            sources.as_ptr(),
            self.host_sources.as_ptr().cast_mut(),
            sources.len(),
        );
        memcpy_htod_async(
            self.metadata.device_pointer().cu_deviceptr(),
            self.host_metadata.as_ptr(),
            metadata.len(),
            exec.get_cuda_stream(),
        )
        .map_err(cuda_async::error::DeviceError::from)?;
        memcpy_htod_async(
            self.sources.device_pointer().cu_deviceptr(),
            self.host_sources.as_ptr(),
            sources.len(),
            exec.get_cuda_stream(),
        )
        .map_err(cuda_async::error::DeviceError::from)?;
        self.graph.launch().execute(exec)?;
        Ok(())
    }
}

pub(super) struct DecodeGraphCache<T: DType> {
    entries: Mutex<Vec<Arc<DecodeGraph<T>>>>,
    captures: AtomicU64,
    replays: AtomicU64,
}

impl<T: DType> Default for DecodeGraphCache<T> {
    fn default() -> Self {
        Self {
            entries: Mutex::default(),
            captures: AtomicU64::new(0),
            replays: AtomicU64::new(0),
        }
    }
}

impl<T: DType> DecodeGraphCache<T> {
    /// Build outside the cache lock and DeviceOp execution lock; never evict a live output.
    pub(super) fn acquire(
        &self,
        key: DecodeGraphKey,
        build: impl FnOnce() -> Result<DecodeGraph<T>, ModelError>,
    ) -> Result<Arc<DecodeGraph<T>>, ModelError> {
        {
            let entries = self.entries.lock().unwrap();
            if let Some(entry) = entries
                .iter()
                .find(|e| e.key == key && DecodeGraph::idle(e))
            {
                return Ok(entry.clone());
            }
        }
        let entry = Arc::new(build()?);
        self.captures.fetch_add(1, Ordering::Relaxed);
        let mut entries = self.entries.lock().unwrap();
        if entries.len() == MAX_CACHED_GRAPHS {
            if let Some(index) = entries.iter().position(DecodeGraph::idle) {
                entries.swap_remove(index);
            }
        }
        // All cached instances may be live; this overflow instance simply dies after use.
        if entries.len() < MAX_CACHED_GRAPHS {
            entries.push(entry.clone());
        }
        Ok(entry)
    }

    pub(super) fn replayed(&self) {
        self.replays.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn metrics(&self) -> CudaGraphMetrics {
        let entries = self.entries.lock().unwrap();
        CudaGraphMetrics {
            captures: self.captures.load(Ordering::Relaxed),
            replays: self.replays.load(Ordering::Relaxed),
            cached_graphs: entries.len(),
            retained_workspace_bytes: entries.iter().map(|e| e.buffers.allocation_bytes).sum(),
        }
    }
}
