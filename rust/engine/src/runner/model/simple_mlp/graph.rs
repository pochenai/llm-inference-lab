use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

use cuda_async::{
    cuda_graph::CudaGraph,
    device_buffer::DeviceAllocation,
    device_operation::{DeviceOp, ExecutionContext},
};
use cutile::{
    api,
    cuda_core::{memcpy_htod_async, sys, Device, DriverError, Stream},
    tensor::Tensor,
};

use super::{workspace::Buffers, ModelError, INPUT_DIM};

const MAX_IDLE_GRAPHS: usize = 4;
// Short graphs finish inline without a reactor/wakeup round trip. Longer work
// falls back to the existing DeviceFuture completion path after this budget.
// This is a caller-thread busy-poll budget, trading CPU availability for latency,
// not a fixed delay: exit immediately when GPU completion is observed.
const INLINE_COMPLETION_BUDGET: Duration = Duration::from_millis(2);

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
    pub _buffers: Buffers<Tensor<f32>>,
    pub output: Arc<MappedOutput>,
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
    /// Caller owns exec. Upload and replay use its stream; output stores target
    /// mapped host memory. Never read that memory until the outer exec completes.
    pub unsafe fn execute(
        self: &Arc<Self>,
        payload: Arc<Vec<i32>>,
        exec: &ExecutionContext,
    ) -> Result<Readback, ModelError> {
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
        // The final kernel targets mapped host RAM, so no DtoH is queued here.
        // Occupy the current CPU/executor thread for up to the 2 ms polling
        // budget to avoid a notification/wakeup round trip for short work.
        // spin_loop() neither sleeps nor yields the Tokio task. GPU completion
        // exits early; budget expiry returns an unread handle so the outer
        // DeviceFuture can finish waiting (its own polling, then notification).
        let deadline = Instant::now() + INLINE_COMPLETION_BUDGET;
        while !exec
            .get_cuda_stream()
            .query()
            .map_err(cuda_async::error::DeviceError::from)?
        {
            if Instant::now() >= deadline {
                break;
            }
            std::hint::spin_loop();
        }
        Ok(Readback {
            lease: self.clone(),
        })
    }
}

pub(super) struct Readback {
    lease: Arc<GraphLease>,
}
impl Readback {
    /// The outer execution context must have completed before reading GPU stores.
    /// The lease keeps host/device aliases alive and prevents instance recycling
    /// even if completion used the asynchronous fallback.
    pub unsafe fn read(self) -> Vec<f32> {
        let output = &self.lease.instance.as_ref().unwrap().output;
        // Both source and destination are CPU RAM: copy completed values from
        // the pinned allocation into an ordinary Vec, not from GPU memory.
        std::slice::from_raw_parts(output.host.as_ptr(), output.len).to_vec()
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
) -> Result<(Buffers<Tensor<f32>>, Arc<MappedOutput>), ModelError> {
    let output = MappedOutput::new(rows, stream.device().clone())?;
    // Wrap the existing allocation and shape for cuTile. from_foreign neither
    // allocates another GPU buffer nor copies data; DeviceAllocation supplies
    // the GPU-visible address used by the kernel's output argument.
    // SAFETY: one tensor handle owns this fixed device alias. The allocation is
    // initialized and retained by both graph storage and the instance; its CPU
    // alias is read only after completion through the exclusive checkout.
    let tensor = unsafe { Tensor::from_foreign(output.clone(), vec![rows as i32], vec![1]) };
    Ok((
        Buffers {
            x: api::zeros(&[rows, INPUT_DIM]).sync_on(stream)?,
            q: api::zeros(&[rows, INPUT_DIM]).sync_on(stream)?,
            attention: api::zeros(&[rows, INPUT_DIM]).sync_on(stream)?,
            residual: api::zeros(&[rows, INPUT_DIM]).sync_on(stream)?,
            hidden: api::zeros(&[rows, INPUT_DIM]).sync_on(stream)?,
            output: tensor,
        },
        output,
    ))
}

/// One fixed mapped host allocation. The graph owns its device alias; only the
/// current checkout may access its host alias, after GPU completion.
/// Both pointer values are virtual addresses for the same physical system RAM,
/// not two copies of the data. Under UVA their numeric values may be equal;
/// keep the explicit CPU/GPU roles rather than assuming equality or inequality.
pub(super) struct MappedOutput {
    /// CPU-visible address, dereferenced only after the outer exec completes.
    host: std::ptr::NonNull<f32>,
    /// Device-visible alias passed to kernels; the allocation remains in RAM.
    device_pointer: u64,
    len: usize,
    device: Arc<Device>,
}
// SAFETY: pointers stay valid on the retained context; the exclusive graph lease
// prevents CPU/GPU or batch/batch overlap. No safe host slice is exposed.
unsafe impl Send for MappedOutput {}
unsafe impl Sync for MappedOutput {}
unsafe impl DeviceAllocation for MappedOutput {
    fn device_ptr(&self) -> u64 {
        // cuTile obtains the kernel argument's GPU address through this trait.
        self.device_pointer
    }
    fn len_bytes(&self) -> usize {
        self.len * std::mem::size_of::<f32>()
    }
    fn device_id(&self) -> usize {
        self.device.ordinal()
    }
}
impl MappedOutput {
    fn new(len: usize, device: Arc<Device>) -> Result<Arc<Self>, ModelError> {
        device
            .bind_to_thread()
            .map_err(cuda_async::error::DeviceError::from)?;
        let bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .filter(|&b| b > 0)
            .ok_or_else(|| ModelError::InvalidInput("mapped output size is invalid".into()))?;
        let mut host = std::mem::MaybeUninit::uninit();
        // Allocate page-locked CPU RAM, not GPU VRAM. DEVICEMAP enables a GPU
        // mapping; PORTABLE makes the allocation pinned across CUDA contexts.
        let code = unsafe {
            sys::cuMemHostAlloc(
                host.as_mut_ptr(),
                bytes,
                sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_DEVICEMAP,
            )
        };
        if code != 0 {
            return Err(cuda_async::error::DeviceError::from(DriverError(code)).into());
        }
        let host = unsafe { host.assume_init() };
        let mut pointer = std::mem::MaybeUninit::uninit();
        // Obtain the GPU-visible alias of that same allocation. This does not
        // allocate device memory or perform a copy. With UVA it may equal host.
        let code = unsafe { sys::cuMemHostGetDevicePointer_v2(pointer.as_mut_ptr(), host, 0) };
        if code != 0 {
            unsafe { sys::cuMemFreeHost(host) };
            return Err(cuda_async::error::DeviceError::from(DriverError(code)).into());
        }
        unsafe { std::ptr::write_bytes(host.cast::<u8>(), 0, bytes) };
        Ok(Arc::new(Self {
            host: std::ptr::NonNull::new(host.cast()).unwrap(),
            device_pointer: unsafe { pointer.assume_init() },
            len,
            device,
        }))
    }
}
impl Drop for MappedOutput {
    fn drop(&mut self) {
        let _ = self.device.bind_to_thread();
        // The two aliases share one allocation: free it once with CUDA's host
        // allocator, after the final graph/storage/checkout owner releases it.
        unsafe { sys::cuMemFreeHost(self.host.as_ptr().cast()) };
    }
}
