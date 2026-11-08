use std::sync::Arc;

use anyhow::{anyhow, Result};
use cuda_async::{
    device_operation::{value, DeviceOp, ExecutionContext, IntoDeviceOp, Value},
    error::DeviceError,
    launch::AsyncKernelLaunchBuilder,
    scheduling_policies::SchedulingPolicy,
};
use cutile::{
    api,
    cuda_core::{Context, DType, DeviceCopy},
    tensor::{KernelInput, KernelInputStored, Reshape, SpecializationBits, Tensor},
};

/// Permanently read-only model storage. Only a completed fresh upload can create it;
/// no tensor/buffer accessor or mutable kernel binding can expose a writer.
pub(super) struct FrozenWeight<T: DType> {
    tensor: Arc<Tensor<T>>,
}

impl<T: DType + DeviceCopy> FrozenWeight<T> {
    pub async fn from_host(
        data: Vec<T>,
        shape: &[usize],
        policy: &Arc<dyn SchedulingPolicy>,
    ) -> Result<Self> {
        let tensor = api::copy_host_vec_to_device(&Arc::new(data))
            .schedule(policy)?
            .await?
            .reshape(shape)?;
        Ok(Self {
            tensor: Arc::new(tensor),
        })
    }
}
impl<T: DType> FrozenWeight<T> {
    pub fn shape(&self) -> &[i32] {
        self.tensor.shape()
    }
    pub fn context(&self) -> &Arc<Context> {
        self.tensor.context()
    }
    pub fn reshape(self, shape: &[usize]) -> Result<Self> {
        let tensor = Arc::try_unwrap(self.tensor)
            .map_err(|_| anyhow!("cannot reshape a shared frozen weight"))?;
        Ok(Self {
            tensor: Arc::new(tensor.reshape(shape)?),
        })
    }
}
impl<T: DType> Clone for FrozenWeight<T> {
    fn clone(&self) -> Self {
        Self {
            tensor: self.tensor.clone(),
        }
    }
}
impl<T: DType> IntoDeviceOp<Self> for FrozenWeight<T> {
    type Op = Value<Self>;
    fn into_op(self) -> Self::Op {
        value(self)
    }
}
impl<T: DType> KernelInput<T> for FrozenWeight<T> {
    type Stored = Self;
    type Returned = Self;
    fn prepare(self) -> Self {
        self
    }
    fn recover(stored: Self) -> Self {
        stored
    }
}

// SAFETY: the factory completes initialization before freezing and no writer can
// escape. Capture owns the tensor for the graph's entire lifetime; replay checks
// the graph context. Mutable storage continues to use ordinary access tracking.
unsafe impl<T: DType> KernelInputStored for FrozenWeight<T> {
    fn retain(&self, ctx: &ExecutionContext) -> Result<(), DeviceError> {
        if !ctx.is_recording() {
            return KernelInputStored::retain(&self.tensor, ctx);
        }
        if self.context().cu_ctx() != ctx.context().cu_ctx() {
            return Err(DeviceError::Internal(
                "frozen weight CUDA context mismatch".into(),
            ));
        }
        ctx.retain(self.tensor.clone())
    }
    fn push_kernel_args(&self, launcher: &mut AsyncKernelLaunchBuilder) {
        KernelInputStored::push_kernel_args(&self.tensor, launcher);
    }
    fn shape(&self) -> &[i32] {
        self.tensor.shape()
    }
    fn strides(&self) -> &[i32] {
        self.tensor.strides()
    }
    fn spec(&self) -> &SpecializationBits {
        self.tensor.spec()
    }
    fn dtype_str(&self) -> &'static str {
        T::DTYPE.as_str()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::model::{
        qwen3_5::kernels,
        submission::{tests::Gate, SubmittedGpuWorkWithCompletion},
    };
    use cuda_async::{
        cuda_graph::CudaGraph, device_operation::with_context, scheduling_policies::SingleStream,
    };
    use cutile::{
        tensor::{PartitionMut, ToHostVec},
        tile_kernel::TileKernel,
    };
    use futures::{poll, FutureExt};
    use std::{
        sync::{atomic::Ordering, mpsc},
        time::Duration,
    };

    struct Rescue {
        _allocation: cutile::cuda_core::RawDeviceBuffer,
        stream: Arc<cutile::cuda_core::Stream>,
    }
    impl Drop for Rescue {
        fn drop(&mut self) {
            // Declared before the gate: failures release the gate before draining.
            let _ = self.stream.synchronize();
        }
    }

    #[tokio::test]
    async fn frozen_capture_owns_weights_after_host_graph_and_weight_drop() -> Result<()> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let context = Context::new(0)?;
        let policy: Arc<dyn SchedulingPolicy> = Arc::new(SingleStream::new(&context)?);
        let stream = policy.next_stream()?;
        let weight = FrozenWeight::from_host(vec![1f32; 128], &[1, 128], &policy).await?;
        // No access through this owner; a failed assertion must not free live input.
        let _rescue = Rescue {
            _allocation: weight.tensor.as_buffer()?.storage().clone(),
            stream: stream.clone(),
        };
        let weak = Arc::downgrade(&weight.tensor);
        let mut out = api::zeros::<f32>(&[1, 128]).sync_on(&stream)?;
        kernels::last_row((&mut out).partition([1, 128]), weight.clone())
            .generics(vec!["f32".into()])
            .sync_on(&stream)?;
        let mut graph = CudaGraph::capture(
            stream.clone(),
            with_context(|exec| {
                value(unsafe {
                    kernels::last_row((&mut out).partition([1, 128]), weight.clone())
                        .generics(vec!["f32".into()])
                        .execute(exec)
                        .map(|_| ())
                })
            }),
        )?;
        graph.take_output().unwrap()?;
        // Force completion initialization before the assertion gate. The independent
        // releaser also covers drivers that synchronize during reactor allocation.
        let warm_gate = Gate::arm(&stream)?;
        let warm_expired = warm_gate.expired.clone();
        let (ready, waiting) = mpsc::channel();
        let release = std::thread::spawn(move || {
            let _ = waiting.recv_timeout(Duration::from_secs(1));
            drop(warm_gate);
        });
        let warm = SubmittedGpuWorkWithCompletion::submit(&policy, |_| ()).await?;
        let _ = ready.send(());
        release.join().unwrap();
        warm.gpu_completion.await?;
        assert!(
            !warm_expired.load(Ordering::Acquire),
            "completion warmup gate expired"
        );
        drop(weight);
        assert!(weak.upgrade().is_some(), "capture must own frozen storage");
        let gate = Gate::arm(&stream)?;
        let expired = gate.expired.clone();
        let work = SubmittedGpuWorkWithCompletion::submit(&policy, move |exec| {
            // The CPU graph owner drops after enqueue; the submission owns its exec.
            unsafe { graph.launch().execute(exec) }
        })
        .await?;
        work.output?;
        let mut done = work.gpu_completion.boxed();
        assert!(poll!(&mut done).is_pending());
        assert!(
            weak.upgrade().is_some(),
            "pending replay must own frozen storage"
        );
        drop(gate);
        done.await?;
        assert!(!expired.load(Ordering::Acquire), "replay gate expired");
        assert!(
            weak.upgrade().is_none(),
            "completed dropped graph should release its input"
        );
        assert_eq!(out.to_host_vec().sync_on(&stream)?, vec![1f32; 128]);
        Ok(())
    }

    #[tokio::test]
    async fn frozen_weight_rejects_another_context_on_the_same_device() -> Result<()> {
        use std::ffi::c_void;
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let primary = Context::new(0)?;
        let policy: Arc<dyn SchedulingPolicy> = Arc::new(SingleStream::new(&primary)?);
        let weight = FrozenWeight::from_host(vec![1f32; 128], &[1, 128], &policy).await?;
        type Destroy = unsafe extern "C" fn(*mut c_void) -> i32;
        struct Owner {
            raw: usize,
            destroy: Destroy,
            _library: libloading::Library,
        }
        impl Drop for Owner {
            fn drop(&mut self) {
                // All derived context/stream/graph owners have gone before this owner.
                unsafe {
                    (self.destroy)(self.raw as *mut c_void);
                }
            }
        }
        {
            // A separate valid user context is required: Context::new retains the
            // same primary context, even when called twice for one ordinal.
            let library = unsafe { libloading::Library::new("libcuda.so.1")? };
            let create = unsafe {
                *library.get::<unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32>(
                    b"cuCtxCreate_v2\0",
                )?
            };
            let destroy = unsafe { *library.get::<Destroy>(b"cuCtxDestroy_v2\0")? };
            let mut raw = std::ptr::null_mut();
            anyhow::ensure!(
                unsafe { create(&mut raw, 0, primary.cu_device()) } == 0,
                "cuCtxCreate failed"
            );
            let owner = Arc::new(Owner {
                raw: raw as usize,
                destroy,
                _library: library,
            });
            // SAFETY: this owner destroys the valid context only after all wrappers drop.
            let other = unsafe { Context::borrow_with_owner(raw, primary.cu_device(), 0, owner) };
            assert_ne!(primary.cu_ctx(), other.cu_ctx());
            let stream = other.new_stream()?;
            let exec = ExecutionContext::new(stream.clone());
            assert!(KernelInputStored::retain(&weight, &exec).is_err());
            let mut graph = CudaGraph::capture(
                stream.clone(),
                with_context(|exec| value(KernelInputStored::retain(&weight, exec))),
            )?;
            assert!(graph.take_output().unwrap().is_err());
            stream.synchronize()?;
        }
        primary.bind_to_thread()?;
        Ok(())
    }
}
