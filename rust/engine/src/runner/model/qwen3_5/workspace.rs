use std::sync::{Arc, Mutex};

use anyhow::{ensure, Result};
use cuda_async::{
    device_buffer::DeviceAllocation,
    device_operation::{DeviceOp, ExecutionContext},
};
use cutile::{
    api,
    cuda_core::{Context, DType, DeviceCopy},
    tensor::Tensor,
};

const MAX_IDLE: usize = 2;

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
/// Model-local scratch accounting; excludes weights, KV/recurrent state and allocator reservation.
pub struct WorkspaceMetrics {
    /// Cumulative backing tensor allocations, including evicted workspaces.
    pub allocated_buffers: usize,
    pub cached_workspaces: usize,
    pub in_use_workspaces: usize,
    pub resident_bytes: usize,
}

struct Bank<T: DType> {
    activations: Vec<Arc<Tensor<T>>>,
    floats: Vec<Arc<Tensor<f32>>>,
    integers: Vec<Arc<Tensor<i32>>>,
}

struct Workspace<T: DType> {
    rows: usize,
    context: Arc<Context>,
    banks: [Bank<T>; 3],
    bytes: usize,
}

struct PoolState<T: DType> {
    idle: Vec<Workspace<T>>,
    metrics: WorkspaceMetrics,
}

pub(super) struct WorkspacePool<T: DType> {
    state: Mutex<PoolState<T>>,
}

impl<T: DType> Default for WorkspacePool<T> {
    fn default() -> Self {
        Self {
            state: Mutex::new(PoolState {
                idle: Vec::new(),
                metrics: WorkspaceMetrics::default(),
            }),
        }
    }
}

impl<T: DType> WorkspacePool<T> {
    pub(super) fn metrics(&self) -> WorkspaceMetrics {
        self.state.lock().unwrap().metrics
    }

    /// Caller owns the submission; views and exec keep this checkout alive through completion.
    pub(super) unsafe fn acquire(
        self: &Arc<Self>,
        rows: usize,
        exec: &ExecutionContext,
    ) -> Result<Scratch<T>> {
        let _range = crate::profiling::ScopedRange::new(c"qwen3_5.workspace.checkout");
        let mut state = self.state.lock().unwrap();
        let workspace = state
            .idle
            .iter()
            .position(|w| w.rows == rows && w.context.cu_ctx() == exec.context().cu_ctx())
            .map(|index| state.idle.swap_remove(index))
            .unwrap_or_else(|| Workspace {
                rows,
                context: exec.context().clone(),
                bytes: 0,
                banks: std::array::from_fn(|_| Bank {
                    activations: Vec::new(),
                    floats: Vec::new(),
                    integers: Vec::new(),
                }),
            });
        state.metrics.cached_workspaces = state.idle.len();
        state.metrics.in_use_workspaces += 1;
        drop(state);
        let lease = Arc::new(Lease {
            workspace: Mutex::new(Some(workspace)),
            pool: self.clone(),
        });
        exec.retain(lease.clone())?;
        Ok(Scratch {
            lease,
            bank: 2,
            cursors: [0; 3],
        })
    }
}

struct Lease<T: DType> {
    workspace: Mutex<Option<Workspace<T>>>,
    pool: Arc<WorkspacePool<T>>,
}

impl<T: DType> Drop for Lease<T> {
    fn drop(&mut self) {
        let workspace = self.workspace.get_mut().unwrap().take().unwrap();
        let mut state = self.pool.state.lock().unwrap();
        state.metrics.in_use_workspaces -= 1;
        if state.idle.len() == MAX_IDLE {
            let evicted = state.idle.remove(0);
            state.metrics.resident_bytes -= evicted.bytes;
        }
        state.idle.push(workspace);
        state.metrics.cached_workspaces = state.idle.len();
    }
}

/// Global hidden/residual/head storage plus separate reusable DeltaNet/GQA layer banks.
/// Layer producers overwrite every valid scratch element; recurrent/KV state is separate.
pub(super) struct Scratch<T: DType> {
    lease: Arc<Lease<T>>,
    bank: usize,
    cursors: [usize; 3],
}

impl<T: DType + DeviceCopy> Scratch<T> {
    pub(super) fn layer(&mut self, delta: bool) {
        self.bank = usize::from(!delta);
        self.cursors = [0; 3];
    }

    pub(super) fn head(&mut self) {
        self.bank = 2;
        // Hidden and residual are the first two global activation slots.
        self.cursors = [2, 0, 0];
    }

    pub(super) unsafe fn activation(
        &mut self,
        shape: &[usize],
        exec: &ExecutionContext,
    ) -> Result<Tensor<T>> {
        self.get(shape, exec, 0, |bank| &mut bank.activations)
    }
    pub(super) unsafe fn float(
        &mut self,
        shape: &[usize],
        exec: &ExecutionContext,
    ) -> Result<Tensor<f32>> {
        self.get(shape, exec, 1, |bank| &mut bank.floats)
    }
    pub(super) unsafe fn integer(
        &mut self,
        shape: &[usize],
        exec: &ExecutionContext,
    ) -> Result<Tensor<i32>> {
        self.get(shape, exec, 2, |bank| &mut bank.integers)
    }

    unsafe fn get<O: DType + DeviceCopy>(
        &mut self,
        shape: &[usize],
        exec: &ExecutionContext,
        kind: usize,
        select: impl FnOnce(&mut Bank<T>) -> &mut Vec<Arc<Tensor<O>>>,
    ) -> Result<Tensor<O>> {
        let index = self.cursors[kind];
        self.cursors[kind] += 1;
        let tensor = {
            let mut slot = self.lease.workspace.lock().unwrap();
            let workspace = slot.as_mut().unwrap();
            let buffers = select(&mut workspace.banks[self.bank]);
            let added = if index == buffers.len() {
                let _range = crate::profiling::ScopedRange::new(c"qwen3_5.workspace.allocate");
                // Initialize on a miss; later producers overwrite scratch. Scalar
                // bias slots are read-only and retain their initial zero.
                let tensor = Arc::new(api::zeros::<O>(shape).execute(exec)?);
                let bytes = tensor.num_bytes();
                buffers.push(tensor);
                bytes
            } else {
                0
            };
            let tensor = buffers[index].clone();
            ensure!(
                tensor
                    .shape()
                    .iter()
                    .map(|&n| n as usize)
                    .eq(shape.iter().copied()),
                "scratch layout changed"
            );
            workspace.bytes += added;
            if added != 0 {
                let mut state = self.lease.pool.state.lock().unwrap();
                state.metrics.allocated_buffers += 1;
                state.metrics.resident_bytes += added;
            }
            tensor
        };
        // SAFETY: each slot is disjoint within a layer. Reusing a bank between
        // layers is ordered on this submission's stream. Owners retain the
        // exclusive checkout, backing memory and actual context until completion.
        Ok(Tensor::from_foreign(
            Arc::new(Owner {
                tensor: tensor.clone(),
                _lease: self.lease.clone(),
            }),
            tensor.shape().to_vec(),
            tensor.strides().to_vec(),
        ))
    }
}

struct Owner<T: DType, O: DType> {
    tensor: Arc<Tensor<O>>,
    _lease: Arc<Lease<T>>,
}

// SAFETY: the backing tensor keeps its pointer, extent and context stable;
// the lease prevents reuse by another submission while any view survives.
unsafe impl<T: DType, O: DType> DeviceAllocation for Owner<T, O> {
    fn device_ptr(&self) -> u64 {
        self.tensor.device_pointer().cu_deviceptr()
    }
    fn len_bytes(&self) -> usize {
        self.tensor.num_bytes()
    }
    fn context(&self) -> Arc<Context> {
        self.tensor.context().clone()
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
    use futures::{poll, FutureExt};
    use std::sync::atomic::Ordering;

    #[tokio::test]
    async fn banks_reuse_storage_without_aliasing_hidden_or_other_mixers() -> Result<()> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = cuda_async::device_context::global_policy(0)?;
        let pool = Arc::new(WorkspacePool::<f32>::default());
        with_context(|exec| {
            value(unsafe {
                (|| -> Result<()> {
                    let mut scratch = pool.acquire(1, exec)?;
                    let hidden = scratch.activation(&[8], exec)?;
                    let residual = scratch.activation(&[8], exec)?;
                    scratch.layer(true);
                    let delta = scratch.activation(&[8], exec)?;
                    let delta_ptr = delta.device_pointer().cu_deviceptr();
                    drop(delta);
                    scratch.layer(true);
                    assert_eq!(
                        scratch
                            .activation(&[8], exec)?
                            .device_pointer()
                            .cu_deviceptr(),
                        delta_ptr
                    );
                    scratch.layer(false);
                    let attention = scratch.activation(&[4], exec)?;
                    assert_ne!(attention.device_pointer().cu_deviceptr(), delta_ptr);
                    assert_ne!(hidden.device_pointer().cu_deviceptr(), delta_ptr);
                    assert_ne!(residual.device_pointer().cu_deviceptr(), delta_ptr);
                    scratch.head();
                    let last = scratch.activation(&[8], exec)?;
                    assert_ne!(
                        last.device_pointer().cu_deviceptr(),
                        hidden.device_pointer().cu_deviceptr()
                    );
                    Ok(())
                })()
            })
        })
        .schedule(&policy)?
        .await??;
        assert_eq!(pool.metrics().allocated_buffers, 5);
        assert_eq!(pool.metrics().resident_bytes, (8 + 8 + 8 + 4 + 8) * 4);
        assert_eq!(pool.metrics().in_use_workspaces, 0);
        Ok(())
    }

    #[tokio::test]
    async fn gated_checkouts_survive_completion_and_escaped_views() -> Result<()> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let context = Context::new(0)?;
        let policy: Arc<dyn SchedulingPolicy> = Arc::new(SingleStream::new(&context)?);
        let stream = policy.next_stream()?;
        let pool = Arc::new(WorkspacePool::<f32>::default());
        // Warm two independent checkouts and completion machinery before gating.
        let warm = with_context(|exec| {
            value(unsafe {
                (|| -> Result<Vec<Tensor<f32>>> {
                    (0..2)
                        .map(|_| pool.acquire(1, exec)?.activation(&[8], exec))
                        .collect()
                })()
            })
        })
        .schedule(&policy)?
        .await??;
        drop(warm);
        SubmittedGpuWorkWithCompletion::submit(&policy, |_| ())
            .await?
            .gpu_completion
            .await?;
        assert_eq!(pool.metrics().cached_workspaces, 2);
        let first_gate = Gate::arm(&stream)?;
        let first_expired = first_gate.expired.clone();
        let retained = pool.clone();
        let first = SubmittedGpuWorkWithCompletion::submit(&policy, move |exec| {
            unsafe { retained.acquire(1, exec)?.activation(&[8], exec) }
                .map_err(|e| ModelError::Execution(e.to_string()))
        })
        .await?;
        let first_tensor = first.output?;
        let mut first_done = first.gpu_completion.boxed();
        assert!(poll!(&mut first_done).is_pending());
        let second_gate = Gate::arm(&stream)?;
        let second_expired = second_gate.expired.clone();
        let retained = pool.clone();
        let second = SubmittedGpuWorkWithCompletion::submit(&policy, move |exec| {
            unsafe { retained.acquire(1, exec)?.activation(&[8], exec) }
                .map_err(|e| ModelError::Execution(e.to_string()))
        })
        .await?;
        let second_tensor = second.output?;
        let mut second_done = second.gpu_completion.boxed();
        assert!(poll!(&mut second_done).is_pending());
        assert_ne!(
            first_tensor.device_pointer().cu_deviceptr(),
            second_tensor.device_pointer().cu_deviceptr()
        );
        assert_eq!(pool.metrics().allocated_buffers, 2);
        assert_eq!(pool.metrics().in_use_workspaces, 2);
        drop(first_gate);
        first_done.await?;
        assert!(
            !first_expired.load(Ordering::Acquire),
            "first GPU gate expired"
        );
        assert!(poll!(&mut second_done).is_pending());
        assert_eq!(
            pool.metrics().in_use_workspaces,
            2,
            "escaped view must retain checkout"
        );
        drop(first_tensor);
        assert_eq!(pool.metrics().in_use_workspaces, 1);
        drop(second_gate);
        second_done.await?;
        assert!(
            !second_expired.load(Ordering::Acquire),
            "second GPU gate expired"
        );
        drop(second_tensor);
        assert_eq!(pool.metrics().in_use_workspaces, 0);
        assert_eq!(pool.metrics().cached_workspaces, 2);
        Ok(())
    }
}
