use std::sync::{Arc, Mutex};

use cuda_async::{
    device_buffer::DeviceAllocation,
    device_operation::{DeviceOp, ExecutionContext},
};
use cutile::{api, tensor::Tensor};

use super::{ModelError, INPUT_DIM};

// Bound retained memory across shape changes; in-flight checkouts are independent.
const MAX_IDLE_WORKSPACES: usize = 4;

struct Workspace {
    rows: usize,
    device_id: usize,
    buffers: Buffers<Arc<Tensor<f32>>>,
}

/// One writable tensor handle per scratch allocation, private to this submission.
pub(super) struct Buffers<T> {
    pub x: T,
    pub q: T,
    pub attention: T,
    pub residual: T,
    pub hidden: T,
    pub output: T,
}

#[derive(Default)]
pub(super) struct WorkspacePool {
    idle: Mutex<Vec<Workspace>>,
}

impl WorkspacePool {
    /// Borrow exact-shape scratch for one submission. Never share a checkout between
    /// batches. Allocation/initialization occurs only on a miss, normally in warmup.
    pub unsafe fn acquire(
        self: &Arc<Self>,
        rows: usize,
        exec: &ExecutionContext,
    ) -> Result<Buffers<Tensor<f32>>, ModelError> {
        let cached = {
            let mut idle = self.idle.lock().unwrap();
            idle.iter()
                .position(|w| w.rows == rows && w.device_id == exec.get_device_id())
                .map(|index| idle.swap_remove(index))
        };
        let workspace = match cached {
            Some(workspace) => workspace,
            None => Workspace {
                rows,
                device_id: exec.get_device_id(),
                buffers: Buffers {
                    x: Arc::new(api::zeros(&[rows, INPUT_DIM]).execute(exec)?),
                    q: Arc::new(api::zeros(&[rows, INPUT_DIM]).execute(exec)?),
                    attention: Arc::new(api::zeros(&[rows, INPUT_DIM]).execute(exec)?),
                    residual: Arc::new(api::zeros(&[rows, INPUT_DIM]).execute(exec)?),
                    hidden: Arc::new(api::zeros(&[rows, INPUT_DIM]).execute(exec)?),
                    output: Arc::new(api::zeros(&[rows]).execute(exec)?),
                },
            },
        };
        let lease = Arc::new(WorkspaceLease {
            workspace: Some(workspace),
            pool: self.clone(),
        });
        // Retain even if submission fails partway or its future is cancelled.
        // exec does not release owners until GPU completion is established.
        exec.retain(lease.clone())?;
        let buffers = &lease.workspace.as_ref().unwrap().buffers;
        // SAFETY: create exactly one handle per allocation for this checkout.
        // Original tensors are only ownership tokens, never used for GPU accesses.
        // All handle accesses are ordered on exec, and each producer overwrites all
        // rows (including padding) before consumers read them. Each handle also holds
        // the lease, so even escaped handles prevent recycling the workspace.
        Ok(Buffers {
            x: handle(&lease, &buffers.x),
            q: handle(&lease, &buffers.q),
            attention: handle(&lease, &buffers.attention),
            residual: handle(&lease, &buffers.residual),
            hidden: handle(&lease, &buffers.hidden),
            output: handle(&lease, &buffers.output),
        })
    }
}

struct WorkspaceLease {
    workspace: Option<Workspace>,
    pool: Arc<WorkspacePool>,
}

impl Drop for WorkspaceLease {
    fn drop(&mut self) {
        let mut idle = self.pool.idle.lock().unwrap();
        if idle.len() == MAX_IDLE_WORKSPACES {
            idle.remove(0);
        }
        idle.push(self.workspace.take().unwrap());
    }
}

/// Keep both the backing allocation and its exclusive workspace checkout alive.
struct BufferOwner {
    tensor: Arc<Tensor<f32>>,
    _lease: Arc<WorkspaceLease>,
}

// SAFETY: tensor owns a fixed-size allocation with a stable device pointer, byte
// length, and ordinal. Holding it prevents the allocation from being freed.
unsafe impl DeviceAllocation for BufferOwner {
    fn device_ptr(&self) -> u64 {
        self.tensor.device_pointer().cu_deviceptr()
    }

    fn len_bytes(&self) -> usize {
        self.tensor.num_bytes()
    }

    fn device_id(&self) -> usize {
        self.tensor.device_id()
    }
}

/// Only acquire calls this, once per backing allocation in an exclusive checkout.
unsafe fn handle(lease: &Arc<WorkspaceLease>, tensor: &Arc<Tensor<f32>>) -> Tensor<f32> {
    let shape = tensor.shape().to_vec();
    let strides = tensor.strides().to_vec();
    Tensor::from_foreign(
        Arc::new(BufferOwner {
            tensor: tensor.clone(),
            _lease: lease.clone(),
        }),
        shape,
        strides,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuda_async::{
        device_context::global_policy,
        device_operation::{value, with_context},
    };

    #[tokio::test]
    async fn scratch_is_not_recycled_until_submission_and_handles_finish() -> Result<(), ModelError>
    {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = global_policy(0)?;
        let pool = Arc::new(WorkspacePool::default());
        let first_pointer = with_context(|exec| {
            value((|| -> Result<_, ModelError> {
                // SAFETY: the outer future establishes completion before pool reuse.
                let buffers = unsafe { pool.acquire(4, exec)? };
                let pointer = buffers.x.device_pointer().cu_deviceptr();
                drop(buffers);
                assert!(pool.idle.lock().unwrap().is_empty());
                Ok(pointer)
            })())
        })
        .schedule(&policy)?
        .await??;
        assert_eq!(pool.idle.lock().unwrap().len(), 1);
        let (a, b) = with_context(|exec| {
            // SAFETY: these handles are retained by their owner and the submission.
            value(unsafe {
                (|| -> Result<_, ModelError> {
                    Ok((pool.acquire(4, exec)?, pool.acquire(4, exec)?))
                })()
            })
        })
        .schedule(&policy)?
        .await??;
        assert_eq!(a.x.device_pointer().cu_deviceptr(), first_pointer);
        assert_ne!(
            a.x.device_pointer().cu_deviceptr(),
            b.x.device_pointer().cu_deviceptr()
        );
        assert!(pool.idle.lock().unwrap().is_empty());
        drop(a);
        assert_eq!(pool.idle.lock().unwrap().len(), 1);
        drop(b);
        assert_eq!(pool.idle.lock().unwrap().len(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn scratch_matches_shape_and_survives_submission_failure() -> Result<(), ModelError> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = global_policy(0)?;
        let pool = Arc::new(WorkspacePool::default());
        for rows in [4, 8, 4, 12, 16, 20] {
            let result = with_context(|exec| {
                value((|| -> Result<(), ModelError> {
                    // SAFETY: exec retains ownership through failure cleanup.
                    let buffers = unsafe { pool.acquire(rows, exec)? };
                    assert_eq!(buffers.x.shape(), &[rows as i32, INPUT_DIM as i32]);
                    assert_eq!(buffers.output.shape(), &[rows as i32]);
                    drop(buffers);
                    Err(ModelError::Execution(
                        "injected failure after checkout".into(),
                    ))
                })())
            })
            .schedule(&policy)?
            .await?;
            assert!(matches!(result, Err(ModelError::Execution(_))));
            assert!(pool.idle.lock().unwrap().len() <= MAX_IDLE_WORKSPACES);
        }
        assert_eq!(pool.idle.lock().unwrap().len(), MAX_IDLE_WORKSPACES);
        Ok(())
    }
}
