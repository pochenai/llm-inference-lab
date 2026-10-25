use std::sync::{Arc, Mutex};

use cuda_async::{
    device_buffer::DeviceAllocation,
    device_operation::{DeviceOp, ExecutionContext},
};
use cutile::{
    cuda_core::{sys, DType, IntoResult},
    tensor::Tensor,
};

use super::engine::Qwen2Config;
use crate::ModelError;

// Keep the usual prefill/decode pair; never cache a checkout still in flight.
const MAX_IDLE_WORKSPACES: usize = 2;

pub(super) struct Buffers {
    pub(super) x: Tensor<f32>,
    pub(super) residual: Tensor<f32>,
    pub(super) norm: Tensor<f32>,
    pub(super) q: Tensor<f32>,
    pub(super) rotated_q: Tensor<f32>,
    pub(super) attention: Tensor<f32>,
    pub(super) k: Tensor<f32>,
    pub(super) rotated_k: Tensor<f32>,
    pub(super) v: Tensor<f32>,
    pub(super) gate: Tensor<f32>,
    pub(super) up: Tensor<f32>,
    pub(super) activated: Tensor<f32>,
    pub(super) final_norm: Tensor<f32>,
    pub(super) logits: Tensor<f32>,
    pub(super) maxima: Tensor<f32>,
    pub(super) indices: Tensor<i32>,
    pub(super) winner: Tensor<i32>,
}

struct Workspace {
    rows: usize,
    floats: Arc<Tensor<f32>>,
    integers: Arc<Tensor<i32>>,
}

/// Stores only idle workspaces; checked-out storage is owned by a lease and
/// returns here when its last GPU/handle owner releases it, without an in-use list.
#[derive(Default)]
pub(super) struct WorkspacePool {
    idle: Mutex<Vec<Workspace>>,
}

impl WorkspacePool {
    /// One exclusive checkout per forward. Disjoint buffers are reused between layers
    /// on exec's stream; producers overwrite every valid element before consumers read.
    pub(super) unsafe fn acquire(
        self: &Arc<Self>,
        rows: usize,
        config: &Qwen2Config,
        exec: &ExecutionContext,
    ) -> Result<Buffers, ModelError> {
        let h = config.hidden_size;
        let k = config.num_key_value_heads * 64;
        let i = config.intermediate_size;
        let vocab = config.vocab_size;
        let blocks = vocab.div_ceil(256);
        let (floats, integers) = element_counts(rows, config)?;
        let cached = {
            let mut idle = self.idle.lock().unwrap();
            let index = idle
                .iter()
                .position(|w| w.rows == rows && w.floats.device_id() == exec.get_device_id());
            let cached = index.map(|index| idle.swap_remove(index));
            // Bound stale shapes while keeping the usual prefill/decode pair.
            if cached.is_none() && idle.len() == MAX_IDLE_WORKSPACES {
                idle.clear();
            }
            cached
        };
        let workspace = match cached {
            Some(workspace) => workspace,
            None => {
                let required = (floats + integers) * 4;
                if check_available(required, exec).is_err() {
                    // Prefer this forward over retaining idle shapes under pressure.
                    self.idle.lock().unwrap().clear();
                    check_available(required, exec)?;
                }
                Workspace {
                    rows,
                    // SAFETY: all f32/i32 bit patterns are valid. No GPU consumer may
                    // read a scratch region until its producer has completely written it.
                    floats: Arc::new(
                        Tensor::<f32>::uninitialized(floats)
                            .execute(exec)?
                            .assume_init(),
                    ),
                    integers: Arc::new(
                        Tensor::<i32>::uninitialized(integers)
                            .execute(exec)?
                            .assume_init(),
                    ),
                }
            }
        };
        let lease = Arc::new(WorkspaceLease {
            workspace: Some(workspace),
            pool: self.clone(),
        });
        // Errors/cancellation must also establish completion before recycling storage.
        exec.retain(lease.clone())?;
        let workspace = lease.workspace.as_ref().unwrap();
        let mut offset = 0;
        let mut integer_offset = 0;
        let buffers = Buffers {
            x: handle(&lease, &workspace.floats, &mut offset, &[rows, h]),
            residual: handle(&lease, &workspace.floats, &mut offset, &[rows, h]),
            norm: handle(&lease, &workspace.floats, &mut offset, &[rows, h]),
            q: handle(&lease, &workspace.floats, &mut offset, &[rows, h]),
            rotated_q: handle(&lease, &workspace.floats, &mut offset, &[rows, h]),
            attention: handle(&lease, &workspace.floats, &mut offset, &[rows, h]),
            k: handle(&lease, &workspace.floats, &mut offset, &[rows, k]),
            rotated_k: handle(&lease, &workspace.floats, &mut offset, &[rows, k]),
            v: handle(&lease, &workspace.floats, &mut offset, &[rows, k]),
            gate: handle(&lease, &workspace.floats, &mut offset, &[rows, i]),
            up: handle(&lease, &workspace.floats, &mut offset, &[rows, i]),
            activated: handle(&lease, &workspace.floats, &mut offset, &[rows, i]),
            final_norm: handle(&lease, &workspace.floats, &mut offset, &[1, h]),
            logits: handle(&lease, &workspace.floats, &mut offset, &[1, vocab]),
            maxima: handle(&lease, &workspace.floats, &mut offset, &[blocks]),
            indices: handle(&lease, &workspace.integers, &mut integer_offset, &[blocks]),
            winner: handle(&lease, &workspace.integers, &mut integer_offset, &[1]),
        };
        debug_assert_eq!(offset, floats);
        debug_assert_eq!(integer_offset, integers);
        Ok(buffers)
    }
}

/// Linear in suffix length, independent of decoder depth. Count the two backing
/// allocations exactly, including last-row logits and both argmax stages.
fn element_counts(rows: usize, config: &Qwen2Config) -> Result<(usize, usize), ModelError> {
    let per_row =
        6 * config.hidden_size + 3 * config.num_key_value_heads * 64 + 3 * config.intermediate_size;
    let blocks = config.vocab_size.div_ceil(256);
    let floats = rows
        .checked_mul(per_row)
        .and_then(|n| n.checked_add(config.hidden_size + config.vocab_size + blocks))
        .filter(|&n| n <= i32::MAX as usize)
        .ok_or_else(|| {
            ModelError::InvalidInput("Qwen workspace shape exceeds tensor limits".into())
        })?;
    Ok((floats, blocks + 1))
}

/// Best-effort scratch admission after weights/KV allocation; concurrent external
/// allocations may still race this snapshot, so allocation errors remain fallible.
unsafe fn check_available(required: usize, exec: &ExecutionContext) -> Result<(), ModelError> {
    exec.get_cuda_stream()
        .device()
        .bind_to_thread()
        .map_err(cuda_async::error::DeviceError::from)?;
    let (mut free, mut total) = (0, 0);
    sys::cuMemGetInfo_v2(&mut free, &mut total)
        .result()
        .map_err(cuda_async::error::DeviceError::from)?;
    let reusable = match exec.get_pool() {
        Some(pool) => {
            let stats = pool
                .mem_stats()
                .map_err(cuda_async::error::DeviceError::from)?;
            stats.reserved_current.saturating_sub(stats.used_current) as usize
        }
        None => 0,
    };
    if required > free.saturating_add(reusable) {
        return Err(ModelError::Execution(format!(
            "Qwen scratch requires {required} bytes; {} bytes available after weights/KV",
            free.saturating_add(reusable)
        )));
    }
    Ok(())
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

struct BufferOwner<T: DType> {
    tensor: Arc<Tensor<T>>,
    offset: usize,
    len: usize,
    _lease: Arc<WorkspaceLease>,
}

// SAFETY: each owner describes a disjoint, bounded region of a stable allocation.
// The lease prevents reuse until every handle and GPU access has been released.
unsafe impl<T: DType> DeviceAllocation for BufferOwner<T> {
    fn device_ptr(&self) -> u64 {
        self.tensor.device_pointer().cu_deviceptr() + (self.offset * size_of::<T>()) as u64
    }
    fn len_bytes(&self) -> usize {
        self.len * size_of::<T>()
    }
    fn device_id(&self) -> usize {
        self.tensor.device_id()
    }
}

/// Called once per disjoint region in an exclusive checkout. Backing tensors only
/// own storage; all GPU accesses use these handles on the checkout's single stream.
unsafe fn handle<T: DType>(
    lease: &Arc<WorkspaceLease>,
    tensor: &Arc<Tensor<T>>,
    offset: &mut usize,
    shape: &[usize],
) -> Tensor<T> {
    let len = shape.iter().product();
    let start = *offset;
    *offset += len;
    assert!(*offset <= tensor.size());
    let mut stride = 1;
    let mut strides = vec![0; shape.len()];
    for (index, &dim) in shape.iter().enumerate().rev() {
        strides[index] = stride as i32;
        stride *= dim;
    }
    Tensor::from_foreign(
        Arc::new(BufferOwner {
            tensor: tensor.clone(),
            offset: start,
            len,
            _lease: lease.clone(),
        }),
        shape.iter().map(|&n| n as i32).collect(),
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

    fn config() -> Qwen2Config {
        serde_json::from_value(serde_json::json!({
            "model_type": "qwen2", "vocab_size": 151936, "hidden_size": 896,
            "intermediate_size": 4864, "num_hidden_layers": 24,
            "num_attention_heads": 14, "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
            "max_position_embeddings": 32768, "tie_word_embeddings": true,
            "eos_token_id": 151645, "hidden_act": "silu"
        }))
        .unwrap()
    }

    #[test]
    fn scratch_budget_depends_on_suffix_length_not_layer_count() {
        let mut config = config();
        let (floats, integers) = element_counts(4096, &config).unwrap();
        assert_eq!((floats + integers) * 4, 334063252);
        config.num_hidden_layers = 48;
        assert_eq!(element_counts(4096, &config).unwrap(), (floats, integers));
        assert!(element_counts(usize::MAX, &config).is_err());
    }

    #[tokio::test]
    async fn exclusive_workspace_survives_inflight_access_and_escaped_handles(
    ) -> Result<(), ModelError> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = global_policy(0)?;
        let config = config();
        let pool = Arc::new(WorkspacePool::default());
        let a = with_context(|exec| {
            value((|| -> Result<_, ModelError> {
                // Oversized scratch is rejected without attempting an allocation.
                assert!(unsafe { check_available(usize::MAX, exec) }.is_err());
                // SAFETY: all handles/allocations remain leased through outer completion.
                let a = unsafe { pool.acquire(3, &config, exec)? };
                let b = unsafe { pool.acquire(3, &config, exec)? };
                assert_ne!(
                    a.x.device_pointer().cu_deviceptr(),
                    b.x.device_pointer().cu_deviceptr()
                );
                drop(b);
                assert!(pool.idle.lock().unwrap().is_empty());
                Ok(a)
            })())
        })
        .schedule(&policy)?
        .await??;
        // b has completed; a's escaped handle still prevents its workspace recycling.
        assert_eq!(pool.idle.lock().unwrap().len(), 1);
        let pointer = a.x.device_pointer().cu_deviceptr();
        drop(a);
        assert_eq!(pool.idle.lock().unwrap().len(), 2);
        let (a, b) = with_context(|exec| {
            // SAFETY: checkouts are exclusive, and completion is awaited.
            value(unsafe {
                (|| -> Result<_, ModelError> {
                    Ok((
                        pool.acquire(3, &config, exec)?,
                        pool.acquire(3, &config, exec)?,
                    ))
                })()
            })
        })
        .schedule(&policy)?
        .await??;
        assert!([
            a.x.device_pointer().cu_deviceptr(),
            b.x.device_pointer().cu_deviceptr()
        ]
        .contains(&pointer));
        drop((a, b));
        // A failure after checkout must return the lease only after cleanup completes.
        let failed = with_context(|exec| {
            value((|| -> Result<(), ModelError> {
                let buffers = unsafe { pool.acquire(3, &config, exec)? };
                drop(buffers);
                assert_eq!(pool.idle.lock().unwrap().len(), 1);
                Err(ModelError::Execution("injected failure".into()))
            })())
        })
        .schedule(&policy)?
        .await?;
        assert!(failed.is_err());
        assert_eq!(pool.idle.lock().unwrap().len(), 2);
        Ok(())
    }
}
