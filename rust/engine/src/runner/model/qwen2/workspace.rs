use std::sync::{Arc, Mutex};

use cuda_async::{
    device_buffer::DeviceAllocation,
    device_operation::{DeviceOp, ExecutionContext},
};
use cutile::{
    cuda_core::{sys, DType, IntoResult},
    tensor::Tensor,
};

use super::{
    engine::Qwen2Config,
    kernels::ARGMAX_TILE,
    metrics::{next_instance, WorkspaceInstruments, WorkspaceMetrics},
};
use crate::ModelError;

// Keep the usual prefill/decode pair; never cache a checkout still in flight.
const MAX_IDLE_WORKSPACES: usize = 2;
/// Bound attention/projection split variants in shared decode scratch.
/// Attention packets hold 64 weighted values plus max/sum.
pub(super) const MAX_DECODE_SPLITS: usize = 16;
pub(super) const KV_PARTIAL_WIDTH: usize = 66;

pub(super) struct Buffers<T: DType> {
    pub(super) allocation_bytes: usize,
    pub(super) x: Tensor<T>,
    pub(super) residual: Tensor<T>,
    pub(super) norm: Tensor<T>,
    pub(super) q: Tensor<T>,
    pub(super) rotated_q: Tensor<T>,
    pub(super) attention: Tensor<T>,
    /// FP32 attention states, then skinny projection partials, reused in stream order.
    /// Absent when a forward contains prefill chunks.
    pub(super) decode_partials: Option<Tensor<f32>>,
    pub(super) k: Tensor<T>,
    pub(super) rotated_k: Tensor<T>,
    pub(super) v: Tensor<T>,
    pub(super) gate: Tensor<T>,
    pub(super) up: Tensor<T>,
    pub(super) activated: Tensor<T>,
    pub(super) final_norm: Tensor<T>,
    pub(super) logits: Tensor<f32>,
    pub(super) maxima: Tensor<f32>,
    pub(super) indices: Tensor<i32>,
    pub(super) winner: Tensor<i32>,
}

struct Workspace<T: DType = f32> {
    rows: usize,
    batch_size: usize,
    /// Activation storage follows the model dtype; logits/maxima stay FP32.
    activations: Arc<Tensor<T>>,
    floats: Arc<Tensor<f32>>,
    integers: Arc<Tensor<i32>>,
}

/// Stores only idle workspaces; checked-out storage is owned by a lease and
/// returns here when its last GPU/handle owner releases it, without an in-use list.
pub(super) struct WorkspacePool<T: DType = f32> {
    state: Mutex<PoolState<T>>,
}

impl<T: DType> Default for WorkspacePool<T> {
    fn default() -> Self {
        Self {
            state: Mutex::new(PoolState::default()),
        }
    }
}

struct PoolState<T: DType = f32> {
    idle: Vec<Workspace<T>>,
    metrics: WorkspaceMetrics,
    instruments: WorkspaceInstruments,
}

impl<T: DType> Default for PoolState<T> {
    fn default() -> Self {
        let model_instance = next_instance();
        Self {
            idle: Vec::new(),
            metrics: WorkspaceMetrics {
                model_instance,
                ..Default::default()
            },
            instruments: WorkspaceInstruments::new(model_instance),
        }
    }
}

impl<T: DType> Drop for PoolState<T> {
    fn drop(&mut self) {
        self.instruments.close();
    }
}

impl<T: DType> PoolState<T> {
    fn checkout(&mut self, bytes: usize) {
        self.metrics.checkout(bytes);
        self.instruments.checkout(bytes);
    }

    fn publish(&self) {
        self.instruments.publish(&self.metrics);
    }

    /// The caller publishes after the complete ownership transfer, not mid-eviction.
    fn evict(&mut self, index: usize) {
        let workspace = self.idle.remove(index);
        self.metrics.idle_bytes -= workspace.bytes();
        self.metrics.idle_workspaces -= 1;
        self.instruments.evictions.increment(1);
        self.metrics.update_peaks();
    }

    fn clear_idle(&mut self) {
        while !self.idle.is_empty() {
            self.evict(self.idle.len() - 1);
        }
    }
}

impl<T: DType> Workspace<T> {
    fn bytes(&self) -> usize {
        self.activations.num_bytes() + self.floats.num_bytes() + self.integers.num_bytes()
    }
}

impl<T: DType> WorkspacePool<T> {
    pub(super) fn metrics(&self) -> WorkspaceMetrics {
        self.state.lock().unwrap().metrics
    }

    /// One exclusive checkout per forward. Disjoint buffers are reused between layers
    /// on exec's stream; producers overwrite every valid element before consumers read.
    pub(super) unsafe fn acquire(
        self: &Arc<Self>,
        rows: usize,
        batch_size: usize,
        config: &Qwen2Config,
        exec: &ExecutionContext,
    ) -> Result<Buffers<T>, ModelError> {
        let h = config.hidden_size;
        let k = config.num_key_value_heads * 64;
        let i = config.intermediate_size;
        let vocab = config.vocab_size;
        let blocks = vocab.div_ceil(ARGMAX_TILE);
        let (all_floats, integers) = element_counts(rows, batch_size, config)?;
        let floats = batch_size * (vocab + blocks)
            + decode_partial_elements(rows, batch_size, config).unwrap();
        let activations = all_floats - floats;
        let required = activations * size_of::<T>() + (floats + integers) * 4;
        let cached = {
            let mut state = self.state.lock().unwrap();
            let index = state.idle.iter().position(|w| {
                w.rows == rows
                    && w.batch_size == batch_size
                    && w.floats.device_id() == exec.get_device_id()
            });
            let cached = match index {
                Some(index) => {
                    let workspace = state.idle.swap_remove(index);
                    state.metrics.idle_bytes -= workspace.bytes();
                    state.metrics.idle_workspaces -= 1;
                    state.instruments.cache_hits.increment(1);
                    state.checkout(required);
                    Some(workspace)
                }
                None => {
                    state.instruments.cache_misses.increment(1);
                    if state.idle.len() == MAX_IDLE_WORKSPACES {
                        state.clear_idle();
                    }
                    None
                }
            };
            state.publish();
            cached
        };
        let cache_hit = cached.is_some();
        let workspace = match cached {
            Some(workspace) => workspace,
            None => {
                let allocated = (|| -> Result<Workspace<T>, ModelError> {
                    if check_available(required, exec).is_err() {
                        {
                            let mut state = self.state.lock().unwrap();
                            state.clear_idle();
                            state.publish();
                        }
                        check_available(required, exec)?;
                    }
                    // SAFETY: producers overwrite every valid element before use.
                    Ok(Workspace {
                        rows,
                        batch_size,
                        activations: Arc::new(
                            Tensor::<T>::uninitialized(activations)
                                .execute(exec)?
                                .assume_init(),
                        ),
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
                    })
                })();
                let mut state = self.state.lock().unwrap();
                match allocated {
                    Ok(workspace) => {
                        state.instruments.workspace_allocations.increment(1);
                        state.checkout(required);
                        state.publish();
                        workspace
                    }
                    Err(error) => {
                        state.instruments.checkout_failures.increment(1);
                        state.publish();
                        return Err(error);
                    }
                }
            }
        };
        tracing::debug!(target: "engine::qwen2", event = "workspace_checkout", rows, batch_size,
            scratch_bytes = required, cache_hit);
        let lease = Arc::new(WorkspaceLease {
            workspace: Some(workspace),
            pool: self.clone(),
        });
        // Errors/cancellation must also establish completion before recycling storage.
        exec.retain(lease.clone())?;
        let workspace = lease.workspace.as_ref().unwrap();
        let mut offset = 0;
        let mut integer_offset = 0;
        let mut float_offset = 0;
        let buffers = Buffers {
            allocation_bytes: required,
            x: handle(&lease, &workspace.activations, &mut offset, &[rows, h]),
            residual: handle(&lease, &workspace.activations, &mut offset, &[rows, h]),
            norm: handle(&lease, &workspace.activations, &mut offset, &[rows, h]),
            q: handle(&lease, &workspace.activations, &mut offset, &[rows, h]),
            rotated_q: handle(&lease, &workspace.activations, &mut offset, &[rows, h]),
            attention: handle(&lease, &workspace.activations, &mut offset, &[rows, h]),
            k: handle(&lease, &workspace.activations, &mut offset, &[rows, k]),
            rotated_k: handle(&lease, &workspace.activations, &mut offset, &[rows, k]),
            v: handle(&lease, &workspace.activations, &mut offset, &[rows, k]),
            gate: handle(&lease, &workspace.activations, &mut offset, &[rows, i]),
            up: handle(&lease, &workspace.activations, &mut offset, &[rows, i]),
            activated: handle(&lease, &workspace.activations, &mut offset, &[rows, i]),
            final_norm: handle(
                &lease,
                &workspace.activations,
                &mut offset,
                &[batch_size, h],
            ),
            logits: handle(
                &lease,
                &workspace.floats,
                &mut float_offset,
                &[batch_size, vocab],
            ),
            // Keep decode scratch addresses independent of the argmax partial count.
            decode_partials: (rows == batch_size).then(|| {
                handle(
                    &lease,
                    &workspace.floats,
                    &mut float_offset,
                    &[
                        batch_size * config.num_attention_heads * MAX_DECODE_SPLITS,
                        KV_PARTIAL_WIDTH,
                    ],
                )
            }),
            maxima: handle(
                &lease,
                &workspace.floats,
                &mut float_offset,
                &[batch_size, blocks],
            ),
            indices: handle(
                &lease,
                &workspace.integers,
                &mut integer_offset,
                &[batch_size, blocks],
            ),
            winner: handle(
                &lease,
                &workspace.integers,
                &mut integer_offset,
                &[batch_size],
            ),
        };
        debug_assert_eq!(offset, activations);
        debug_assert_eq!(float_offset, floats);
        debug_assert_eq!(integer_offset, integers);
        Ok(buffers)
    }
}

/// No per-layer allocation: reserve bounded split states only for one-token/request steps.
fn decode_partial_elements(rows: usize, batch_size: usize, config: &Qwen2Config) -> Option<usize> {
    if rows == batch_size {
        batch_size
            .checked_mul(config.num_attention_heads)?
            .checked_mul(MAX_DECODE_SPLITS * KV_PARTIAL_WIDTH)
    } else {
        Some(0)
    }
}

/// Linear in packed suffix length; logits/argmax storage also scales with batch size.
/// Decoder depth does not affect scratch capacity.
fn element_counts(
    rows: usize,
    batch_size: usize,
    config: &Qwen2Config,
) -> Result<(usize, usize), ModelError> {
    let per_row =
        6 * config.hidden_size + 3 * config.num_key_value_heads * 64 + 3 * config.intermediate_size;
    let blocks = config.vocab_size.div_ceil(ARGMAX_TILE);
    let invalid = || ModelError::InvalidInput("Qwen workspace shape exceeds tensor limits".into());
    if batch_size == 0 || rows < batch_size {
        return Err(invalid());
    }
    let floats = batch_size
        .checked_mul(config.hidden_size + config.vocab_size + blocks)
        .and_then(|tail| rows.checked_mul(per_row)?.checked_add(tail))
        .and_then(|n| n.checked_add(decode_partial_elements(rows, batch_size, config)?))
        .filter(|&n| n <= i32::MAX as usize)
        .ok_or_else(invalid)?;
    let integers = batch_size
        .checked_mul(blocks + 1)
        .filter(|&n| n <= i32::MAX as usize)
        .ok_or_else(invalid)?;
    Ok((floats, integers))
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

struct WorkspaceLease<T: DType = f32> {
    workspace: Option<Workspace<T>>,
    pool: Arc<WorkspacePool<T>>,
}

impl<T: DType> Drop for WorkspaceLease<T> {
    fn drop(&mut self) {
        let workspace = self.workspace.take().unwrap();
        let bytes = workspace.bytes();
        let mut state = self.pool.state.lock().unwrap();
        state.metrics.in_use_bytes -= bytes;
        state.metrics.in_use_workspaces -= 1;
        if state.idle.len() == MAX_IDLE_WORKSPACES {
            state.evict(0);
        }
        state.metrics.idle_bytes += bytes;
        state.metrics.idle_workspaces += 1;
        state.idle.push(workspace);
        state.metrics.update_peaks();
        state.publish();
        let metrics = state.metrics;
        drop(state);
        tracing::debug!(target: "engine::qwen2", event = "workspace_returned",
            in_use_bytes = metrics.in_use_bytes, idle_bytes = metrics.idle_bytes,
            resident_bytes = metrics.resident_bytes, peak_in_use_bytes = metrics.peak_in_use_bytes,
            peak_idle_bytes = metrics.peak_idle_bytes, peak_resident_bytes = metrics.peak_resident_bytes);
    }
}

struct BufferOwner<T: DType, E: DType> {
    tensor: Arc<Tensor<T>>,
    offset: usize,
    len: usize,
    _lease: Arc<WorkspaceLease<E>>,
}

// SAFETY: each owner describes a disjoint, bounded region of a stable allocation.
// The lease prevents reuse until every handle and GPU access has been released.
unsafe impl<T: DType, E: DType> DeviceAllocation for BufferOwner<T, E> {
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
unsafe fn handle<T: DType, E: DType>(
    lease: &Arc<WorkspaceLease<E>>,
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

    /// Capture absolute resident snapshots, including intermediate publications.
    #[derive(Clone, Default)]
    struct ResidentRecorder {
        samples: Arc<Mutex<Vec<f64>>>,
    }

    impl metrics::GaugeFn for ResidentRecorder {
        fn increment(&self, _value: f64) {
            unreachable!("resource snapshots use set");
        }
        fn decrement(&self, _value: f64) {
            unreachable!("resource snapshots use set");
        }
        fn set(&self, value: f64) {
            self.samples.lock().unwrap().push(value);
        }
    }

    impl metrics::Recorder for ResidentRecorder {
        fn describe_counter(
            &self,
            _: metrics::KeyName,
            _: Option<metrics::Unit>,
            _: metrics::SharedString,
        ) {
        }
        fn describe_gauge(
            &self,
            _: metrics::KeyName,
            _: Option<metrics::Unit>,
            _: metrics::SharedString,
        ) {
        }
        fn describe_histogram(
            &self,
            _: metrics::KeyName,
            _: Option<metrics::Unit>,
            _: metrics::SharedString,
        ) {
        }
        fn register_counter(
            &self,
            _: &metrics::Key,
            _: &metrics::Metadata<'_>,
        ) -> metrics::Counter {
            metrics::Counter::noop()
        }
        fn register_histogram(
            &self,
            _: &metrics::Key,
            _: &metrics::Metadata<'_>,
        ) -> metrics::Histogram {
            metrics::Histogram::noop()
        }
        fn register_gauge(&self, key: &metrics::Key, _: &metrics::Metadata<'_>) -> metrics::Gauge {
            if key.name() == "qwen2_workspace_resident_bytes" {
                metrics::Gauge::from_arc(Arc::new(self.clone()))
            } else {
                metrics::Gauge::noop()
            }
        }
    }

    #[tokio::test]
    async fn full_idle_cache_return_publishes_only_complete_resident_state(
    ) -> Result<(), ModelError> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = global_policy(0)?;
        let config = config();
        let recorder = ResidentRecorder::default();
        let pool =
            metrics::with_local_recorder(&recorder, || Arc::new(WorkspacePool::<f32>::default()));
        let mut buffers = with_context(|exec| {
            // SAFETY: exclusive checkouts survive outer completion through their handles.
            value(
                (0..3)
                    .map(|_| unsafe { pool.acquire(3, 1, &config, exec) })
                    .collect::<Result<Vec<_>, _>>(),
            )
        })
        .schedule(&policy)?
        .await??;
        let bytes = pool.metrics().peak_forward_bytes;
        drop(buffers.pop());
        drop(buffers.pop());
        assert_eq!(pool.metrics().resident_bytes, bytes * 3);
        recorder.samples.lock().unwrap().clear();
        // Evict one of two idle workspaces and cache the returning third one.
        drop(buffers);
        let samples = recorder.samples.lock().unwrap().clone();
        assert_eq!(samples, vec![(bytes * 2) as f64]);
        assert_eq!(pool.metrics().resident_bytes, bytes * 2);
        Ok(())
    }

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
        let (floats, integers) = element_counts(4096, 1, &config).unwrap();
        assert_eq!((floats + integers) * 4, 334058804);
        config.num_hidden_layers = 48;
        assert_eq!(
            element_counts(4096, 1, &config).unwrap(),
            (floats, integers)
        );
        assert!(element_counts(usize::MAX, 1, &config).is_err());
        let (packed_floats, packed_ints) = element_counts(4096, 3, &config).unwrap();
        let tail_bytes = 4
            * (config.hidden_size
                + config.vocab_size
                + 2 * config.vocab_size.div_ceil(ARGMAX_TILE)
                + 1);
        assert_eq!(
            (packed_floats + packed_ints - floats - integers) * 4,
            2 * tail_bytes
        );
        assert!(element_counts(1, 2, &config).is_err());
        assert!(element_counts(1, 0, &config).is_err());
    }

    #[tokio::test]
    async fn workspace_cache_distinguishes_batch_size_with_equal_token_count(
    ) -> Result<(), ModelError> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = global_policy(0)?;
        let config = config();
        let pool = Arc::new(WorkspacePool::<f32>::default());
        let mut pointers = Vec::new();
        for batch_size in [1, 3, 1, 3] {
            let buffers = with_context(|exec| {
                // SAFETY: exclusive handles keep the workspace alive through outer completion.
                value(unsafe { pool.acquire(3, batch_size, &config, exec) })
            })
            .schedule(&policy)?
            .await??;
            assert_eq!(
                buffers.logits.shape(),
                [batch_size as i32, config.vocab_size as i32]
            );
            pointers.push(buffers.x.device_pointer().cu_deviceptr());
            drop(buffers);
        }
        assert_ne!(pointers[0], pointers[1]);
        assert_eq!(pointers[0], pointers[2]);
        assert_eq!(pointers[1], pointers[3]);
        assert_eq!(pool.metrics().in_use_bytes, 0);
        Ok(())
    }

    #[tokio::test]
    async fn exclusive_workspace_survives_inflight_access_and_escaped_handles(
    ) -> Result<(), ModelError> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = global_policy(0)?;
        let config = config();
        let recorder = metrics_util::debugging::DebuggingRecorder::new();
        let pool =
            metrics::with_local_recorder(&recorder, || Arc::new(WorkspacePool::<f32>::default()));
        let a = with_context(|exec| {
            value((|| -> Result<_, ModelError> {
                // Oversized scratch is rejected without attempting an allocation.
                assert!(unsafe { check_available(usize::MAX, exec) }.is_err());
                // SAFETY: all handles/allocations remain leased through outer completion.
                let a = unsafe { pool.acquire(3, 1, &config, exec)? };
                let b = unsafe { pool.acquire(3, 1, &config, exec)? };
                assert_ne!(
                    a.x.device_pointer().cu_deviceptr(),
                    b.x.device_pointer().cu_deviceptr()
                );
                drop(b);
                assert!(pool.state.lock().unwrap().idle.is_empty());
                Ok(a)
            })())
        })
        .schedule(&policy)?
        .await??;
        // b has completed; a's escaped handle still prevents its workspace recycling.
        assert_eq!(pool.state.lock().unwrap().idle.len(), 1);
        let pointer = a.x.device_pointer().cu_deviceptr();
        drop(a);
        assert_eq!(pool.state.lock().unwrap().idle.len(), 2);
        let (a, b) = with_context(|exec| {
            // SAFETY: checkouts are exclusive, and completion is awaited.
            value(unsafe {
                (|| -> Result<_, ModelError> {
                    Ok((
                        pool.acquire(3, 1, &config, exec)?,
                        pool.acquire(3, 1, &config, exec)?,
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
                let buffers = unsafe { pool.acquire(3, 1, &config, exec)? };
                drop(buffers);
                assert_eq!(pool.state.lock().unwrap().idle.len(), 1);
                Err(ModelError::Execution("injected failure".into()))
            })())
        })
        .schedule(&policy)?
        .await?;
        assert!(failed.is_err());
        assert_eq!(pool.state.lock().unwrap().idle.len(), 2);
        let metrics = pool.metrics();
        let bytes = metrics.peak_forward_bytes;
        assert_eq!(metrics.in_use_bytes, 0);
        assert_eq!(metrics.peak_in_use_bytes, bytes * 2);
        assert_eq!(metrics.idle_bytes, bytes * 2);
        assert_eq!(metrics.peak_idle_bytes, bytes * 2);
        assert_eq!(metrics.peak_resident_bytes, bytes * 2);
        let exported = recorder.snapshotter().snapshot().into_vec();
        let get = |name: &str| {
            &exported
                .iter()
                .find(|(key, ..)| key.key().name() == name)
                .unwrap()
                .3
        };
        use metrics_util::debugging::DebugValue;
        assert_eq!(
            get("qwen2_workspace_cache_hits_total"),
            &DebugValue::Counter(3)
        );
        assert_eq!(
            get("qwen2_workspace_cache_misses_total"),
            &DebugValue::Counter(2)
        );
        assert_eq!(
            get("qwen2_workspace_allocations_total"),
            &DebugValue::Counter(2)
        );
        assert_eq!(
            get("qwen2_workspace_in_use_bytes"),
            &DebugValue::Gauge(0.0.into())
        );
        assert_eq!(
            get("qwen2_workspace_idle_bytes"),
            &DebugValue::Gauge(((bytes * 2) as f64).into())
        );
        assert_eq!(
            get("qwen2_workspace_forward_bytes"),
            &DebugValue::Histogram(vec![(bytes as f64).into(); 5])
        );
        drop(pool);
        let closed = recorder.snapshotter().snapshot().into_vec();
        for (key, _, _, value) in closed {
            if [
                "qwen2_workspace_idle_bytes",
                "qwen2_workspace_resident_bytes",
            ]
            .contains(&key.key().name())
            {
                assert_eq!(value, DebugValue::Gauge(0.0.into()));
            }
        }
        Ok(())
    }
    #[tokio::test]
    async fn bf16_workspace_accounts_for_mixed_storage_and_reuses_idle() -> Result<(), ModelError> {
        use cutile::core::bf16;
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = global_policy(0)?;
        let config = config();
        let pool = Arc::new(WorkspacePool::<bf16>::default());
        let (all_floats, integers) = element_counts(33, 2, &config)?;
        let fp32 = 2 * (config.vocab_size + config.vocab_size.div_ceil(ARGMAX_TILE));
        let bytes = (all_floats - fp32) * 2 + (fp32 + integers) * 4;
        let mut pointer = None;
        for _ in 0..2 {
            let buffers = with_context(|exec| {
                // SAFETY: no GPU consumer reads these uninitialized diagnostic handles.
                value(unsafe { pool.acquire(33, 2, &config, exec) })
            })
            .schedule(&policy)?
            .await??;
            let current = buffers.x.device_pointer().cu_deviceptr();
            assert_eq!(*pointer.get_or_insert(current), current);
            assert_eq!(buffers.x.num_bytes(), 33 * config.hidden_size * 2);
            assert_eq!(buffers.logits.num_bytes(), 2 * config.vocab_size * 4);
            assert_eq!(pool.metrics().in_use_bytes, bytes);
            drop(buffers);
            assert_eq!(pool.metrics().idle_bytes, bytes);
            assert_eq!(pool.metrics().in_use_bytes, 0);
        }
        Ok(())
    }
}
