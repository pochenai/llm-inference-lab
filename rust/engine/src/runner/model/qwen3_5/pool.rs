use std::{
    ops::Range,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use cuda_async::{device_operation::ExecutionContext, scheduling_policies::SchedulingPolicy};
use cutile::{
    core::{bf16, ElementType},
    tensor::TensorView,
    DType,
};

use super::{
    config::Qwen3_5TextConfig,
    state::{Qwen3_5Context, Qwen3_5ExecutionView},
    Qwen3_5Engine,
};
use crate::{
    radix_tree::SplitPrefix,
    runner::{
        kv_pool::{
            ExecutionStatePool, KVPool, KVPoolError, KVPoolMetrics, NaiveContextState, NaiveKVPool,
            NaiveKVPoolConfig, NaiveReadView, NaiveWriteView, PrefixLease,
        },
        loader::WeightElement,
    },
};

type Result<T> = std::result::Result<T, KVPoolError>;

/// Token-granular GQA slots plus fixed-size, exclusive DeltaNet state per session.
pub struct Qwen3_5StatePool<T: DType = bf16> {
    owner: Arc<()>,
    kv: Arc<NaiveKVPool<T>>,
    config: Qwen3_5TextConfig,
    recurrent_bytes: usize,
    initialized: Arc<AtomicUsize>,
}

pub struct Qwen3_5SessionState<T: DType = bf16> {
    owner: Arc<()>,
    kv: NaiveContextState,
    context: Option<Qwen3_5Context<T>>,
    initialized: Arc<AtomicUsize>,
}

impl<T: WeightElement + ElementType> Qwen3_5SessionState<T> {
    pub(super) fn validate(
        &self,
        owner: &Arc<()>,
        tokens: &[u32],
        ready_len: usize,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            Arc::ptr_eq(owner, &self.owner),
            "pool belongs to another engine"
        );
        if let Some(context) = &self.context {
            context.validate(owner, tokens)?;
            anyhow::ensure!(
                context.ready_len() == ready_len,
                "hybrid logical progress mismatch"
            );
        } else {
            anyhow::ensure!(ready_len == 0, "uninitialized hybrid history");
        }
        Ok(())
    }
}

impl<T: DType> Drop for Qwen3_5SessionState<T> {
    fn drop(&mut self) {
        if self.context.is_some() {
            self.initialized.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

/// No values exist until recurrent prefix snapshots are supported.
pub enum Qwen3_5PrefixState {}

impl SplitPrefix for Qwen3_5PrefixState {
    fn split_prefix(&mut self, _len: usize) -> Self {
        match *self {}
    }
}

impl<T: WeightElement + ElementType> Qwen3_5StatePool<T> {
    pub async fn new(
        engine: &Qwen3_5Engine<T>,
        config: NaiveKVPoolConfig,
        policy: &Arc<dyn SchedulingPolicy>,
    ) -> Result<Arc<Self>> {
        let expected = engine
            .kv_pool_config(config.max_seq_len)
            .map_err(|e| KVPoolError::InvalidInput(e.to_string()))?;
        if config.num_layers != expected.num_layers || config.head_dim != expected.head_dim {
            return Err(KVPoolError::InvalidInput(
                "GQA pool geometry differs from model".into(),
            ));
        }
        let recurrent_bytes = engine
            .config()
            .text_config
            .state_bytes::<T>(0)
            .filter(|bytes| bytes.checked_mul(config.max_rows).is_some())
            .ok_or_else(|| KVPoolError::InvalidInput("hybrid state budget overflow".into()))?;
        Ok(Arc::new(Self {
            owner: engine.owner.clone(),
            kv: NaiveKVPool::new(config, policy).await?,
            config: engine.config().text_config.clone(),
            recurrent_bytes,
            initialized: Arc::new(AtomicUsize::new(0)),
        }))
    }
    fn validate_state(&self, state: &Qwen3_5SessionState<T>) -> Result<()> {
        if !Arc::ptr_eq(&state.initialized, &self.initialized) {
            return Err(KVPoolError::InvalidInput(
                "hybrid state belongs to another pool".into(),
            ));
        }
        Ok(())
    }
}

fn unsupported_prefix() -> KVPoolError {
    KVPoolError::InvalidInput("Qwen3.5 prefix reuse requires recurrent state snapshots".into())
}

impl<T: WeightElement + ElementType> KVPool for Qwen3_5StatePool<T> {
    type Element = T;
    type ContextState = Qwen3_5SessionState<T>;
    type PrefixState = Qwen3_5PrefixState;
    type ReadView<'a> = NaiveReadView<'a, T>;
    type WriteView<'a> = NaiveWriteView<'a, T>;

    fn supports_prefix_cache(&self) -> bool {
        false
    }
    fn max_seq_len(&self) -> usize {
        self.kv.max_seq_len()
    }
    fn alloc_state(&self) -> Result<Self::ContextState> {
        Ok(Qwen3_5SessionState {
            owner: self.owner.clone(),
            kv: self.kv.alloc_state()?,
            context: None,
            initialized: self.initialized.clone(),
        })
    }
    fn can_append(&self, tokens: usize, new_contexts: usize) -> bool {
        self.kv.can_append(tokens, new_contexts)
    }
    fn prepare_append(
        &self,
        state: &mut Self::ContextState,
        positions: Range<usize>,
        exec: &ExecutionContext,
    ) -> Result<()> {
        self.validate_state(state)?;
        if state.context.is_none() {
            // SAFETY: model/pool dimensions were validated during construction.
            state.context = Some(
                unsafe {
                    Qwen3_5Context::allocate(
                        &self.config,
                        self.owner.clone(),
                        self.kv.max_seq_len(),
                        true,
                        exec,
                    )
                }
                .map_err(|e| KVPoolError::Device(e.to_string()))?,
            );
            self.initialized.fetch_add(1, Ordering::Relaxed);
        }
        let context = state.context.as_ref().unwrap();
        if !context.is_valid() || positions.start != context.ready_len() {
            return Err(KVPoolError::InvalidInput("invalid hybrid append".into()));
        }
        self.kv.prepare_append(&mut state.kv, positions, exec)
    }
    fn commit_append(&self, state: &mut Self::ContextState, positions: Range<usize>) -> Result<()> {
        self.validate_state(state)?;
        state
            .context
            .as_mut()
            .ok_or_else(|| KVPoolError::InvalidInput("missing hybrid state".into()))?
            .commit(positions)
            .map_err(|e| KVPoolError::InvalidInput(e.to_string()))
    }
    fn release_state(&self, state: Self::ContextState) {
        drop(state);
    }
    fn metrics(&self) -> KVPoolMetrics {
        let mut metrics = self.kv.metrics();
        let recurrent = self.initialized.load(Ordering::Relaxed) * self.recurrent_bytes;
        metrics.layers.allocated_bytes += recurrent;
        metrics.layers.assigned_bytes += recurrent;
        metrics
    }
    fn take_prefix(
        &self,
        _state: &mut Self::ContextState,
        _positions: Range<usize>,
    ) -> Result<Self::PrefixState> {
        Err(unsupported_prefix())
    }
    fn bind_prefix(
        &self,
        _state: &mut Self::ContextState,
        _segments: &[&Self::PrefixState],
        _lease: &PrefixLease,
    ) -> Result<Option<PrefixLease>> {
        Err(unsupported_prefix())
    }
    fn read_view<'a>(
        &'a self,
        state: &'a Self::ContextState,
        layer: usize,
        read_range: Range<usize>,
    ) -> Result<Self::ReadView<'a>> {
        self.kv.read_view(&state.kv, layer, read_range)
    }
    fn write_view<'a>(
        &'a self,
        state: &'a mut Self::ContextState,
        layer: usize,
        positions: Range<usize>,
        exec: &'a ExecutionContext,
    ) -> Result<Self::WriteView<'a>> {
        self.kv.write_view(&mut state.kv, layer, positions, exec)
    }
    fn store(
        &self,
        dst: Self::WriteView<'_>,
        k: TensorView<'_, T>,
        v: TensorView<'_, T>,
    ) -> Result<()> {
        self.kv.store(dst, k, v)
    }
}

impl<T: WeightElement + ElementType> ExecutionStatePool for Qwen3_5StatePool<T> {
    type ExecutionView<'a> = Qwen3_5ExecutionView<'a, T>;

    fn execution_view<'a>(
        &'a self,
        state: &'a mut Self::ContextState,
        positions: Range<usize>,
        tokens: &[u32],
        exec: &'a ExecutionContext,
    ) -> Result<Self::ExecutionView<'a>> {
        self.validate_state(state)?;
        state
            .context
            .as_mut()
            .ok_or_else(|| KVPoolError::InvalidInput("missing hybrid state".into()))?
            .begin(tokens, positions, Some((&self.kv, &mut state.kv)), exec)
            .map_err(|e| KVPoolError::InvalidInput(e.to_string()))
    }
}
