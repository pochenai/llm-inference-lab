use std::{ops::Range, sync::Arc};

use cuda_async::device_operation::ExecutionContext;
use cutile::tensor::TensorView;

use super::{KVContextError, KVPoolError};

type Result<T> = std::result::Result<T, KVPoolError>;
type ContextResult<T> = std::result::Result<T, KVContextError>;

/// Storage and mapping contract. Prefix sharing and eviction are deliberately deferred.
/// A read view cannot be used as a store destination:
///
/// ```compile_fail,E0308
/// use engine::runner::kv_pool::{KVPool, NaiveKVPool, NaiveReadView};
/// use cutile::tensor::TensorView;
/// fn invalid_store(pool: &NaiveKVPool, read: NaiveReadView<'_>,
///                  k: TensorView<'_, f32>, v: TensorView<'_, f32>) {
///     pool.store(read, k, v).unwrap();
/// }
/// ```
pub trait KVPool: Sized + Send + Sync {
    type ContextState: Send + Sync;
    type ReadView<'a>
    where
        Self: 'a;
    type WriteView<'a>
    where
        Self: 'a;

    fn max_seq_len(&self) -> usize;
    fn alloc_state(&self) -> Result<Self::ContextState>;

    /// Conservative admission snapshot for unreserved token appends and additional
    /// context states. Actual reservation still happens in alloc_state/prepare_append.
    fn can_append(&self, tokens: usize, new_contexts: usize) -> bool;

    /// Reserve positions on CPU and immediately enqueue mapping updates on exec.
    /// The GPU may already be executing those updates when this returns; no wait occurs.
    /// K/V values are written later by store on the same exec.
    /// Implementations retain allocation ownership in exec before submitting GPU accesses.
    fn prepare_append(
        &self,
        state: &mut Self::ContextState,
        positions: Range<usize>,
        exec: &ExecutionContext,
    ) -> Result<()>;

    /// Reserve and enqueue mappings for a batch from this pool, on the same exec.
    /// Backends may combine uploads/launches; the default preserves per-context behavior.
    /// A failure may leave reservations pending; callers must release affected contexts.
    fn prepare_append_batch(
        &self,
        appends: &mut [(&mut Self::ContextState, Range<usize>)],
        exec: &ExecutionContext,
    ) -> Result<()> {
        for (state, positions) in appends {
            self.prepare_append(state, positions.clone(), exec)?;
        }
        Ok(())
    }
    /// View a logical token range using only backend state. The range may include
    /// pending K/V writes; reads must follow those writes on the same exec.
    /// During append forward, read_range covers history plus the current queries' K/V;
    /// the attention backend applies the per-query causal mask within this range.
    fn read_view<'a>(
        &'a self,
        state: &'a Self::ContextState,
        layer: usize,
        // Logical KV positions: history plus the current queries' K/V during append
        // forward. Attention applies the per-query causal mask within this range.
        read_range: Range<usize>,
    ) -> Result<Self::ReadView<'a>>;

    /// Expose writable destinations for already reserved positions and retain their
    /// buffers/leases in exec. No K/V writes or GPU wait occur here. KVContext chooses
    /// pending positions; this lower-level API validates backend bounds and ownership.
    fn write_view<'a>(
        &'a self,
        state: &'a mut Self::ContextState,
        layer: usize,
        positions: Range<usize>,
        exec: &'a ExecutionContext,
    ) -> Result<Self::WriteView<'a>>;

    /// Consume a write view and enqueue K/V copies on its execution context.
    /// Read views are not valid destinations. Return does not imply GPU completion.
    fn store(
        &self,
        dst: Self::WriteView<'_>,
        k: TensorView<'_, f32>,
        v: TensorView<'_, f32>,
    ) -> Result<()>;

    /// Drop context ownership; in-flight submission leases delay actual row/slot reuse.
    fn release_state(&self, state: Self::ContextState);
    fn metrics(&self) -> KVPoolMetrics;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferMetrics {
    pub allocated_bytes: usize,
    pub assigned_bytes: usize,
    pub free_bytes: usize,
    /// Unused capacity within assigned bytes, not an additional allocation.
    pub internal_unused_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KVPoolMetrics {
    pub index: BufferMetrics,
    pub layers: BufferMetrics,
}

/// Logical progress independent of the pool's physical layout.
pub struct KVContext<P: KVPool> {
    pool: Arc<P>,
    state: Option<P::ContextState>,
    ready_len: usize,
    allocated_len: usize,
    pending: Option<Range<usize>>,
}

impl<P: KVPool> KVContext<P> {
    pub fn new(pool: Arc<P>) -> ContextResult<Self> {
        let state = pool.alloc_state()?;
        Ok(Self {
            pool,
            state: Some(state),
            ready_len: 0,
            allocated_len: 0,
            pending: None,
        })
    }

    /// Read-only backend state for KVPool implementations.
    pub fn state(&self) -> &P::ContextState {
        self.state.as_ref().unwrap()
    }

    /// Logical append range; returning a copy does not expose mutable progress.
    pub fn pending_range(&self) -> Option<Range<usize>> {
        self.pending.clone()
    }

    pub fn ready_len(&self) -> usize {
        self.ready_len
    }
    pub fn allocated_len(&self) -> usize {
        self.allocated_len
    }
    pub fn max_seq_len(&self) -> usize {
        self.pool.max_seq_len()
    }
    pub fn is_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Reserve an append and enqueue GPU mapping updates without advancing ready_len.
    /// CPU reservation is complete on return; GPU updates may be pending, running, or
    /// finished. This does not write K/V or wait: subsequent work uses the same exec,
    /// and commit_append follows successful outer completion. A failed submission
    /// invalidates this context until release; rollback/truncate is not implemented.
    pub fn prepare_append(&mut self, count: usize, exec: &ExecutionContext) -> ContextResult<()> {
        let _range = crate::profiling::ScopedRange::new(c"prepare_append");
        if self.pending.is_some() || count == 0 {
            return Err(KVContextError::InvalidState(
                "append must be nonempty and context idle".into(),
            ));
        }
        let end = self
            .ready_len
            .checked_add(count)
            .filter(|&end| end <= self.pool.max_seq_len())
            .ok_or(KVContextError::CapacityExceeded)?;
        let positions = self.ready_len..end;
        // Mark pending before submission: partial failure must never expose ready KV.
        self.pending = Some(positions.clone());
        self.pool
            .prepare_append(self.state.as_mut().unwrap(), positions, exec)?;
        self.allocated_len = end;
        Ok(())
    }

    /// Prepare a batch from one pool without advancing ready_len or waiting for GPU work.
    /// Validate logical inputs first, then mark all contexts pending before backend
    /// reservation/submission. On backend failure, release every affected context.
    pub fn prepare_append_batch(
        appends: &mut [(&mut Self, usize)],
        exec: &ExecutionContext,
    ) -> ContextResult<()> {
        let _range = crate::profiling::ScopedRange::new(c"prepare_append_batch");
        let Some((first, _)) = appends.first() else {
            return Ok(());
        };
        let pool = first.pool.clone();
        for (context, count) in appends.iter() {
            if !Arc::ptr_eq(&pool, &context.pool) {
                return Err(KVContextError::InvalidState(
                    "append batch must use one KV pool".into(),
                ));
            }
            if context.pending.is_some() || *count == 0 {
                return Err(KVContextError::InvalidState(
                    "append must be nonempty and context idle".into(),
                ));
            }
            context
                .ready_len
                .checked_add(*count)
                .filter(|&end| end <= pool.max_seq_len())
                .ok_or(KVContextError::CapacityExceeded)?;
        }
        for (context, count) in appends.iter_mut() {
            context.pending = Some(context.ready_len..context.ready_len + *count);
        }
        let mut states: Vec<_> = appends
            .iter_mut()
            .map(|(context, _)| {
                (
                    context.state.as_mut().unwrap(),
                    context.pending.clone().unwrap(),
                )
            })
            .collect();
        pool.prepare_append_batch(&mut states, exec)?;
        for (context, _) in appends {
            context.allocated_len = context.pending.as_ref().unwrap().end;
        }
        Ok(())
    }

    /// View all reserved positions, including history and the current pending append.
    /// Pending K/V writes must precede attention reads on the same exec.
    pub fn read_view(&self, layer: usize) -> ContextResult<P::ReadView<'_>> {
        Ok(self
            .pool
            .read_view(self.state(), layer, 0..self.allocated_len)?)
    }

    /// Expose only the current pending append as writable destinations; callers cannot
    /// choose a range. Hold an exclusive context borrow and retain GPU resources in exec.
    /// Dropping the view does not complete GPU work or advance ready_len.
    ///
    /// ```compile_fail,E0502
    /// use engine::runner::kv_pool::{KVContext, KVContextError, NaiveKVPool};
    /// use cuda_async::device_operation::ExecutionContext;
    /// fn overlapping_views(kv: &mut KVContext<NaiveKVPool>, exec: &ExecutionContext)
    ///     -> Result<(), KVContextError> {
    ///     let dst = kv.write_view(0, exec)?;
    ///     let read = kv.read_view(0)?;
    ///     drop(dst);
    ///     Ok(())
    /// }
    /// ```
    pub fn write_view<'a>(
        &'a mut self,
        layer: usize,
        exec: &'a ExecutionContext,
    ) -> ContextResult<P::WriteView<'a>> {
        let positions = self
            .pending
            .clone()
            .ok_or_else(|| KVContextError::InvalidState("no pending KV append".into()))?;
        Ok(self
            .pool
            .write_view(self.state.as_mut().unwrap(), layer, positions, exec)?)
    }

    /// Enqueue GPU K/V writes only to the current pending append range.
    /// This ordinary fn does not wait for GPU completion or advance ready_len.
    /// Mapping updates, these writes, and subsequent attention use the same exec
    /// so stream order preserves their dependencies.
    pub fn store(
        &mut self,
        layer: usize,
        k: TensorView<'_, f32>,
        v: TensorView<'_, f32>,
        exec: &ExecutionContext,
    ) -> ContextResult<()> {
        let positions = self
            .pending
            .clone()
            .ok_or_else(|| KVContextError::InvalidState("no pending KV append".into()))?;
        let dst = self
            .pool
            .write_view(self.state.as_mut().unwrap(), layer, positions, exec)?;
        Ok(self.pool.store(dst, k, v)?)
    }

    /// Internal: execute_append_batch advances CPU ready_len only after GPU success.
    /// This is not a GPU synchronization operation. The caller must first confirm
    /// successful completion of this append's mapping updates, K/V writes, and all
    /// layers, e.g. after the outer scheduled operation is awaited or sync_on succeeds.
    pub(in crate::runner) fn commit_append(&mut self) -> ContextResult<()> {
        let _range = crate::profiling::ScopedRange::new(c"commit_append");
        let pending = self
            .pending
            .take()
            .ok_or_else(|| KVContextError::InvalidState("no pending KV append".into()))?;
        self.ready_len = pending.end;
        Ok(())
    }

    /// Release ownership, including failed pending appends. Pool leases protect GPU work.
    pub fn release(mut self) {
        if let Some(state) = self.state.take() {
            self.pool.release_state(state);
        }
    }
}

impl<P: KVPool> Drop for KVContext<P> {
    fn drop(&mut self) {
        if let Some(state) = self.state.take() {
            self.pool.release_state(state);
        }
    }
}
