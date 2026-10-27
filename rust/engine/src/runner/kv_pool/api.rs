use std::{collections::VecDeque, ops::Range, sync::Arc};

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
    /// Physical storage owner; logical progress does not depend on its layout.
    pool: Arc<P>,
    /// Backend ownership, temporarily absent while moved into a submission task.
    state: Option<P::ContextState>,
    /// End of the prefix whose GPU work completed successfully and was committed.
    ready_len: usize,
    /// FIFO ends of at most two uncommitted appends; starts follow ready_len/previous end.
    /// Example: ready_len=10, ends=[14,15] means [10..14,14..15]; only the last is writable.
    pending_ends: VecDeque<usize>,
}

impl<P: KVPool> KVContext<P> {
    pub fn new(pool: Arc<P>) -> ContextResult<Self> {
        let state = pool.alloc_state()?;
        Ok(Self {
            pool,
            state: Some(state),
            ready_len: 0,
            pending_ends: VecDeque::new(),
        })
    }

    /// Read-only backend state for KVPool implementations.
    pub fn state(&self) -> &P::ContextState {
        self.state.as_ref().unwrap()
    }

    /// Newest reserved append, not the full in-flight FIFO; returning a copy
    /// does not expose mutable progress.
    pub fn pending_range(&self) -> Option<Range<usize>> {
        let end = *self.pending_ends.back()?;
        let start = self
            .pending_ends
            .iter()
            .rev()
            .nth(1)
            .copied()
            .unwrap_or(self.ready_len);
        Some(start..end)
    }

    pub fn ready_len(&self) -> usize {
        self.ready_len
    }
    /// Logical reservation cursor, derived from the newest append or committed prefix.
    /// After a backend prepare failure it includes attempted work; release the context.
    pub fn allocated_len(&self) -> usize {
        self.pending_ends.back().copied().unwrap_or(self.ready_len)
    }
    pub fn max_seq_len(&self) -> usize {
        self.pool.max_seq_len()
    }
    pub fn is_pending(&self) -> bool {
        !self.pending_ends.is_empty()
    }

    /// Reserve an append and enqueue GPU mapping updates without advancing ready_len.
    /// CPU reservation is complete on return; GPU updates may be pending, running, or
    /// finished. This does not write K/V or wait: subsequent work uses the same exec,
    /// and commit_append follows successful outer completion. A failed submission
    /// invalidates this context until release; rollback/truncate is not implemented.
    pub fn prepare_append(&mut self, count: usize, exec: &ExecutionContext) -> ContextResult<()> {
        let _range = crate::profiling::ScopedRange::new(c"prepare_append");
        if !self.pending_ends.is_empty() || count == 0 {
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
        self.pending_ends.push_back(end);
        self.pool
            .prepare_append(self.state.as_mut().unwrap(), positions, exec)?;
        Ok(())
    }

    /// Prepare a batch from one pool without advancing ready_len or waiting for GPU work.
    /// Validate logical inputs first, then mark all contexts pending before backend
    /// reservation/submission. On backend failure, release every affected context.
    pub fn prepare_append_batch(
        appends: &mut [(&mut Self, usize)],
        exec: &ExecutionContext,
    ) -> ContextResult<()> {
        if appends.is_empty() {
            return Ok(());
        }
        Self::prepare_batch_with(appends, exec, false, |pool, states, exec| {
            pool.prepare_append_batch(states, exec)
        })
    }

    /// Ordered compute-stream path; at most the previous and current appends coexist.
    pub(in crate::runner) fn prepare_queued_append_batch(
        appends: &mut [(&mut Self, usize)],
        exec: &ExecutionContext,
    ) -> ContextResult<()> {
        Self::prepare_batch_with(appends, exec, true, |pool, states, exec| {
            pool.prepare_append_batch(states, exec)
        })
    }

    pub(super) fn prepare_batch_with<T>(
        appends: &mut [(&mut Self, usize)],
        exec: &ExecutionContext,
        queued: bool,
        prepare: impl FnOnce(
            &P,
            &mut [(&mut P::ContextState, Range<usize>)],
            &ExecutionContext,
        ) -> Result<T>,
    ) -> ContextResult<T> {
        let _range = crate::profiling::ScopedRange::new(c"prepare_append_batch");
        let (first, _) = appends
            .first()
            .ok_or_else(|| KVContextError::InvalidState("empty append batch".into()))?;
        let pool = first.pool.clone();
        for (context, count) in appends.iter() {
            if !Arc::ptr_eq(&pool, &context.pool) {
                return Err(KVContextError::InvalidState(
                    "append batch must use one KV pool".into(),
                ));
            }
            if (!queued && !context.pending_ends.is_empty())
                || context.pending_ends.len() >= 2
                || *count == 0
            {
                return Err(KVContextError::InvalidState(
                    "append must be nonempty and context idle".into(),
                ));
            }
            context
                .allocated_len()
                .checked_add(*count)
                .filter(|&end| end <= pool.max_seq_len())
                .ok_or(KVContextError::CapacityExceeded)?;
        }
        for (context, count) in appends.iter_mut() {
            let end = context.allocated_len() + *count;
            context.pending_ends.push_back(end);
        }
        let mut states: Vec<_> = appends
            .iter_mut()
            .map(|(context, _)| {
                let positions = context.pending_range().unwrap();
                (context.state.as_mut().unwrap(), positions)
            })
            .collect();
        Ok(prepare(&pool, &mut states, exec)?)
    }

    /// View all reserved positions, including history and the current pending append.
    /// Pending K/V writes must precede attention reads on the same exec.
    pub fn read_view(&self, layer: usize) -> ContextResult<P::ReadView<'_>> {
        Ok(self
            .pool
            .read_view(self.state(), layer, 0..self.allocated_len())?)
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
            .pending_range()
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
            .pending_range()
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
        self.ready_len = self
            .pending_ends
            .pop_front()
            .ok_or_else(|| KVContextError::InvalidState("no pending KV append".into()))?;
        Ok(())
    }

    /// Move ownership into a submission task. The vacant source must be restored
    /// before scheduling/access, or invalidated on a failed/cancelled submission.
    pub(in crate::runner) fn take_context_for_gpu_submission(&mut self) -> Self {
        Self {
            pool: self.pool.clone(),
            state: self.state.take(),
            ready_len: self.ready_len,
            pending_ends: std::mem::take(&mut self.pending_ends),
        }
    }

    /// False only after ownership was taken for submission or released.
    pub(in crate::runner) fn has_owned_pool_state(&self) -> bool {
        self.state.is_some()
    }

    /// Commit the FIFO append only after its particular completion succeeded.
    pub(in crate::runner) fn commit_completed_append(&mut self, end: usize) -> ContextResult<()> {
        if self.pending_ends.front().copied() != Some(end) {
            return Err(KVContextError::InvalidState(
                "KV completions must commit in submission order".into(),
            ));
        }
        self.commit_append()
    }

    /// Release ownership, including failed pending appends. Pool leases protect GPU work.
    pub fn release(self) {
        drop(self);
    }
}

impl<P: KVPool> Drop for KVContext<P> {
    fn drop(&mut self) {
        if let Some(state) = self.state.take() {
            self.pool.release_state(state);
        }
    }
}
