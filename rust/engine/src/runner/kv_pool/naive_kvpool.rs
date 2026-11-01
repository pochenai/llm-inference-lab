use std::{
    ops::Range,
    sync::{Arc, Mutex},
};

use cuda_async::{
    device_buffer::DevicePointer,
    device_operation::{DeviceOp, ExecutionContext, GraphNode},
    scheduling_policies::SchedulingPolicy,
};
use cutile::{
    api,
    core::ElementType,
    tensor::{Tensor, TensorView},
    tile_kernel::TileKernel,
    DType,
};

use crate::radix_tree::SplitPrefix;
use crate::runner::upload::UploadPool;

use super::{
    BufferMetrics, KVContext, KVContextError, KVPool, KVPoolError, KVPoolMetrics, PrefixLease,
};

type Result<T> = std::result::Result<T, KVPoolError>;

const BIND_TOKENS_PER_TILE: usize = 128;
const STORE_TOKENS_PER_TILE: usize = 8;

#[derive(Clone, Copy, Debug)]
pub struct NaiveKVPoolConfig {
    pub num_layers: usize,
    pub num_slots: usize,
    /// Maximum resident contexts; mapping rows are all preallocated at startup.
    pub max_rows: usize,
    pub max_seq_len: usize,
    /// Stored K/V vector width per token; a backend may flatten kv_heads * head_dim here.
    pub head_dim: usize,
}

impl Default for NaiveKVPoolConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            num_slots: 4096,
            max_rows: 32,
            max_seq_len: 512,
            head_dim: 16,
        }
    }
}

/// Per-layer physical KV storage and independent logical context rows.
/// One slot ID addresses a K vector and a V vector of head_dim elements in EVERY
/// layer; each layer has separate buffers. num_slots is the shared token capacity,
/// not a separate allocation budget for each row or layer.
///
/// ```text
/// Physical K/V buffers per layer: [num_slots, head_dim] (single head).
/// Logical slot_table:             [max_rows, max_seq_len].
///
///                       token_position
///                     0   1   2   3   4
/// context_row 0      [10, 11, 12, 13,  ?]
/// context_row 1      [20, 21, 22,  ?,  ?]
///
/// If all max_rows contexts must fit at max_seq_len simultaneously:
/// max_rows <= num_slots / max_seq_len // Integer division.
/// ```
pub struct NaiveKVPool<T: DType = f32> {
    /// 按 layer + slot 存储，逻辑 context 经 slot_table 寻址，后续可共享前缀。
    /// context 若独占连续 KV 区域，fork 就需要复制历史；间接映射解除了这一限制。
    layers: Vec<(Arc<Tensor<T>>, Arc<Tensor<T>>)>,
    /// token_position → slot mapping
    slot_table: Arc<Tensor<i32>>,
    config: NaiveKVPoolConfig,
    allocation: Arc<Mutex<Allocation>>,
    mapping_uploads: Arc<UploadPool>,
}

/// Opaque mapping payload and its GPU destination, produced during CPU reservation.
pub(in crate::runner) struct NaiveMappingPlan {
    updates: Vec<i32>,
    table: Arc<Tensor<i32>>,
}

impl NaiveMappingPlan {
    pub(in crate::runner) fn payload(&self) -> &[i32] {
        &self.updates
    }

    /// The payload must be uploaded unchanged at offset. Retain matching state
    /// leases until execution completes, and order bind before dependent K/V work.
    /// Capture records only backing buffers; every replay must retain its own
    /// request's reservations on the outer execution context.
    pub(in crate::runner) unsafe fn submit(
        &self,
        upload: &Arc<Tensor<i32>>,
        offset: usize,
        exec: &ExecutionContext,
    ) -> Result<()> {
        if self.updates.is_empty() {
            return Ok(());
        }
        let end = offset
            .checked_add(self.updates.len())
            .filter(|&end| end <= upload.size())
            .ok_or_else(|| KVPoolError::InvalidInput("mapping payload exceeds upload".into()))?;
        if upload.shape().len() != 1
            || upload.device_id() != self.table.device_id()
            || exec.get_device_id() != self.table.device_id()
        {
            return Err(KVPoolError::InvalidInput(
                "mapping upload shape or device differs".into(),
            ));
        }
        let count = self.updates.len() / 2;
        let flat = upload.slice(&[offset..end])?;
        let updates = flat.view(&[2, count])?;
        exec.retain(self.table.clone())?;
        self.bind_op(&updates)?.execute(exec)?;
        Ok(())
    }

    /// Construct bind without submitting it, for eager execution or scope recording.
    /// The caller retains the table and matching reservations until GPU completion;
    /// updates must contain the unchanged payload returned by this plan.
    pub(crate) unsafe fn bind_op<'a>(
        &'a self,
        updates: &'a TensorView<'_, i32>,
    ) -> Result<impl GraphNode + 'a> {
        let count = self.updates.len() / 2;
        if count == 0 || updates.shape() != [2, count as i32] {
            return Err(KVPoolError::InvalidInput(
                "bind needs a nonempty [2, mapping_count] payload".into(),
            ));
        }
        Ok(kernels::bind_batch(self.table.device_pointer(), updates)
            .generics(vec![BIND_TOKENS_PER_TILE.to_string()])
            .grid((count.div_ceil(BIND_TOKENS_PER_TILE) as u32, 1, 1)))
    }
}

impl<T: DType + ElementType> KVContext<NaiveKVPool<T>> {
    /// Naive-only deferred reservation for packing mapping and model uploads.
    /// Submit the plan before K/V access; commit only after successful GPU completion.
    pub(in crate::runner) fn reserve_append_batch(
        appends: &mut [(&mut Self, usize)],
        exec: &ExecutionContext,
    ) -> std::result::Result<NaiveMappingPlan, KVContextError> {
        Self::prepare_batch_with(appends, exec, false, |pool, states, exec| {
            pool.reserve_append_batch(states, exec)
        })
    }
}

struct Allocation {
    free_rows: Vec<usize>,
    free_slots: Vec<i32>,
    bound_positions: usize,
}

/// A submission retains this lease, not just the pool buffers, to prevent slot reuse.
struct StateLease {
    row: usize,
    slots: Mutex<ContextSlots>,
    allocation: Arc<Mutex<Allocation>>,
}

/// Slot IDs are CPU metadata; neither vector contains GPU K/V data.
/// For example, mapping = [10, 11, 12, 20, 21], owned = [20, 21]:
/// the prefix lease protects cached slots [10, 11, 12]; this row owns the suffix.
#[derive(Default)]
struct ContextSlots {
    /// Logical token position -> physical slot, including the shared prefix.
    mapping: Vec<i32>,
    /// Slots this row must return on Drop. take_prefix removes transferred slots
    /// immediately, before bind_prefix replaces the lease, preventing double-free.
    owned: Vec<i32>,
    /// Protects tree-owned prefix slots until this row's final GPU lease is dropped.
    prefix: Option<PrefixLease>,
}

impl Drop for StateLease {
    fn drop(&mut self) {
        let slots = self.slots.get_mut().unwrap();
        {
            let mut allocation = self.allocation.lock().unwrap();
            allocation.bound_positions -= slots.mapping.len();
            allocation.free_slots.append(&mut slots.owned);
            allocation.free_rows.push(self.row);
        }
        // Unpin may destroy the tree and return cached slots: never hold allocation here.
        drop(slots.prefix.take());
    }
}

pub struct NaiveContextState {
    lease: Arc<StateLease>,
    /// CPU attachment defers mapping upload until this row's first append.
    needs_prefix_bind: bool,
}

/// Cache ownership has no slot-table row and never copies K/V buffers.
pub struct NaivePrefixState {
    /// CPU IDs of the physical token slots owned by this tree edge, not K/V data.
    /// Each ID addresses that token's K/V vectors in every layer of this pool.
    slots: Vec<i32>,
    /// Shared CPU allocator bookkeeping; Drop returns slots to its free list.
    /// Arc clones share this allocator, without copying or owning the GPU buffers.
    allocation: Arc<Mutex<Allocation>>,
}

impl Drop for NaivePrefixState {
    fn drop(&mut self) {
        self.allocation
            .lock()
            .unwrap()
            .free_slots
            .append(&mut self.slots);
    }
}

impl SplitPrefix for NaivePrefixState {
    fn split_prefix(&mut self, len: usize) -> Self {
        Self {
            slots: self.slots.split_prefix(len),
            allocation: self.allocation.clone(),
        }
    }
}

struct LayerStorage<'a, T: DType> {
    k: &'a Arc<Tensor<T>>,
    v: &'a Arc<Tensor<T>>,
    slot_table: &'a Arc<Tensor<i32>>,
    row: usize,
    positions: Range<usize>,
    lease: Arc<StateLease>,
}

impl<T: DType> LayerStorage<'_, T> {
    fn retain(&self, exec: &ExecutionContext) -> Result<()> {
        exec.retain(self.lease.clone())?;
        exec.retain(self.k.clone())?;
        exec.retain(self.v.clone())?;
        exec.retain(self.slot_table.clone())?;
        Ok(())
    }
}

/// Read-only addressing. Safe accessors expose const pointers, never writable buffers.
/// Building a view performs no GPU reads: graph preparation may inspect addresses
/// before the first append binds a reused row. GPU loads must follow that bind.
pub struct NaiveReadView<'a, T: DType = f32> {
    inner: LayerStorage<'a, T>,
}

/// Writable addressing for reserved positions, borrowed exclusively from context state.
/// Construction retains GPU resources on exec; this is not proof of GPU completion.
pub struct NaiveWriteView<'a, T: DType = f32> {
    inner: LayerStorage<'a, T>,
    exec: &'a ExecutionContext,
}

impl<T: DType> NaiveReadView<'_, T> {
    pub fn k_ptr(&self) -> *const T {
        self.inner.k.device_pointer().cu_deviceptr() as *const T
    }
    pub fn v_ptr(&self) -> *const T {
        self.inner.v.device_pointer().cu_deviceptr() as *const T
    }
    pub fn slot_table_ptr(&self) -> *const i32 {
        self.inner.slot_table.device_pointer().cu_deviceptr() as *const i32
    }
    pub fn row(&self) -> usize {
        self.inner.row
    }
    pub fn positions(&self) -> &Range<usize> {
        &self.inner.positions
    }
    pub fn head_dim(&self) -> usize {
        self.inner.k.shape()[1] as usize
    }
    pub fn table_width(&self) -> usize {
        self.inner.slot_table.shape()[1] as usize
    }
    /// Retain buffers and allocation ownership before submitting any GPU reads.
    pub fn retain(&self, exec: &ExecutionContext) -> Result<()> {
        self.inner.retain(exec)
    }
    /// Internal bridge for tensor-based kernels; safe callers only receive const pointers.
    ///
    /// # Safety
    /// Reads require retaining this view and staying within its positions. These
    /// owners alone grant no write permission: fused/graph writes require a matching
    /// pending WriteView on the actual submission. Only its unique pending slots may
    /// be written; acquiring a view or recording a graph does not complete GPU work.
    pub(crate) unsafe fn tensors(&self) -> (&Arc<Tensor<T>>, &Arc<Tensor<T>>, &Arc<Tensor<i32>>) {
        (self.inner.k, self.inner.v, self.inner.slot_table)
    }
}

impl<T: DType> NaiveWriteView<'_, T> {
    pub fn k_mut_ptr(&self) -> DevicePointer<T> {
        self.inner.k.device_pointer()
    }
    pub fn v_mut_ptr(&self) -> DevicePointer<T> {
        self.inner.v.device_pointer()
    }
    pub fn slot_table_ptr(&self) -> *const i32 {
        self.inner.slot_table.device_pointer().cu_deviceptr() as *const i32
    }
    pub fn row(&self) -> usize {
        self.inner.row
    }
    pub fn positions(&self) -> &Range<usize> {
        &self.inner.positions
    }
    pub fn head_dim(&self) -> usize {
        self.inner.k.shape()[1] as usize
    }
    pub fn table_width(&self) -> usize {
        self.inner.slot_table.shape()[1] as usize
    }
    /// Submit fused writes on the same context that retains these destinations.
    pub fn execution_context(&self) -> &ExecutionContext {
        self.exec
    }
}

impl<T: DType + ElementType> NaiveKVPool<T> {
    /// Allocate backing buffers once; requests subsequently reserve rows and slots.
    pub async fn new(
        config: NaiveKVPoolConfig,
        policy: &Arc<dyn SchedulingPolicy>,
    ) -> Result<Arc<Self>> {
        let dims = [
            config.num_layers,
            config.num_slots,
            config.max_rows,
            config.max_seq_len,
            config.head_dim,
        ];
        if !config.head_dim.is_power_of_two()
            || dims.iter().any(|&n| n == 0 || n > i32::MAX as usize)
            || config
                .num_slots
                .checked_mul(config.head_dim)
                .filter(|&n| n <= i32::MAX as usize)
                .is_none()
            || config
                .max_rows
                .checked_mul(config.max_seq_len)
                .filter(|&n| n <= i32::MAX as usize)
                .is_none()
        {
            return Err(KVPoolError::InvalidInput(
                "invalid KV pool dimensions".into(),
            ));
        }
        let mut layers = Vec::new();
        for _ in 0..config.num_layers {
            let k = api::zeros::<T>(&[config.num_slots, config.head_dim])
                .schedule(policy)?
                .await?;
            let v = api::zeros::<T>(&[config.num_slots, config.head_dim])
                .schedule(policy)?
                .await?;
            layers.push((Arc::new(k), Arc::new(v)));
        }
        // Preallocate max_seq_len mapping cells for every row, not K/V slots.
        // Physical K/V slots are assigned from the shared pool by prepare_append.
        let slot_table = Arc::new(
            api::zeros::<i32>(&[config.max_rows, config.max_seq_len])
                .schedule(policy)?
                .await?,
        );
        Ok(Arc::new(Self {
            layers,
            slot_table,
            config,
            mapping_uploads: Arc::default(),
            allocation: Arc::new(Mutex::new(Allocation {
                free_rows: (0..config.max_rows).rev().collect(),
                free_slots: (0..config.num_slots as i32).rev().collect(),
                bound_positions: 0,
            })),
        }))
    }

    pub fn config(&self) -> NaiveKVPoolConfig {
        self.config
    }

    fn layer_storage<'a>(
        &'a self,
        state: &'a NaiveContextState,
        layer: usize,
        positions: Range<usize>,
    ) -> Result<LayerStorage<'a, T>> {
        self.validate_state(state)?;
        if positions.start > positions.end
            || positions.end > state.lease.slots.lock().unwrap().mapping.len()
        {
            return Err(KVPoolError::InvalidInput(
                "KV view range exceeds mapped tokens".into(),
            ));
        }
        let (k, v) = self
            .layers
            .get(layer)
            .ok_or_else(|| KVPoolError::InvalidInput("invalid KV layer".into()))?;
        Ok(LayerStorage {
            k,
            v,
            slot_table: &self.slot_table,
            row: state.lease.row,
            positions,
            lease: state.lease.clone(),
        })
    }

    /// Reserve physical slots on CPU, retaining the lease before any GPU submission.
    fn reserve_append(
        &self,
        state: &mut NaiveContextState,
        positions: Range<usize>,
        exec: &ExecutionContext,
    ) -> Result<Vec<i32>> {
        self.validate_state(state)?;
        if exec.get_device_id() != self.slot_table.device_id() {
            return Err(KVPoolError::InvalidInput(
                "KV pool and execution device differ".into(),
            ));
        }
        let mut slots = state.lease.slots.lock().unwrap();
        if positions.start != slots.mapping.len()
            || positions.end > self.max_seq_len()
            || positions.is_empty()
        {
            return Err(KVPoolError::InvalidInput(
                "KV append range does not match allocation".into(),
            ));
        }
        exec.retain(state.lease.clone())?;
        let new_slots = {
            let mut allocation = self.allocation.lock().unwrap();
            if positions.len() > allocation.free_slots.len() {
                return Err(KVPoolError::CapacityExhausted(
                    "KV pool has no free token slots".into(),
                ));
            }
            let start = allocation.free_slots.len() - positions.len();
            let reserved = allocation.free_slots.split_off(start);
            allocation.bound_positions += reserved.len();
            reserved
        };
        slots.mapping.extend_from_slice(&new_slots);
        slots.owned.extend_from_slice(&new_slots);
        let mapping = if state.needs_prefix_bind {
            state.needs_prefix_bind = false;
            slots.mapping.clone()
        } else {
            new_slots
        };
        drop(slots);
        Ok(mapping)
    }

    /// Reserve slots on CPU, leaving the upload/bind to the returned backend plan.
    pub(in crate::runner) fn reserve_append_batch(
        &self,
        appends: &mut [(&mut NaiveContextState, Range<usize>)],
        exec: &ExecutionContext,
    ) -> Result<NaiveMappingPlan> {
        let mut offsets = Vec::new();
        let mut slot_ids = Vec::new();
        for (state, positions) in appends {
            let slots = self.reserve_append(state, positions.clone(), exec)?;
            let base = state.lease.row * self.config.max_seq_len;
            offsets.extend(
                (positions.end - slots.len()..positions.end)
                    .map(|position| (base + position) as i32),
            );
            slot_ids.extend(slots);
        }
        offsets.extend(slot_ids);
        Ok(NaiveMappingPlan {
            updates: offsets,
            table: self.slot_table.clone(),
        })
    }

    fn validate_state(&self, state: &NaiveContextState) -> Result<()> {
        if !Arc::ptr_eq(&self.allocation, &state.lease.allocation) {
            return Err(KVPoolError::InvalidInput(
                "KV state belongs to a different pool".into(),
            ));
        }
        Ok(())
    }
}

impl<T: DType + ElementType> KVPool for NaiveKVPool<T> {
    type Element = T;
    type ContextState = NaiveContextState;
    type PrefixState = NaivePrefixState;

    fn take_prefix(
        &self,
        state: &mut Self::ContextState,
        positions: Range<usize>,
    ) -> Result<Self::PrefixState> {
        self.validate_state(state)?;
        let mut slots = state.lease.slots.lock().unwrap();
        let cached = slots.prefix.as_ref().map_or(0, PrefixLease::token_len);
        if positions.start < cached
            || positions.is_empty()
            || positions.end > slots.mapping.len()
            || slots.owned.len() != slots.mapping.len() - cached
        {
            return Err(KVPoolError::InvalidInput(
                "prefix transfer exceeds private slots".into(),
            ));
        }
        let transferred = if positions.start == cached {
            // Move the whole private suffix when possible; a capped prefix allocates
            // only the smaller half instead of copying all cached slot IDs.
            slots.owned.split_prefix(positions.end - cached)
        } else {
            slots
                .owned
                .drain(positions.start - cached..positions.end - cached)
                .collect()
        };
        Ok(NaivePrefixState {
            slots: transferred,
            allocation: self.allocation.clone(),
        })
    }

    fn bind_prefix(
        &self,
        state: &mut Self::ContextState,
        segments: &[&Self::PrefixState],
        lease: &PrefixLease,
    ) -> Result<Option<PrefixLease>> {
        self.validate_state(state)?;
        let len = lease.token_len();
        let mut slots = state.lease.slots.lock().unwrap();
        let old_len = slots.prefix.as_ref().map_or(0, PrefixLease::token_len);
        if len == 0
            || len > self.config.max_seq_len
            || len < old_len
            || segments
                .iter()
                .any(|s| !Arc::ptr_eq(&self.allocation, &s.allocation))
            || segments.iter().map(|s| s.slots.len()).sum::<usize>() != len
        {
            return Err(KVPoolError::InvalidInput(
                "invalid canonical prefix or pool".into(),
            ));
        }
        let empty = slots.mapping.is_empty();
        if !empty && (len > slots.mapping.len() || slots.owned.len() < slots.mapping.len() - len) {
            return Err(KVPoolError::InvalidInput(
                "prefix binding exceeds mapped/private slots".into(),
            ));
        }
        if empty {
            slots.mapping.reserve(len);
            for segment in segments {
                slots.mapping.extend_from_slice(&segment.slots);
            }
            self.allocation.lock().unwrap().bound_positions += len;
            state.needs_prefix_bind = true;
        } else {
            let private_tail = slots.mapping.len() - len;
            let duplicates = slots.owned.len() - private_tail;
            if duplicates != 0 {
                // Only deduplication changes the row. Newly transferred slots already
                // have canonical IDs, so normal publication needs no copy or rebind.
                self.allocation
                    .lock()
                    .unwrap()
                    .free_slots
                    .extend(slots.owned.drain(..duplicates));
                let mut offset = 0;
                for segment in segments {
                    let end = offset + segment.slots.len();
                    slots.mapping[offset..end].copy_from_slice(&segment.slots);
                    offset = end;
                }
                state.needs_prefix_bind = true;
            }
        }
        Ok(slots.prefix.replace(lease.clone()))
    }
    type ReadView<'a> = NaiveReadView<'a, T>;
    type WriteView<'a> = NaiveWriteView<'a, T>;

    fn max_seq_len(&self) -> usize {
        self.config.max_seq_len
    }

    fn can_append(&self, tokens: usize, new_contexts: usize) -> bool {
        let allocation = self.allocation.lock().unwrap();
        tokens <= allocation.free_slots.len() && new_contexts <= allocation.free_rows.len()
    }

    /// Claim an exclusive slot_table row for this context's logical-to-physical mapping:
    /// slot_table[row, token_position] -> slot -> layers[layer].K/V[slot].
    /// This only claims the mapping row; prepare_append assigns physical KV slots
    /// incrementally. The final GPU lease releases the row/private suffix and unpins
    /// the cached path. Cached slots remain owned by the tree until it is destroyed.
    fn alloc_state(&self) -> Result<Self::ContextState> {
        let row = self
            .allocation
            .lock()
            .unwrap()
            .free_rows
            .pop()
            .ok_or_else(|| {
                KVPoolError::CapacityExhausted("KV pool has no free context rows".into())
            })?;
        Ok(NaiveContextState {
            lease: Arc::new(StateLease {
                row,
                slots: Mutex::new(ContextSlots::default()),
                allocation: self.allocation.clone(),
            }),
            needs_prefix_bind: false,
        })
    }

    /// Reserve slots in preallocated K/V storage and enqueue slot-table updates.
    /// No new K/V backing allocation or GPU wait; return is not a completion signal.
    fn prepare_append(
        &self,
        state: &mut Self::ContextState,
        positions: Range<usize>,
        exec: &ExecutionContext,
    ) -> Result<()> {
        let new_slots = self.reserve_append(state, positions.clone(), exec)?;
        let mapping_start = positions.end - new_slots.len();
        let mapping_count = new_slots.len();
        // SAFETY: exec owns the submission; its caller establishes completion.
        unsafe {
            let count = new_slots.len();
            let indices = self
                .mapping_uploads
                .upload(Arc::new(new_slots), &[count], exec)?;
            // The raw GPU pointer passed to bind carries no ownership: retain the
            // table until submission completion. Arc::clone does not copy GPU data.
            exec.retain(self.slot_table.clone())?;
            // Write the assigned slot IDs into this context's GPU mapping row.
            // Example: row=1, positions=3..6, indices=[10,25,7]:
            // slot_table[1]: [a,b,c,?,?,?,...] -> [a,b,c,10,25,7,...].
            // Arguments: destination pointer, GPU slot IDs, row width, row, start.
            // Each tile program updates BIND_TOKENS_PER_TILE cells with a masked tail.
            // execute enqueues without waiting.
            // Later K/V writes on the same exec run after these mapping updates.
            kernels::bind(
                self.slot_table.device_pointer(),
                indices,
                self.config.max_seq_len as i32,
                state.lease.row as i32,
                mapping_start as i32,
            )
            .generics(vec![BIND_TOKENS_PER_TILE.to_string()])
            .grid((mapping_count.div_ceil(BIND_TOKENS_PER_TILE) as u32, 1, 1))
            .execute(exec)?;
        }
        Ok(())
    }

    /// Compatibility path for callers that want reservation and immediate submission.
    fn prepare_append_batch(
        &self,
        appends: &mut [(&mut Self::ContextState, Range<usize>)],
        exec: &ExecutionContext,
    ) -> Result<()> {
        let mapping = self.reserve_append_batch(appends, exec)?;
        if mapping.payload().is_empty() {
            return Ok(());
        }
        // SAFETY: the same exec retains the upload and all reservations until completion.
        unsafe {
            let upload = self.mapping_uploads.upload(
                Arc::new(mapping.payload().to_vec()),
                &[mapping.payload().len()],
                exec,
            )?;
            mapping.submit(&upload, 0, exec)?;
        }
        Ok(())
    }

    fn read_view<'a>(
        &'a self,
        state: &'a Self::ContextState,
        layer: usize,
        positions: Range<usize>,
    ) -> Result<Self::ReadView<'a>> {
        Ok(NaiveReadView {
            inner: self.layer_storage(state, layer, positions)?,
        })
    }

    fn write_view<'a>(
        &'a self,
        state: &'a mut Self::ContextState,
        layer: usize,
        positions: Range<usize>,
        exec: &'a ExecutionContext,
    ) -> Result<Self::WriteView<'a>> {
        let inner = self.layer_storage(state, layer, positions)?;
        if inner.positions.is_empty() || exec.get_device_id() != inner.k.device_id() {
            return Err(KVPoolError::InvalidInput(
                "invalid KV write range or device".into(),
            ));
        }
        inner.retain(exec)?;
        Ok(NaiveWriteView { inner, exec })
    }

    /// Enqueue K/V scatter writes through slot_table on exec, without waiting.
    /// The same stream orders mapping updates before writes and attention afterward;
    /// returning successfully means submission succeeded, not that GPU writes finished.
    /// Each [B, head_dim] tile handles up to B tokens (B=8); the last tile is masked.
    fn store(
        &self,
        dst: Self::WriteView<'_>,
        k: TensorView<'_, T>,
        v: TensorView<'_, T>,
    ) -> Result<()> {
        let width = self.config.head_dim;
        if !Arc::ptr_eq(dst.inner.slot_table, &self.slot_table)
            || k.shape() != v.shape()
            || k.shape().len() != 2
            || k.shape()[0] < dst.positions().len() as i32
            || k.shape()[1] != width as i32
        {
            return Err(KVPoolError::InvalidInput(
                "invalid KV store pool or shape".into(),
            ));
        }
        // SAFETY: the view validates mapped bounds and retains storage/leases on exec.
        // Reserved slots are unique to this state; mapping updates precede the scatter.
        unsafe {
            kernels::store(
                dst.k_mut_ptr(),
                dst.v_mut_ptr(),
                dst.inner.slot_table.clone(),
                &k,
                &v,
                dst.row() as i32,
                dst.positions().start as i32,
                dst.positions().len() as i32,
            )
            .generics(vec![
                T::DTYPE.to_string(),
                STORE_TOKENS_PER_TILE.to_string(),
                width.to_string(),
            ])
            .grid((
                dst.positions().len().div_ceil(STORE_TOKENS_PER_TILE) as u32,
                1,
                1,
            ))
            .execute(dst.exec)?;
        }
        Ok(())
    }

    fn release_state(&self, state: Self::ContextState) {
        drop(state);
    }

    fn metrics(&self) -> KVPoolMetrics {
        let allocation = self.allocation.lock().unwrap();
        let c = self.config;
        let row_bytes = c.max_seq_len * size_of::<i32>();
        let slot_bytes = c.num_layers * 2 * c.head_dim * size_of::<T>();
        let active_rows = c.max_rows - allocation.free_rows.len();
        KVPoolMetrics {
            index: BufferMetrics {
                allocated_bytes: c.max_rows * row_bytes,
                assigned_bytes: active_rows * row_bytes,
                free_bytes: allocation.free_rows.len() * row_bytes,
                internal_unused_bytes: (active_rows * c.max_seq_len - allocation.bound_positions)
                    * size_of::<i32>(),
            },
            layers: BufferMetrics {
                allocated_bytes: c.num_slots * slot_bytes,
                assigned_bytes: (c.num_slots - allocation.free_slots.len()) * slot_bytes,
                free_bytes: allocation.free_slots.len() * slot_bytes,
                internal_unused_bytes: 0,
            },
        }
    }
}

#[cutile::module]
mod kernels {
    use cutile::core::*;

    /// Write slot_table[row, start+i] = slots[i], B entries per tile program.
    /// The table is flat in GPU memory: element offset = row * cols + start + i.
    /// Example: row=1, start=3, slots=[10,25,7] updates positions 3, 4, 5.
    /// The host must provide valid bounds and retain the table for the GPU launch.
    #[cutile::entry()]
    pub(super) unsafe fn bind<const B: i32>(
        table_ptr: *mut i32,
        slots: &Tensor<i32, { [-1] }>,
        cols: i32,
        row: i32,
        start: i32,
    ) {
        let pid = program_id(0);
        let local: Tile<i32, { [B] }> = iota(shape![B]);
        let indices = local + (pid * B).broadcast(shape![B]);
        let valid = lt_tile(indices, slots.shape()[0].broadcast(shape![B]));
        let slot = slots.partition(shape![B]).load([pid]);
        let pointer: PointerTile<*mut i32, { [] }> = pointer_to_tile(table_ptr);
        let pointer: PointerTile<*mut i32, { [] }> = pointer.offset(row * cols + start);
        let pointer: PointerTile<*mut i32, { [1] }> = pointer.reshape(shape![1]);
        let pointer: PointerTile<*mut i32, { [B] }> = pointer.broadcast(shape![B]);
        let destination: PointerTile<*mut i32, { [B] }> = pointer.offset_tile(indices);
        unsafe {
            store_ptr_tko(
                destination,
                slot,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(valid),
                None,
                Latency::<1>,
            );
        }
    }

    /// Scatter all contexts' mapping updates; mask the final partial tile.
    #[cutile::entry()]
    pub(super) unsafe fn bind_batch<const B: i32>(
        table_ptr: *mut i32,
        updates: &Tensor<i32, { [-1, -1] }>,
    ) {
        let pid = program_id(0);
        let local: Tile<i32, { [B] }> = iota(shape![B]);
        let indices = local + (pid * B).broadcast(shape![B]);
        let valid = lt_tile(indices, updates.shape()[1].broadcast(shape![B]));
        let parts = updates.partition(shape![1, B]);
        let offsets = parts.load([0, pid]).reshape(shape![B]);
        let slots = parts.load([1, pid]).reshape(shape![B]);
        let pointer: PointerTile<*mut i32, { [] }> = pointer_to_tile(table_ptr);
        let pointer: PointerTile<*mut i32, { [1] }> = pointer.reshape(shape![1]);
        let pointer: PointerTile<*mut i32, { [B] }> = pointer.broadcast(shape![B]);
        let destinations = pointer.offset_tile(offsets);
        unsafe {
            store_ptr_tko(
                destinations,
                slots,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(valid),
                None,
                Latency::<1>,
            );
        }
    }

    /// Scatter [B, D] K/V tiles: each program handles B tokens, masking tail rows.
    /// slot_table maps logical positions to potentially noncontiguous physical slots,
    /// so each tile element writes through its computed pointer instead of a contiguous slice.
    /// Example: slots=[10,25,7] writes the three token vectors to K/V rows 10, 25, 7.
    #[cutile::entry()]
    pub(super) unsafe fn store<T: ElementType, const B: i32, const D: i32>(
        cache_k_ptr: *mut T,
        cache_v_ptr: *mut T,
        table: &Tensor<i32, { [-1, -1] }>,
        k: &Tensor<T, { [-1, D] }>,
        v: &Tensor<T, { [-1, D] }>,
        row: i32,
        start: i32,
        count: i32,
    ) {
        let pid = program_id(0);
        let local: Tile<i32, { [B] }> = iota(shape![B]);
        let indices = local + (pid * B).broadcast(shape![B]);
        let valid = lt_tile(indices, count.broadcast(shape![B]));
        let table_base: PointerTile<*const i32, { [] }> = pointer_to_tile(table.as_ptr());
        let table_base: PointerTile<*const i32, { [] }> =
            table_base.offset(row * table.shape()[1] + start);
        let table_base: PointerTile<*const i32, { [1] }> = table_base.reshape(shape![1]);
        let table_base: PointerTile<*const i32, { [B] }> = table_base.broadcast(shape![B]);
        let table_ptrs: PointerTile<*const i32, { [B] }> = table_base.offset_tile(indices);
        let (slots, _token): (Tile<i32, { [B] }>, Token) = unsafe {
            load_ptr_tko(
                table_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(valid),
                Some(0i32),
                None,
                Latency::<1>,
            )
        };
        let slot_offsets = slots * D.broadcast(shape![B]);
        let slot_offsets: Tile<i32, { [B, D] }> =
            slot_offsets.reshape(shape![B, 1]).broadcast(shape![B, D]);
        let columns: Tile<i32, { [D] }> = iota(shape![D]);
        let columns: Tile<i32, { [B, D] }> = columns.reshape(shape![1, D]).broadcast(shape![B, D]);
        let offsets = slot_offsets + columns;
        let mask: Tile<bool, { [B, D] }> = valid.reshape(shape![B, 1]).broadcast(shape![B, D]);
        let kp: PointerTile<*mut T, { [] }> = pointer_to_tile(cache_k_ptr);
        let kp: PointerTile<*mut T, { [1, 1] }> = kp.reshape(shape![1, 1]);
        let kp: PointerTile<*mut T, { [B, D] }> = kp.broadcast(shape![B, D]);
        let kp: PointerTile<*mut T, { [B, D] }> = kp.offset_tile(offsets);
        let vp: PointerTile<*mut T, { [] }> = pointer_to_tile(cache_v_ptr);
        let vp: PointerTile<*mut T, { [1, 1] }> = vp.reshape(shape![1, 1]);
        let vp: PointerTile<*mut T, { [B, D] }> = vp.broadcast(shape![B, D]);
        let vp: PointerTile<*mut T, { [B, D] }> = vp.offset_tile(offsets);
        // Partition loads zero-pad source rows beyond the tensor; masked scatters
        // ensure those padding lanes never read table cells or overwrite slot zero.
        let keys = k.partition(shape![B, D]).load([pid, 0]);
        let values = v.partition(shape![B, D]).load([pid, 0]);
        // store_ptr_tko scatters values to the pointer tile; tko means Token-Ordered.
        // Its optional input Token and returned Token express GPU operation dependencies,
        // unrelated to text tokens or CPU-side awaiting. These stores pass no input Token.
        unsafe {
            store_ptr_tko(
                kp,
                keys,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<1>,
            );
            store_ptr_tko(
                vp,
                values,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<1>,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::kv_pool::{KVContext, KVContextError};
    use anyhow::Result;
    use cuda_async::{
        device_context::global_policy,
        device_operation::{value, with_context},
    };
    use cutile::tensor::{Reshape, ToHostVec};

    /// Cache/path pins retain physical slots without rows; children append unique suffixes.
    #[tokio::test]
    async fn shared_prefix_ownership_is_rowless_immutable_and_released_once() -> Result<()> {
        use crate::runner::{prefix_cache::PrefixCache, PrefixCacheConfig};
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = global_policy(0)?;
        let pool = NaiveKVPool::<f32>::new(
            NaiveKVPoolConfig {
                num_slots: 8,
                max_rows: 2,
                max_seq_len: 4,
                head_dim: 4,
                ..Default::default()
            },
            &policy,
        )
        .await?;
        let initial = pool.metrics();
        let cache = PrefixCache::new(PrefixCacheConfig::default(), pool.clone());
        let mut source = KVContext::new(pool.clone())?;
        assert!(source.take_prefix(0..1).is_err());
        with_context(|exec| value(source.prepare_append(2, exec)))
            .schedule(&policy)?
            .await??;
        assert!(cache.publish(&[1, 2], &mut source).is_err());
        source.commit_append()?;
        let prefix_slots = source.state().lease.slots.lock().unwrap().mapping.clone();
        cache.publish(&[1, 2], &mut source)?;
        source.release();
        assert_eq!(pool.metrics().index.assigned_bytes, 0);
        assert_eq!(
            pool.metrics().layers.assigned_bytes,
            2 * 2 * 4 * size_of::<f32>()
        );
        let mut a = KVContext::new(pool.clone())?;
        let mut b = KVContext::new(pool.clone())?;
        let hit = cache.find(&[1, 2, 3]).unwrap();
        cache.reuse(&hit, &mut a)?;
        cache.reuse(&hit, &mut b)?;
        assert!(cache.reuse(&hit, &mut a).is_err());
        assert!(
            pool.alloc_state().is_err(),
            "cache owners consume no extra rows"
        );
        drop(hit);
        drop(cache); // Row leases now keep the tree and its path protection alive.
        with_context(|exec| {
            value((|| -> Result<()> {
                assert!(
                    a.write_view(0, exec).is_err(),
                    "idle shared contexts are not writable"
                );
                KVContext::prepare_append_batch(&mut [(&mut a, 1), (&mut b, 1)], exec)?;
                assert_eq!(a.write_view(0, exec)?.positions(), &(2..3));
                assert_eq!(b.write_view(0, exec)?.positions(), &(2..3));
                Ok(())
            })())
        })
        .schedule(&policy)?
        .await??;
        a.commit_append()?;
        b.commit_append()?;
        let slots_a = a.state().lease.slots.lock().unwrap().mapping.clone();
        let slots_b = b.state().lease.slots.lock().unwrap().mapping.clone();
        assert_eq!(&slots_a[..2], &prefix_slots);
        assert_eq!(&slots_b[..2], &prefix_slots);
        assert_ne!(slots_a[2], slots_b[2]);
        let table = pool
            .slot_table
            .clone()
            .to_host_vec()
            .schedule(&policy)?
            .await?;
        for state in [a.state(), b.state()] {
            let row = state.lease.row * 4;
            assert_eq!(
                &table[row..row + 3],
                &state.lease.slots.lock().unwrap().mapping
            );
        }
        assert_eq!(
            pool.metrics().layers.assigned_bytes,
            4 * 2 * 4 * size_of::<f32>()
        );
        a.release();
        assert_eq!(
            pool.metrics().layers.assigned_bytes,
            3 * 2 * 4 * size_of::<f32>()
        );
        b.release();
        assert_eq!(pool.metrics(), initial);
        Ok(())
    }

    /// CPU reservation is separate from GPU binding; the plan consumes only its
    /// subrange in a larger upload, including an unaligned prefix and tile tail.
    #[tokio::test]
    async fn deferred_mapping_binds_only_after_submit_from_packed_subrange() -> Result<()> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = global_policy(0)?;
        let pool = NaiveKVPool::<f32>::new(
            NaiveKVPoolConfig {
                num_slots: 512,
                max_rows: 2,
                max_seq_len: 256,
                head_dim: 4,
                ..Default::default()
            },
            &policy,
        )
        .await?;
        let uploads = Arc::new(UploadPool::default());
        let initial = pool.metrics();
        let mut contexts = [KVContext::new(pool.clone())?, KVContext::new(pool.clone())?];
        let counts = [129, 3];
        let table = with_context(|exec| {
            value((|| -> Result<_> {
                let mut appends: Vec<_> = contexts.iter_mut().zip(counts).collect();
                let plan = KVContext::reserve_append_batch(&mut appends, exec)?;
                for (context, count) in contexts.iter().zip(counts) {
                    assert_eq!(context.ready_len(), 0);
                    assert_eq!(context.allocated_len(), count);
                }
                // SAFETY: this read and later bind are ordered on the same exec.
                let before = unsafe { pool.slot_table.clone().to_host_vec().execute(exec)? };
                assert!(
                    before.iter().all(|&slot| slot == 0),
                    "reservation submitted GPU mapping work"
                );
                let mut packed = vec![12345i32; 7];
                packed.extend_from_slice(plan.payload());
                packed.extend([54321; 5]);
                let len = packed.len();
                // SAFETY: the uploaded plan is unchanged and all work uses exec.
                unsafe {
                    let upload = uploads.upload(Arc::new(packed), &[len], exec)?;
                    assert!(plan.submit(&upload, usize::MAX, exec).is_err());
                    plan.submit(&upload, 7, exec)?;
                    Ok(pool.slot_table.clone().to_host_vec().execute(exec)?)
                }
            })())
        })
        .schedule(&policy)?
        .await??;
        let mut expected = vec![0i32; 2 * 256];
        for context in &contexts {
            let state = context.state();
            let slots = state.lease.slots.lock().unwrap();
            let base = state.lease.row * 256;
            expected[base..base + slots.mapping.len()].copy_from_slice(&slots.mapping);
        }
        assert_eq!(table, expected);
        for context in &mut contexts {
            context.commit_append()?;
        }
        drop(contexts);
        assert_eq!(pool.metrics(), initial);
        Ok(())
    }

    /// Dropping an unsubmitted plan on failure never advances logical progress or
    /// recycles reserved slots before the submission releases its state leases.
    #[tokio::test]
    async fn unsubmitted_mapping_plan_failure_releases_pending_reservations() -> Result<()> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = global_policy(0)?;
        let pool = NaiveKVPool::<f32>::new(
            NaiveKVPoolConfig {
                num_slots: 4,
                max_rows: 2,
                max_seq_len: 4,
                head_dim: 4,
                ..Default::default()
            },
            &policy,
        )
        .await?;
        let initial = pool.metrics();
        let mut a = KVContext::new(pool.clone())?;
        let mut b = KVContext::new(pool.clone())?;
        let result = with_context(|exec| {
            value((|| -> Result<()> {
                let plan = KVContext::reserve_append_batch(&mut [(&mut a, 2), (&mut b, 1)], exec)?;
                assert_eq!(a.ready_len(), 0);
                assert_eq!(b.ready_len(), 0);
                drop(plan);
                a.release();
                b.release();
                assert_eq!(pool.metrics().layers.assigned_bytes, 3 * 2 * 4 * 4);
                anyhow::bail!("injected failure before mapping submit")
            })())
        })
        .schedule(&policy)?
        .await?;
        assert!(result.is_err());
        assert_eq!(pool.metrics(), initial);
        Ok(())
    }

    /// Batch mappings cover multiple rows, nonzero append starts, and partial tiles.
    #[tokio::test]
    async fn batch_mapping_preserves_prefixes_and_masks_tails() -> Result<()> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = global_policy(0)?;
        let pool = NaiveKVPool::<f32>::new(
            NaiveKVPoolConfig {
                num_slots: 512,
                max_rows: 3,
                max_seq_len: 256,
                head_dim: 4,
                ..Default::default()
            },
            &policy,
        )
        .await?;
        let initial = pool.metrics();
        let mut contexts = (0..3)
            .map(|_| KVContext::new(pool.clone()))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut ready = [0; 3];
        let mut expected = vec![0i32; 3 * 256];
        for counts in [[129, 1, 7], [3, 128, 2]] {
            let table = with_context(|exec| {
                value((|| -> Result<_> {
                    let mut appends: Vec<_> = contexts.iter_mut().zip(counts).collect();
                    KVContext::prepare_append_batch(&mut appends, exec)?;
                    for (i, context) in contexts.iter().enumerate() {
                        assert_eq!(context.ready_len(), ready[i]);
                        assert_eq!(context.allocated_len(), ready[i] + counts[i]);
                        let state = context.state();
                        let slots = state.lease.slots.lock().unwrap();
                        let base = state.lease.row * 256;
                        expected[base..base + slots.mapping.len()].copy_from_slice(&slots.mapping);
                    }
                    // SAFETY: the outer future completes before CPU inspection.
                    unsafe { Ok(pool.slot_table.clone().to_host_vec().execute(exec)?) }
                })())
            })
            .schedule(&policy)?
            .await??;
            assert_eq!(table, expected);
            for (i, context) in contexts.iter_mut().enumerate() {
                context.commit_append()?;
                ready[i] += counts[i];
                assert_eq!(context.ready_len(), ready[i]);
            }
        }
        drop(contexts);
        assert_eq!(pool.metrics(), initial);
        Ok(())
    }

    /// Reject logical errors before mutation; partial backend OOM remains releasable.
    #[tokio::test]
    async fn batch_mapping_failure_preserves_progress_and_releases_slots() -> Result<()> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = global_policy(0)?;
        let pool = NaiveKVPool::<f32>::new(
            NaiveKVPoolConfig {
                num_slots: 3,
                max_rows: 2,
                max_seq_len: 4,
                head_dim: 4,
                ..Default::default()
            },
            &policy,
        )
        .await?;
        let initial = pool.metrics();
        let mut a = KVContext::new(pool.clone())?;
        let mut b = KVContext::new(pool.clone())?;
        with_context(|exec| {
            value((|| -> Result<()> {
                assert!(
                    KVContext::prepare_append_batch(&mut [(&mut a, 1), (&mut b, 0)], exec).is_err()
                );
                assert!(!a.is_pending());
                assert!(!b.is_pending());
                assert!(matches!(
                    KVContext::prepare_append_batch(&mut [(&mut a, 2), (&mut b, 2)], exec),
                    Err(KVContextError::Pool(KVPoolError::CapacityExhausted(_)))
                ));
                assert!(a.is_pending() && b.is_pending());
                assert_eq!(a.ready_len(), 0);
                assert_eq!(b.ready_len(), 0);
                // Attempted boundaries stay pending after failure; neither context is reusable.
                assert_eq!(a.allocated_len(), 2);
                assert_eq!(b.allocated_len(), 2);
                assert_eq!(a.pending_range(), Some(0..2));
                assert_eq!(b.pending_range(), Some(0..2));
                assert!(a.prepare_append(1, exec).is_err());
                assert!(b.prepare_append(1, exec).is_err());
                // The partially reserved slots stay leased until outer completion.
                a.release();
                b.release();
                assert_eq!(pool.metrics().layers.assigned_bytes, 2 * 2 * 4 * 4);
                Ok(())
            })())
        })
        .schedule(&policy)?
        .await??;
        assert_eq!(pool.metrics(), initial);
        Ok(())
    }

    /// Explicit store/readback checks the physical scatter and GPU logical mapping.
    #[tokio::test]
    async fn store_reads_back_through_slots_and_releases() -> Result<()> {
        let _guard = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = global_policy(0)?;
        let pool = NaiveKVPool::<f32>::new(
            NaiveKVPoolConfig {
                num_slots: 4,
                max_rows: 1,
                max_seq_len: 4,
                head_dim: 4,
                ..Default::default()
            },
            &policy,
        )
        .await?;
        let initial = pool.metrics();
        let mut kv = KVContext::new(pool.clone())?;
        assert!(matches!(
            KVContext::new(pool.clone()),
            Err(KVContextError::Pool(KVPoolError::CapacityExhausted(_)))
        ));
        assert!(matches!(
            kv.commit_append(),
            Err(KVContextError::InvalidState(_))
        ));
        let data = Arc::new(vec![1.0f32, 2.0, 3.0, 4.0, 9.0, 8.0, 7.0, 6.0]);
        let k = Arc::new(
            api::copy_host_vec_to_device(&data)
                .schedule(&policy)?
                .await?
                .reshape(&[2, 4])?,
        );
        let v = Arc::new(
            api::copy_host_vec_to_device(&data)
                .schedule(&policy)?
                .await?
                .reshape(&[2, 4])?,
        );
        let (indices, keys, values) = with_context(|exec| {
            value((|| -> Result<_> {
                kv.prepare_append(2, exec)?;
                assert_eq!(kv.ready_len(), 0);
                assert_eq!(kv.allocated_len(), 2);
                assert!(kv.prepare_append(1, exec).is_err());
                kv.store(0, k.view(&[2, 4])?, v.view(&[2, 4])?, exec)?;
                let view = kv.read_view(0)?;
                // SAFETY: the outer future is awaited before host access.
                unsafe {
                    Ok((
                        view.inner.slot_table.clone().to_host_vec().execute(exec)?,
                        view.inner.k.clone().to_host_vec().execute(exec)?,
                        view.inner.v.clone().to_host_vec().execute(exec)?,
                    ))
                }
            })())
        })
        .schedule(&policy)?
        .await??;
        kv.commit_append()?;
        assert_eq!(kv.ready_len(), 2);
        for i in 0..2 {
            let slot = indices[i] as usize;
            assert_eq!(&keys[slot * 4..slot * 4 + 4], &data[i * 4..i * 4 + 4]);
            assert_eq!(&values[slot * 4..slot * 4 + 4], &data[i * 4..i * 4 + 4]);
        }
        assert_eq!(pool.metrics().index.internal_unused_bytes, 2 * 4);
        assert_eq!(pool.metrics().layers.assigned_bytes, 2 * 2 * 4 * 4);
        kv.release();
        assert_eq!(pool.metrics(), initial);
        assert!(KVContext::new(pool.clone()).is_ok());
        Ok(())
    }

    /// Cover bind/store tile boundaries, masked tails, append starts, and separate layers.
    #[tokio::test]
    async fn tiled_store_preserves_prefixes_and_masks_tail_tokens() -> Result<()> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = global_policy(0)?;
        let pool = NaiveKVPool::<f32>::new(
            NaiveKVPoolConfig {
                num_layers: 2,
                num_slots: 512,
                max_rows: 2,
                max_seq_len: 512,
                head_dim: 4,
            },
            &policy,
        )
        .await?;
        let initial = pool.metrics();
        let mut kv = KVContext::new(pool.clone())?;
        let mut total = 0usize;
        let b = STORE_TOKENS_PER_TILE;
        let bind_b = BIND_TOKENS_PER_TILE;
        for count in [
            1,
            b - 1,
            b,
            b + 1,
            2 * b + 1,
            bind_b - 1,
            bind_b,
            bind_b + 1,
        ] {
            let mut inputs = Vec::new();
            for layer in 0..2 {
                let keys: Vec<f32> = (total..total + count)
                    .flat_map(|token| {
                        (0..4).map(move |column| (1 + layer * 1000 + token * 4 + column) as f32)
                    })
                    .collect();
                let values: Vec<f32> = keys.iter().map(|value| -*value).collect();
                let k = Arc::new(
                    api::copy_host_vec_to_device(&Arc::new(keys))
                        .schedule(&policy)?
                        .await?
                        .reshape(&[count, 4])?,
                );
                let v = Arc::new(
                    api::copy_host_vec_to_device(&Arc::new(values))
                        .schedule(&policy)?
                        .await?
                        .reshape(&[count, 4])?,
                );
                inputs.push((k, v));
            }
            with_context(|exec| {
                value((|| -> Result<()> {
                    assert!(matches!(
                        kv.write_view(0, exec),
                        Err(KVContextError::InvalidState(_))
                    ));
                    kv.prepare_append(count, exec)?;
                    for (layer, (k, v)) in inputs.iter().enumerate() {
                        let dst = kv.write_view(layer, exec)?;
                        assert_eq!(dst.positions(), &(total..total + count));
                        pool.store(dst, k.slice(&[0..count])?, v.slice(&[0..count])?)?;
                    }
                    assert_eq!(kv.read_view(0)?.positions(), &(0..total + count));
                    assert_eq!(kv.ready_len(), total);
                    Ok(())
                })())
            })
            .schedule(&policy)?
            .await??;
            kv.commit_append()?;
            total += count;
        }
        for layer in 0..2 {
            let view = kv.read_view(layer)?;
            assert_eq!(view.positions().clone(), 0..total);
            let indices = view
                .inner
                .slot_table
                .clone()
                .to_host_vec()
                .schedule(&policy)?
                .await?;
            let k = view
                .inner
                .k
                .clone()
                .to_host_vec()
                .schedule(&policy)?
                .await?;
            let v = view
                .inner
                .v
                .clone()
                .to_host_vec()
                .schedule(&policy)?
                .await?;
            for (token, &slot) in indices.iter().take(total).enumerate() {
                for column in 0..4 {
                    let expected = (1 + layer * 1000 + token * 4 + column) as f32;
                    assert_eq!(k[slot as usize * 4 + column], expected);
                    assert_eq!(v[slot as usize * 4 + column], -expected);
                }
            }
        }
        assert!(pool.read_view(kv.state(), 0, 0..total + 1).is_err());
        assert_eq!(
            pool.read_view(kv.state(), 0, 5..total)?.positions().clone(),
            5..total
        );
        kv.release();
        assert_eq!(pool.metrics(), initial);
        Ok(())
    }

    /// Releasing a pending context cannot recycle its row or slots during submission.
    #[tokio::test]
    async fn submission_lease_delays_release_and_oom_does_not_leak() -> Result<()> {
        let _guard = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = global_policy(0)?;
        let pool = NaiveKVPool::<f32>::new(
            NaiveKVPoolConfig {
                num_slots: 2,
                max_rows: 2,
                max_seq_len: 4,
                ..Default::default()
            },
            &policy,
        )
        .await?;
        let initial = pool.metrics();
        let mut kv = KVContext::new(pool.clone())?;
        with_context(|exec| {
            value((|| -> Result<()> {
                kv.prepare_append(2, exec)?;
                kv.release();
                assert_eq!(pool.metrics().layers.free_bytes, 0);
                assert_eq!(pool.metrics().index.assigned_bytes, 4 * 4);
                Ok(())
            })())
        })
        .schedule(&policy)?
        .await??;
        assert_eq!(pool.metrics(), initial);
        let mut kv = KVContext::new(pool.clone())?;
        let result = with_context(|exec| value(kv.prepare_append(3, exec)))
            .schedule(&policy)?
            .await?;
        assert!(matches!(
            result,
            Err(KVContextError::Pool(KVPoolError::CapacityExhausted(_)))
        ));
        assert_eq!(kv.ready_len(), 0);
        assert_eq!(pool.metrics().layers.assigned_bytes, 0);
        kv.release();
        assert_eq!(pool.metrics(), initial);
        Ok(())
    }
}
