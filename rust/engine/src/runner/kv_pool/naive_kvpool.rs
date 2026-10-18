use std::{
    ops::Range,
    sync::{Arc, Mutex},
};

use cuda_async::{
    device_operation::{DeviceOp, ExecutionContext},
    scheduling_policies::SchedulingPolicy,
};
use cutile::{
    api,
    tensor::{Tensor, TensorView},
    tile_kernel::TileKernel,
};

use super::{BufferMetrics, KVPool, KVPoolError, KVPoolMetrics};

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
pub struct NaiveKVPool {
    /// 按 layer + slot 存储，逻辑 context 经 slot_table 寻址，后续可共享前缀。
    /// context 若独占连续 KV 区域，fork 就需要复制历史；间接映射解除了这一限制。
    layers: Vec<(Arc<Tensor<f32>>, Arc<Tensor<f32>>)>,
    /// token_position → slot mapping
    slot_table: Arc<Tensor<i32>>,
    config: NaiveKVPoolConfig,
    allocation: Arc<Mutex<Allocation>>,
}

struct Allocation {
    free_rows: Vec<usize>,
    free_slots: Vec<i32>,
    bound_positions: usize,
}

/// A submission retains this lease, not just the pool buffers, to prevent slot reuse.
struct StateLease {
    row: usize,
    slots: Mutex<Vec<i32>>,
    allocation: Arc<Mutex<Allocation>>,
}

impl Drop for StateLease {
    fn drop(&mut self) {
        let slots = self.slots.get_mut().unwrap();
        let mut allocation = self.allocation.lock().unwrap();
        allocation.bound_positions -= slots.len();
        allocation.free_slots.append(slots);
        allocation.free_rows.push(self.row);
    }
}

pub struct NaiveContextState {
    lease: Arc<StateLease>,
}

pub struct NaiveLayerView<'a> {
    pub(crate) k: &'a Arc<Tensor<f32>>,
    pub(crate) v: &'a Arc<Tensor<f32>>,
    pub(crate) slot_table: &'a Arc<Tensor<i32>>,
    pub(crate) row: usize,
    /// Logical KV positions covering history plus the current queries' K/V during
    /// append forward; the attention backend further restricts reads per query.
    pub(crate) read_range: Range<usize>,
    lease: Arc<StateLease>,
}

impl NaiveLayerView<'_> {
    /// Protect logical allocation ownership before attention submits a read.
    pub(crate) fn retain(&self, exec: &ExecutionContext) -> Result<()> {
        exec.retain(self.lease.clone())?;
        Ok(())
    }
}

impl NaiveKVPool {
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
            let k = api::zeros::<f32>(&[config.num_slots, config.head_dim])
                .schedule(policy)?
                .await?;
            let v = api::zeros::<f32>(&[config.num_slots, config.head_dim])
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

    fn validate_state(&self, state: &NaiveContextState) -> Result<()> {
        if !Arc::ptr_eq(&self.allocation, &state.lease.allocation) {
            return Err(KVPoolError::InvalidInput(
                "KV state belongs to a different pool".into(),
            ));
        }
        Ok(())
    }
}

impl KVPool for NaiveKVPool {
    type ContextState = NaiveContextState;
    type LayerView<'a> = NaiveLayerView<'a>;

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
    /// incrementally. The row and slots are recycled when the final GPU lease drops.
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
                slots: Mutex::new(Vec::new()),
                allocation: self.allocation.clone(),
            }),
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
        self.validate_state(state)?;
        if exec.get_device_id() != self.slot_table.device_id() {
            return Err(KVPoolError::InvalidInput(
                "KV pool and execution device differ".into(),
            ));
        }
        let mut slots = state.lease.slots.lock().unwrap();
        if positions.start != slots.len()
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
        slots.extend_from_slice(&new_slots);
        drop(slots);
        // SAFETY: exec owns the submission; its caller establishes completion.
        unsafe {
            let indices = api::copy_host_vec_to_device(&Arc::new(new_slots)).execute(exec)?;
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
                positions.start as i32,
            )
            .generics(vec![BIND_TOKENS_PER_TILE.to_string()])
            .grid((positions.len().div_ceil(BIND_TOKENS_PER_TILE) as u32, 1, 1))
            .execute(exec)?;
        }
        Ok(())
    }

    fn layer_view<'a>(
        &'a self,
        state: &'a Self::ContextState,
        layer: usize,
        read_range: Range<usize>,
    ) -> Result<Self::LayerView<'a>> {
        self.validate_state(state)?;
        if read_range.start > read_range.end
            || read_range.end > state.lease.slots.lock().unwrap().len()
        {
            return Err(KVPoolError::InvalidInput(
                "KV read range exceeds mapped tokens".into(),
            ));
        }
        let (k, v) = self
            .layers
            .get(layer)
            .ok_or_else(|| KVPoolError::InvalidInput("invalid KV layer".into()))?;
        Ok(NaiveLayerView {
            k,
            v,
            slot_table: &self.slot_table,
            row: state.lease.row,
            read_range,
            lease: state.lease.clone(),
        })
    }

    /// Enqueue K/V scatter writes through slot_table on exec, without waiting.
    /// The same stream orders mapping updates before writes and attention afterward;
    /// returning successfully means submission succeeded, not that GPU writes finished.
    /// Each [B, head_dim] tile handles up to B tokens (B=8); the last tile is masked.
    fn store(
        &self,
        state: &Self::ContextState,
        layer: usize,
        positions: Range<usize>,
        k: TensorView<'_, f32>,
        v: TensorView<'_, f32>,
        exec: &ExecutionContext,
    ) -> Result<()> {
        self.validate_state(state)?;
        let (cache_k, cache_v) = self
            .layers
            .get(layer)
            .ok_or_else(|| KVPoolError::InvalidInput("invalid KV layer".into()))?;
        let width = self.config.head_dim;
        if positions.is_empty()
            || exec.get_device_id() != cache_k.device_id()
            || k.shape() != v.shape()
            || k.shape().len() != 2
            || k.shape()[0] < positions.len() as i32
            || k.shape()[1] != width as i32
            || positions.end > state.lease.slots.lock().unwrap().len()
        {
            return Err(KVPoolError::InvalidInput(
                "invalid KV store shape or range".into(),
            ));
        }
        exec.retain(state.lease.clone())?;
        // SAFETY: reserved slots are unique to this state; mapping updates precede this
        // scatter on exec's stream. Raw output buffers are retained explicitly;
        // generated launchers retain input tensor storage. Raw argument names differ
        // from input names to avoid collisions with generated <name>_ptr bindings.
        unsafe {
            exec.retain(cache_k.clone())?;
            exec.retain(cache_v.clone())?;
            kernels::store(
                cache_k.device_pointer(),
                cache_v.device_pointer(),
                self.slot_table.clone(),
                &k,
                &v,
                state.lease.row as i32,
                positions.start as i32,
                positions.len() as i32,
            )
            .generics(vec![STORE_TOKENS_PER_TILE.to_string(), width.to_string()])
            .grid((positions.len().div_ceil(STORE_TOKENS_PER_TILE) as u32, 1, 1))
            .execute(exec)?;
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
        let slot_bytes = c.num_layers * 2 * c.head_dim * size_of::<f32>();
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

    /// Scatter [B, D] K/V tiles: each program handles B tokens, masking tail rows.
    /// slot_table maps logical positions to potentially noncontiguous physical slots,
    /// so each tile element writes through its computed pointer instead of a contiguous slice.
    /// Example: slots=[10,25,7] writes the three token vectors to K/V rows 10, 25, 7.
    #[cutile::entry()]
    pub(super) unsafe fn store<const B: i32, const D: i32>(
        cache_k_ptr: *mut f32,
        cache_v_ptr: *mut f32,
        table: &Tensor<i32, { [-1, -1] }>,
        k: &Tensor<f32, { [-1, D] }>,
        v: &Tensor<f32, { [-1, D] }>,
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
        let kp: PointerTile<*mut f32, { [] }> = pointer_to_tile(cache_k_ptr);
        let kp: PointerTile<*mut f32, { [1, 1] }> = kp.reshape(shape![1, 1]);
        let kp: PointerTile<*mut f32, { [B, D] }> = kp.broadcast(shape![B, D]);
        let kp: PointerTile<*mut f32, { [B, D] }> = kp.offset_tile(offsets);
        let vp: PointerTile<*mut f32, { [] }> = pointer_to_tile(cache_v_ptr);
        let vp: PointerTile<*mut f32, { [1, 1] }> = vp.reshape(shape![1, 1]);
        let vp: PointerTile<*mut f32, { [B, D] }> = vp.broadcast(shape![B, D]);
        let vp: PointerTile<*mut f32, { [B, D] }> = vp.offset_tile(offsets);
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

    /// Explicit store/readback checks the physical scatter and GPU logical mapping.
    #[tokio::test]
    async fn store_reads_back_through_slots_and_releases() -> Result<()> {
        let _guard = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = global_policy(0)?;
        let pool = NaiveKVPool::new(
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
                let view = kv.layer_view(0)?;
                // SAFETY: the outer future is awaited before host access.
                unsafe {
                    Ok((
                        view.slot_table.clone().to_host_vec().execute(exec)?,
                        view.k.clone().to_host_vec().execute(exec)?,
                        view.v.clone().to_host_vec().execute(exec)?,
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
        let pool = NaiveKVPool::new(
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
                    kv.prepare_append(count, exec)?;
                    for (layer, (k, v)) in inputs.iter().enumerate() {
                        kv.store(layer, k.slice(&[0..count])?, v.slice(&[0..count])?, exec)?;
                    }
                    Ok(())
                })())
            })
            .schedule(&policy)?
            .await??;
            kv.commit_append()?;
            total += count;
        }
        for layer in 0..2 {
            let view = kv.layer_view(layer)?;
            assert_eq!(view.read_range, 0..total);
            let indices = view
                .slot_table
                .clone()
                .to_host_vec()
                .schedule(&policy)?
                .await?;
            let k = view.k.clone().to_host_vec().schedule(&policy)?.await?;
            let v = view.v.clone().to_host_vec().schedule(&policy)?.await?;
            for (token, &slot) in indices.iter().take(total).enumerate() {
                for column in 0..4 {
                    let expected = (1 + layer * 1000 + token * 4 + column) as f32;
                    assert_eq!(k[slot as usize * 4 + column], expected);
                    assert_eq!(v[slot as usize * 4 + column], -expected);
                }
            }
        }
        assert!(pool.layer_view(kv.state(), 0, 0..total + 1).is_err());
        assert_eq!(
            pool.layer_view(kv.state(), 0, 5..total)?.read_range,
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
        let pool = NaiveKVPool::new(
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
