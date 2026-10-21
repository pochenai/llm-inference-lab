// MLP kernels adapted from NVIDIA cutile-examples/examples/async_mlp.rs.
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
pub(super) use kernels::*;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub(super) fn embedding(
        out: &mut Tensor<f32, { [1, 16] }>,
        packed_token_ids: &Tensor<i32, { [-1] }>,
        weights: &Tensor<f32, { [-1, 16] }>,
        divisor: f32,
    ) {
        let i = program_id(0);
        let id: i32 = tile_to_scalar(
            packed_token_ids
                .partition(shape![1])
                .load([i])
                .reshape(shape![]),
        );
        out.store(
            weights.partition(shape![1, 16]).load([id, 0]) / divisor.broadcast(shape![1, 16]),
        );
    }

    /// Compute Q/K/V together; only Q is materialized as a temporary tensor.
    /// K/V accumulators scatter directly into mapped cache slots. Invalid packed rows
    /// are masked, so padding cannot overwrite another context or an unused slot.
    #[cutile::entry()]
    pub(super) unsafe fn qkv_to_cache<const BM: i32, const BN: i32, const BK: i32>(
        q: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, 16] }>,
        wq: &Tensor<f32, { [16, 16] }>,
        wk: &Tensor<f32, { [16, 16] }>,
        wv: &Tensor<f32, { [16, 16] }>,
        cache_k_ptr: *mut f32,
        cache_v_ptr: *mut f32,
        slot_table_base: *const i32,
        metadata: &Tensor<i32, { [-1, 3] }>,
        table_width: i32,
    ) {
        let pid = get_tile_block_id();
        let part_x = x.partition(shape![BM, BK]);
        let part_q = wq.partition(shape![BK, BN]);
        let part_k = wk.partition(shape![BK, BN]);
        let part_v = wv.partition(shape![BK, BN]);
        let mut q_values: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        let mut k_values: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        let mut v_values: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        for i in 0i32..(16 / BK) {
            let inputs = part_x.load([pid.0, i]);
            q_values = mma(inputs, part_q.load([i, pid.1]), q_values);
            k_values = mma(inputs, part_k.load([i, pid.1]), k_values);
            v_values = mma(inputs, part_v.load([i, pid.1]), v_values);
        }
        q.store(q_values);

        let context_rows: Tile<i32, { [BM] }> = metadata
            .partition(shape![BM, 1])
            .load([pid.0, 0])
            .reshape(shape![BM]);
        let positions: Tile<i32, { [BM] }> = metadata
            .partition(shape![BM, 1])
            .load([pid.0, 1])
            .reshape(shape![BM]);
        let padding_marker: Tile<i32, { [BM] }> = constant(-1i32, shape![BM]);
        let valid = lt_tile(padding_marker, context_rows);
        let table_offsets = context_rows * table_width.broadcast(shape![BM]) + positions;
        let tp: PointerTile<*const i32, { [] }> = pointer_to_tile(slot_table_base);
        let tp: PointerTile<*const i32, { [1] }> = tp.reshape(shape![1]);
        let tp: PointerTile<*const i32, { [BM] }> = tp.broadcast(shape![BM]);
        let tp: PointerTile<*const i32, { [BM] }> = tp.offset_tile(table_offsets);
        let (slots, _token): (Tile<i32, { [BM] }>, Token) = unsafe {
            load_ptr_tko(
                tp,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(valid),
                Some(0i32),
                None,
                Latency::<1>,
            )
        };
        let row_offsets = slots * 16i32.broadcast(shape![BM]);
        let row_offsets: Tile<i32, { [BM, BN] }> =
            row_offsets.reshape(shape![BM, 1]).broadcast(shape![BM, BN]);
        let columns: Tile<i32, { [BN] }> = iota(shape![BN]);
        let columns = columns + (pid.1 * BN).broadcast(shape![BN]);
        let columns: Tile<i32, { [BM, BN] }> =
            columns.reshape(shape![1, BN]).broadcast(shape![BM, BN]);
        let offsets = row_offsets + columns;
        let mask: Tile<bool, { [BM, BN] }> = valid.reshape(shape![BM, 1]).broadcast(shape![BM, BN]);
        let kp: PointerTile<*mut f32, { [] }> = pointer_to_tile(cache_k_ptr);
        let kp: PointerTile<*mut f32, { [1, 1] }> = kp.reshape(shape![1, 1]);
        let kp: PointerTile<*mut f32, { [BM, BN] }> = kp.broadcast(shape![BM, BN]);
        let kp: PointerTile<*mut f32, { [BM, BN] }> = kp.offset_tile(offsets);
        let vp: PointerTile<*mut f32, { [] }> = pointer_to_tile(cache_v_ptr);
        let vp: PointerTile<*mut f32, { [1, 1] }> = vp.reshape(shape![1, 1]);
        let vp: PointerTile<*mut f32, { [BM, BN] }> = vp.broadcast(shape![BM, BN]);
        let vp: PointerTile<*mut f32, { [BM, BN] }> = vp.offset_tile(offsets);
        // tko = Token-Ordered: these stores use computed addresses and value dependencies.
        unsafe {
            store_ptr_tko(
                kp,
                k_values,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<1>,
            );
            store_ptr_tko(
                vp,
                v_values,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<1>,
            );
        }
    }

    #[cutile::entry()]
    pub(super) fn attention(
        out: &mut Tensor<f32, { [1, 16] }>,
        q: &Tensor<f32, { [-1, 16] }>,
        k: &Tensor<f32, { [-1, 16] }>,
        v: &Tensor<f32, { [-1, 16] }>,
        table: &Tensor<i32, { [-1, -1] }>,
        metadata: &Tensor<i32, { [-1, 3] }>,
    ) {
        let i = program_id(0);
        let mut acc: Tile<f32, { [1, 16] }> = constant(0.0f32, shape![1, 16]);
        let row: i32 = tile_to_scalar(
            metadata
                .partition(shape![1, 1])
                .load([i, 0])
                .reshape(shape![]),
        );
        let history_start: i32 = tile_to_scalar(
            metadata
                .partition(shape![1, 1])
                .load([i, 2])
                .reshape(shape![]),
        );
        let position: i32 = tile_to_scalar(
            metadata
                .partition(shape![1, 1])
                .load([i, 1])
                .reshape(shape![]),
        );
        let history_end = position + 1;
        if history_end > history_start {
            let query = q.partition(shape![1, 16]).load([i, 0]);
            let mut maximum: Tile<f32, { [1, 1] }> = constant(f32::NEG_INFINITY, shape![1, 1]);
            let mut denominator: Tile<f32, { [1, 1] }> = constant(0.0f32, shape![1, 1]);
            for position in history_start..history_end {
                let slot: i32 = tile_to_scalar(
                    table
                        .partition(shape![1, 1])
                        .load([row, position])
                        .reshape(shape![]),
                );
                let key = k.partition(shape![1, 16]).load([slot, 0]);
                let dot: Tile<f32, { [1] }> = reduce_sum(query * key, 1);
                let score = dot.reshape(shape![1, 1]) * constant(0.25f32, shape![1, 1]);
                let next_max = max_tile(maximum, score);
                let alpha = exp(maximum - next_max);
                let probability = exp(score - next_max);
                acc = acc * alpha.broadcast(shape![1, 16])
                    + v.partition(shape![1, 16]).load([slot, 0])
                        * probability.broadcast(shape![1, 16]);
                denominator = denominator * alpha + probability;
                maximum = next_max;
            }
            acc = true_div(acc, denominator.broadcast(shape![1, 16]));
        }
        out.store(acc);
    }

    #[cutile::entry()]
    pub(super) fn residual(
        out: &mut Tensor<f32, { [1, 16] }>,
        x: &Tensor<f32, { [-1, 16] }>,
        attention: &Tensor<f32, { [-1, 16] }>,
    ) {
        let i = program_id(0);
        out.store(
            x.partition(shape![1, 16]).load([i, 0])
                + attention.partition(shape![1, 16]).load([i, 0]),
        );
    }

    #[cutile::entry()]
    pub(super) fn gemm<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        y: &Tensor<f32, { [K, -1] }>,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_y = y.partition(shape![BK, BN]);
        let pid = get_tile_block_id();
        let mut tile_z: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        for i in 0i32..(K / BK) {
            tile_z = mma(part_x.load([pid.0, i]), part_y.load([i, pid.1]), tile_z);
        }
        z.store(tile_z);
    }

    #[cutile::entry()]
    pub(super) fn relu<const BM: i32, const BN: i32>(z: &mut Tensor<f32, { [BM, BN] }>) {
        let input = z.load();
        let zero = constant(0.0f32, shape![BM, BN]);
        z.store(max_tile(zero, input));
    }

    #[cutile::entry()]
    pub(super) fn matvec<const BM: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<f32, { [BM] }>,
        x: &Tensor<f32, { [-1, K] }>,
        y: &Tensor<f32, { [K] }>,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_y = y.partition(shape![BK]);
        let pid = get_tile_block_id();
        let mut tile_z: Tile<f32, { [BM, 1] }> = constant(0.0f32, shape![BM, 1]);
        for i in 0i32..(K / BK) {
            tile_z = mma(
                part_x.load([pid.0, i]),
                part_y.load([i]).reshape(shape![BK, 1]),
                tile_z,
            );
        }
        // Store through the caller's output address. On the graph path this
        // writes mapped host RAM over the interconnect, without a DtoH command.
        z.store(tile_z.reshape(shape![BM]));
    }
}
