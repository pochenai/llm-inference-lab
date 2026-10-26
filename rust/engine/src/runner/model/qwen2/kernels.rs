pub(super) use ops::*;

/// Shared metadata is a flat i32 tensor: three integers per packed token, followed
/// by one final packed-row index per request (not a position in its token history).
/// Token records are [context_row, absolute_position, history_start]. RoPE uses the
/// position; attention reads history_start..=absolute_position through
/// slot_table[context_row, position]. history_start is the visible KV range's lower
/// bound, not the cached-prefix length; it is currently 0 for full-history reads.
///
/// Example: A uses context row 7, has 4 cached tokens and adds [a, b]; B uses row 2,
/// has 1 cached token and adds [c]. The packed hidden rows correspond to [a, b, c].
/// ```text
/// metadata = [7, 4, 0,   // Packed row 0: A's token a, absolute position 4.
///             7, 5, 0,   // Packed row 1: A's token b, absolute position 5.
///             2, 1, 0,   // Packed row 2: B's token c, absolute position 1.
///             1, 2]     // Final packed rows of A and B, used before the LM head.
/// token i's record starts at i * TOKEN_METADATA_WIDTH.
/// last_rows_offset = 3 packed tokens * TOKEN_METADATA_WIDTH = 9.
/// metadata[9..] = [1, 2], selecting hidden[1] and hidden[2].
/// ```
/// After the last Transformer layer, hidden = [h_a, h_b, h_c]. Select [h_b, h_c]
/// for final RMSNorm and the LM head to predict A's and B's next tokens. All three
/// input tokens undergo the full Transformer computation and populate per-layer KV;
/// h_a is simply not needed by the LM head for this next-token generation step.
#[cutile::module]
pub(super) mod ops {
    use cutile::core::*;

    /// Integers per packed token: context row, absolute position, history start.
    pub(in crate::runner::model::qwen2) const TOKEN_METADATA_WIDTH: i32 = 3;

    #[cutile::entry()]
    pub(super) fn embedding(
        out: &mut Tensor<f32, { [1, 256] }>,
        ids: &Tensor<u32, { [-1] }>,
        w: &Tensor<f32, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        // Validated token IDs fit i32; reinterpret bits without a numeric conversion.
        let id: Tile<i32, { [1] }> = bitcast(ids.partition(shape![1]).load([pid.0]));
        let id: i32 = tile_to_scalar(id.reshape(shape![]));
        out.store(w.partition(shape![1, 256]).load([id, pid.1]));
    }

    /// PyTorch checkpoint weights are [out,in]; transpose each tile in registers.
    /// Optional projection bias is fused into the epilogue before the single store.
    /// BM token rows reuse weights; BN covers output channels independently of heads.
    #[cutile::entry(optimization_hints = (sm_120 = (occupancy = 2,),))]
    pub(super) fn linear<const K: i32, const HAS_BIAS: bool, const BM: i32, const BN: i32>(
        out: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        w: &Tensor<f32, { [-1, K] }>,
        bias: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let px = x.partition(shape![BM, 32]);
        let pw = w.partition(shape![BN, 32]);
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        for i in 0i32..(K / 32) {
            acc = mma(px.load([pid.0, i]), pw.load([pid.1, i]).transpose(), acc);
        }
        if HAS_BIAS {
            let bias_tile = bias.partition(shape![BN]).load([pid.1]);
            acc = acc + bias_tile.reshape(shape![1, BN]).broadcast(shape![BM, BN]);
        }
        out.store(acc);
    }

    /// Normalize every packed row, or gather each request's last row when GATHER=true.
    /// Example: A adds two tokens and B adds one, so x has three packed rows.
    /// ```text
    /// metadata = [row_A, pos_A0, history_start_A,
    ///             row_A, pos_A1, history_start_A,
    ///             row_B, pos_B0, history_start_B,
    ///             1, 2] // Last packed row of A and B.
    /// last_rows_offset = 3 tokens * TOKEN_METADATA_WIDTH = 9
    /// request 0 reads x[metadata[9 + 0]] = x[1]
    /// request 1 reads x[metadata[9 + 1]] = x[2]
    /// ```
    /// With GATHER=false, normalize x[program_id(0)] and ignore last_rows_offset.
    #[cutile::entry()]
    pub(super) fn rms_norm<const B: i32, const GATHER: bool>(
        out: &mut Tensor<f32, { [1, B] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        w: &Tensor<f32, { [-1] }>,
        metadata: &Tensor<i32, { [-1] }>,
        eps: f32,
        width: i32,
        last_rows_offset: i32,
    ) {
        let mut row = program_id(0);
        if GATHER {
            row = tile_to_scalar(
                metadata
                    .partition(shape![1])
                    .load([last_rows_offset + row])
                    .reshape(shape![]),
            );
        }
        let value = x.partition(shape![1, B]).load([row, 0]);
        let sum: Tile<f32, { [1] }> = reduce_sum(value * value, 1);
        let width_f: f32 = convert_scalar(width);
        let mean = sum.reshape(shape![1, 1]) / width_f.broadcast(shape![1, 1]);
        let inv = rsqrt(mean + eps.broadcast(shape![1, 1]), ftz::Disabled);
        out.store(
            value
                * inv.broadcast(shape![1, B])
                * w.partition(shape![B]).load([0]).reshape(shape![1, B]),
        );
    }

    /// Qwen2 uses split-half RoPE, applied after the Q/K projection bias.
    #[cutile::entry()]
    pub(super) fn rope(
        out: &mut Tensor<f32, { [1, 64] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        bias: &Tensor<f32, { [-1] }>,
        freq: &Tensor<f32, { [32] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let lo = x.partition(shape![1, 32]).load([pid.0, pid.1 * 2])
            + bias
                .partition(shape![32])
                .load([pid.1 * 2])
                .reshape(shape![1, 32]);
        let hi = x.partition(shape![1, 32]).load([pid.0, pid.1 * 2 + 1])
            + bias
                .partition(shape![32])
                .load([pid.1 * 2 + 1])
                .reshape(shape![1, 32]);
        let position: i32 = tile_to_scalar(
            metadata
                .partition(shape![1])
                .load([pid.0 * TOKEN_METADATA_WIDTH + 1])
                .reshape(shape![]),
        );
        let position: f32 = convert_scalar(position);
        let angles = freq.partition(shape![32]).load([0]).reshape(shape![1, 32])
            * position.broadcast(shape![1, 32]);
        let c = cos(angles);
        let s = sin(angles);
        let a = lo * c - hi * s;
        let b = hi * c + lo * s;
        let result: Tile<f32, { [1, 64] }> = cat(a, b, 1);
        out.store(result);
    }

    /// Gather BK physical KV slots per iteration, then update online softmax by block.
    /// Mask the final block at each query's causal boundary; slots need not be contiguous.
    /// Target four resident CTAs per SM on SM120 to reduce register pressure.
    #[cutile::entry(optimization_hints = (sm_120 = (occupancy = 4,),))]
    pub(super) unsafe fn attention<const BK: i32>(
        out: &mut Tensor<f32, { [1, 64] }>,
        q: &Tensor<f32, { [-1, -1] }>,
        k: &Tensor<f32, { [-1, -1] }>,
        v: &Tensor<f32, { [-1, -1] }>,
        table: &Tensor<i32, { [-1, -1] }>,
        metadata: &Tensor<i32, { [-1] }>,
        qheads: i32,
        kvheads: i32,
    ) {
        let pid = get_tile_block_id();
        let row: i32 = tile_to_scalar(
            metadata
                .partition(shape![1])
                .load([pid.0 * TOKEN_METADATA_WIDTH])
                .reshape(shape![]),
        );
        let position: i32 = tile_to_scalar(
            metadata
                .partition(shape![1])
                .load([pid.0 * TOKEN_METADATA_WIDTH + 1])
                .reshape(shape![]),
        );
        let history_start: i32 = tile_to_scalar(
            metadata
                .partition(shape![1])
                .load([pid.0 * TOKEN_METADATA_WIDTH + 2])
                .reshape(shape![]),
        );
        let head = pid.1 / (qheads / kvheads);
        let query = q.partition(shape![1, 64]).load([pid.0, pid.1]);
        let query: Tile<f32, { [BK, 64] }> = query.broadcast(shape![BK, 64]);
        let mut maximum: Tile<f32, { [1, 1] }> = constant(f32::NEG_INFINITY, shape![1, 1]);
        let mut denominator: Tile<f32, { [1, 1] }> = constant(0.0f32, shape![1, 1]);
        let mut acc: Tile<f32, { [1, 64] }> = constant(0.0f32, shape![1, 64]);
        let zero_values: Tile<f32, { [BK, 64] }> = constant(0.0f32, shape![BK, 64]);
        let scale: Tile<f32, { [BK] }> = constant(0.125f32, shape![BK]);
        let negative_infinity: Tile<f32, { [BK] }> = constant(f32::NEG_INFINITY, shape![BK]);
        let local: Tile<i32, { [BK] }> = iota(shape![BK]);
        let columns: Tile<i32, { [64] }> = iota(shape![64]);
        let columns: Tile<i32, { [1, 64] }> = columns.reshape(shape![1, 64]);
        let columns: Tile<i32, { [BK, 64] }> = columns.broadcast(shape![BK, 64]);
        let table_base: PointerTile<*const i32, { [] }> = pointer_to_tile(table.as_ptr());
        let table_base: PointerTile<*const i32, { [] }> = table_base.offset(row * table.shape()[1]);
        let table_base: PointerTile<*const i32, { [1] }> = table_base.reshape(shape![1]);
        let table_base: PointerTile<*const i32, { [BK] }> = table_base.broadcast(shape![BK]);
        let kp: PointerTile<*const f32, { [] }> = pointer_to_tile(k.as_ptr());
        let kp: PointerTile<*const f32, { [1, 1] }> = kp.reshape(shape![1, 1]);
        let kp: PointerTile<*const f32, { [BK, 64] }> = kp.broadcast(shape![BK, 64]);
        let vp: PointerTile<*const f32, { [] }> = pointer_to_tile(v.as_ptr());
        let vp: PointerTile<*const f32, { [1, 1] }> = vp.reshape(shape![1, 1]);
        let vp: PointerTile<*const f32, { [BK, 64] }> = vp.broadcast(shape![BK, 64]);
        for block in 0i32..((position - history_start + BK) / BK) {
            let positions = local + (history_start + block * BK).broadcast(shape![BK]);
            let valid = lt_tile(positions, (position + 1).broadcast(shape![BK]));
            // Read only mapped, visible positions; masked lanes never dereference a slot.
            let table_ptrs: PointerTile<*const i32, { [BK] }> = table_base.offset_tile(positions);
            let (slots, _token): (Tile<i32, { [BK] }>, Token) = load_ptr_tko(
                table_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(valid),
                Some(0i32),
                None,
                Latency::<1>,
            );
            let offsets =
                slots * (kvheads * 64).broadcast(shape![BK]) + (head * 64).broadcast(shape![BK]);
            let offsets: Tile<i32, { [BK, 1] }> = offsets.reshape(shape![BK, 1]);
            let offsets: Tile<i32, { [BK, 64] }> = offsets.broadcast(shape![BK, 64]);
            let offsets = offsets + columns;
            let mask: Tile<bool, { [BK, 1] }> = valid.reshape(shape![BK, 1]);
            let mask: Tile<bool, { [BK, 64] }> = mask.broadcast(shape![BK, 64]);
            let key_ptrs: PointerTile<*const f32, { [BK, 64] }> = kp.offset_tile(offsets);
            let (keys, _token): (Tile<f32, { [BK, 64] }>, Token) = load_ptr_tko(
                key_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None::<f32>,
                None,
                Latency::<1>,
            );
            // cuTile 0.4 scalar padding cannot broadcast to a 2D pointer-load result.
            let keys = select(mask, keys, zero_values);
            let dots: Tile<f32, { [BK] }> = reduce_sum(keys * query, 1);
            // Preserve the symbolic BK shape after reduction in cuTile 0.4's type checker.
            let dots: Tile<f32, { [BK] }> = dots.reshape(shape![BK]);
            let scores = select(valid, dots * scale, negative_infinity);
            let block_max: Tile<f32, { [1] }> = reduce_max(scores, 0);
            let next = max_tile(maximum, block_max.reshape(shape![1, 1]));
            let alpha = exp(maximum - next);
            let next_vector: Tile<f32, { [1] }> = next.reshape(shape![1]);
            let next_vector: Tile<f32, { [BK] }> = next_vector.broadcast(shape![BK]);
            let probabilities = exp(scores - next_vector);
            let value_ptrs: PointerTile<*const f32, { [BK, 64] }> = vp.offset_tile(offsets);
            let (values, _token): (Tile<f32, { [BK, 64] }>, Token) = load_ptr_tko(
                value_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None::<f32>,
                None,
                Latency::<1>,
            );
            let values = select(mask, values, zero_values);
            let weights: Tile<f32, { [BK, 1] }> = probabilities.reshape(shape![BK, 1]);
            let weights: Tile<f32, { [BK, 64] }> = weights.broadcast(shape![BK, 64]);
            let weighted = values * weights;
            let block_acc: Tile<f32, { [64] }> = reduce_sum(weighted, 0);
            let block_sum: Tile<f32, { [1] }> = reduce_sum(probabilities, 0);
            acc = acc * alpha.broadcast(shape![1, 64]) + block_acc.reshape(shape![1, 64]);
            denominator = denominator * alpha + block_sum.reshape(shape![1, 1]);
            maximum = next;
        }
        out.store(true_div(acc, denominator.broadcast(shape![1, 64])));
    }

    #[cutile::entry()]
    pub(super) fn add(
        out: &mut Tensor<f32, { [1, 256] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        y: &Tensor<f32, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        out.store(
            x.partition(shape![1, 256]).load([pid.0, pid.1])
                + y.partition(shape![1, 256]).load([pid.0, pid.1]),
        );
    }

    #[cutile::entry()]
    pub(super) fn swiglu(
        out: &mut Tensor<f32, { [1, 256] }>,
        gate: &Tensor<f32, { [-1, -1] }>,
        up: &Tensor<f32, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let g = gate.partition(shape![1, 256]).load([pid.0, pid.1]);
        let one: Tile<f32, { [1, 256] }> = constant(1.0f32, shape![1, 256]);
        let zero: Tile<f32, { [1, 256] }> = constant(0.0f32, shape![1, 256]);
        out.store(
            true_div(g, one + exp(zero - g)) * up.partition(shape![1, 256]).load([pid.0, pid.1]),
        );
    }

    #[cutile::entry()]
    pub(super) fn argmax_blocks(
        maxima: &mut Tensor<f32, { [1, 1] }>,
        indices: &mut Tensor<i32, { [1, 1] }>,
        logits: &Tensor<f32, { [-1, -1] }>,
        len: i32,
    ) {
        let pid = get_tile_block_id();
        let offsets: Tile<i32, { [256] }> = iota(shape![256]);
        let ids = offsets + (pid.1 * 256).broadcast(shape![256]);
        let valid = lt_tile(ids, len.broadcast(shape![256]));
        let neg: Tile<f32, { [256] }> = constant(f32::NEG_INFINITY, shape![256]);
        let values = select(
            valid,
            logits
                .partition(shape![1, 256])
                .load([pid.0, pid.1])
                .reshape(shape![256]),
            neg,
        );
        let maximum: Tile<f32, { [1] }> = reduce_max(values, 0);
        let big: Tile<i32, { [256] }> = constant(2147483647i32, shape![256]);
        let max_value: f32 = tile_to_scalar(maximum.reshape(shape![]));
        let mask = eq_tile(values, max_value.broadcast(shape![256]));
        let candidates = select(mask, ids, big);
        let index: Tile<i32, { [1] }> = reduce_min(candidates, 0);
        maxima.store(maximum.reshape(shape![1, 1]));
        indices.store(index.reshape(shape![1, 1]));
    }

    #[cutile::entry()]
    pub(super) fn argmax_final(
        out: &mut Tensor<i32, { [1] }>,
        maxima: &Tensor<f32, { [-1, -1] }>,
        indices: &Tensor<i32, { [-1, -1] }>,
        count: i32,
    ) {
        let offsets: Tile<i32, { [1024] }> = iota(shape![1024]);
        let valid = lt_tile(offsets, count.broadcast(shape![1024]));
        let neg: Tile<f32, { [1024] }> = constant(f32::NEG_INFINITY, shape![1024]);
        let values = select(
            valid,
            maxima
                .partition(shape![1, 1024])
                .load([program_id(0), 0])
                .reshape(shape![1024]),
            neg,
        );
        let maximum: Tile<f32, { [1] }> = reduce_max(values, 0);
        let big: Tile<i32, { [1024] }> = constant(2147483647i32, shape![1024]);
        let max_value: f32 = tile_to_scalar(maximum.reshape(shape![]));
        let mask = eq_tile(values, max_value.broadcast(shape![1024]));
        let candidates = select(
            mask,
            indices
                .partition(shape![1, 1024])
                .load([program_id(0), 0])
                .reshape(shape![1024]),
            big,
        );
        let index: Tile<i32, { [1] }> = reduce_min(candidates, 0);
        out.store(index.reshape(shape![1]));
    }
}
