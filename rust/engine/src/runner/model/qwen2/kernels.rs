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
    #[cutile::entry()]
    pub(super) fn linear<const K: i32, const HAS_BIAS: bool>(
        out: &mut Tensor<f32, { [4, 32] }>,
        x: &Tensor<f32, { [-1, K] }>,
        w: &Tensor<f32, { [-1, K] }>,
        bias: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let px = x.partition(shape![4, 32]);
        let pw = w.partition(shape![32, 32]);
        let mut acc: Tile<f32, { [4, 32] }> = constant(0.0f32, shape![4, 32]);
        for i in 0i32..(K / 32) {
            acc = mma(px.load([pid.0, i]), pw.load([pid.1, i]).transpose(), acc);
        }
        if HAS_BIAS {
            let bias_tile = bias.partition(shape![32]).load([pid.1]);
            acc = acc + bias_tile.reshape(shape![1, 32]).broadcast(shape![4, 32]);
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

    /// Causal GQA reads the shared per-layer flat KV vectors through the slot table.
    #[cutile::entry()]
    pub(super) fn attention(
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
        let mut maximum: Tile<f32, { [1, 1] }> = constant(f32::NEG_INFINITY, shape![1, 1]);
        let mut denominator: Tile<f32, { [1, 1] }> = constant(0.0f32, shape![1, 1]);
        let mut acc: Tile<f32, { [1, 64] }> = constant(0.0f32, shape![1, 64]);
        for pos in history_start..(position + 1) {
            let slot: i32 = tile_to_scalar(
                table
                    .partition(shape![1, 1])
                    .load([row, pos])
                    .reshape(shape![]),
            );
            let key = k.partition(shape![1, 64]).load([slot, head]);
            let dot: Tile<f32, { [1] }> = reduce_sum(query * key, 1);
            let score = dot.reshape(shape![1, 1]) * constant(0.125f32, shape![1, 1]);
            let next = max_tile(maximum, score);
            let alpha = exp(maximum - next);
            let probability = exp(score - next);
            acc = acc * alpha.broadcast(shape![1, 64])
                + v.partition(shape![1, 64]).load([slot, head])
                    * probability.broadcast(shape![1, 64]);
            denominator = denominator * alpha + probability;
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
