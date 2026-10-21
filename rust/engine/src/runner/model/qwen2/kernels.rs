pub(super) use ops::*;

#[cutile::module]
pub(super) mod ops {
    use cutile::core::*;

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

    #[cutile::entry()]
    pub(super) fn rms_norm<const B: i32>(
        out: &mut Tensor<f32, { [1, B] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        w: &Tensor<f32, { [-1] }>,
        eps: f32,
        width: i32,
        offset: i32,
    ) {
        let row = program_id(0) + offset;
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
        start: i32,
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
        let position: f32 = convert_scalar(start + pid.0);
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
        row: i32,
        start: i32,
        qheads: i32,
        kvheads: i32,
    ) {
        let pid = get_tile_block_id();
        let head = pid.1 / (qheads / kvheads);
        let query = q.partition(shape![1, 64]).load([pid.0, pid.1]);
        let mut maximum: Tile<f32, { [1, 1] }> = constant(f32::NEG_INFINITY, shape![1, 1]);
        let mut denominator: Tile<f32, { [1, 1] }> = constant(0.0f32, shape![1, 1]);
        let mut acc: Tile<f32, { [1, 64] }> = constant(0.0f32, shape![1, 64]);
        for pos in 0i32..(start + pid.0 + 1) {
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
        maxima: &mut Tensor<f32, { [1] }>,
        indices: &mut Tensor<i32, { [1] }>,
        logits: &Tensor<f32, { [-1] }>,
        len: i32,
    ) {
        let pid = program_id(0);
        let offsets: Tile<i32, { [256] }> = iota(shape![256]);
        let ids = offsets + (pid * 256).broadcast(shape![256]);
        let valid = lt_tile(ids, len.broadcast(shape![256]));
        let neg: Tile<f32, { [256] }> = constant(f32::NEG_INFINITY, shape![256]);
        let values = select(valid, logits.partition(shape![256]).load([pid]), neg);
        let maximum: Tile<f32, { [1] }> = reduce_max(values, 0);
        let big: Tile<i32, { [256] }> = constant(2147483647i32, shape![256]);
        let max_value: f32 = tile_to_scalar(maximum.reshape(shape![]));
        let mask = eq_tile(values, max_value.broadcast(shape![256]));
        let candidates = select(mask, ids, big);
        let index: Tile<i32, { [1] }> = reduce_min(candidates, 0);
        maxima.store(maximum.reshape(shape![1]));
        indices.store(index.reshape(shape![1]));
    }

    #[cutile::entry()]
    pub(super) fn argmax_final(
        out: &mut Tensor<i32, { [1] }>,
        maxima: &Tensor<f32, { [-1] }>,
        indices: &Tensor<i32, { [-1] }>,
        count: i32,
    ) {
        let offsets: Tile<i32, { [1024] }> = iota(shape![1024]);
        let valid = lt_tile(offsets, count.broadcast(shape![1024]));
        let neg: Tile<f32, { [1024] }> = constant(f32::NEG_INFINITY, shape![1024]);
        let values = select(valid, maxima.partition(shape![1024]).load([0]), neg);
        let maximum: Tile<f32, { [1] }> = reduce_max(values, 0);
        let big: Tile<i32, { [1024] }> = constant(2147483647i32, shape![1024]);
        let max_value: f32 = tile_to_scalar(maximum.reshape(shape![]));
        let mask = eq_tile(values, max_value.broadcast(shape![1024]));
        let candidates = select(mask, indices.partition(shape![1024]).load([0]), big);
        let index: Tile<i32, { [1] }> = reduce_min(candidates, 0);
        out.store(index.reshape(shape![1]));
    }
}
