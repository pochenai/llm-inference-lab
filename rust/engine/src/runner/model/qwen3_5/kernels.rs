pub(super) use ops::*;

#[cutile::module]
mod ops {
    use cutile::core::*;

    pub(in crate::runner::model::qwen3_5) const CONV_KERNEL_SIZE: i32 = 4;
    pub(in crate::runner::model::qwen3_5) const CONV_HISTORY_LEN: i32 = CONV_KERNEL_SIZE - 1;
    const CONV_STATE_TILE_ROWS: i32 = 4;
    pub(in crate::runner::model::qwen3_5) const POINTWISE_COLUMNS: i32 = 128;
    pub(in crate::runner::model::qwen3_5) const DELTA_VALUE_COLUMNS: i32 = 32;
    pub(in crate::runner::model::qwen3_5) const EXPERT_OUTPUT_COLUMNS: i32 = 32;
    const EXPERT_REDUCTION_COLUMNS: i32 = 64;

    unsafe fn store_values<T: ElementType, const M: i32, const N: i32>(
        out: *mut T,
        value: Tile<T, { [M, N] }>,
        row: i32,
        column: i32,
        width: i32,
    ) {
        let rows: Tile<i32, { [M] }> = iota(shape![M]);
        let columns: Tile<i32, { [N] }> = iota(shape![N]);
        let offsets = ((rows + row.broadcast(shape![M])) * width.broadcast(shape![M]))
            .reshape(shape![M, 1])
            .broadcast(shape![M, N])
            + (columns + column.broadcast(shape![N]))
                .reshape(shape![1, N])
                .broadcast(shape![M, N]);
        let base = pointer_to_tile(out);
        let base = base.reshape(shape![1, 1]);
        let base = base.broadcast(shape![M, N]);
        let destination = base.offset_tile(offsets);
        store_ptr_tko(
            destination,
            value,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            Latency::<1>,
        );
    }

    unsafe fn store_f32<const M: i32, const N: i32>(
        out: *mut f32,
        value: Tile<f32, { [M, N] }>,
        row: i32,
        column: i32,
        width: i32,
    ) {
        let rows: Tile<i32, { [M] }> = iota(shape![M]);
        let columns: Tile<i32, { [N] }> = iota(shape![N]);
        let offsets = ((rows + row.broadcast(shape![M])) * width.broadcast(shape![M]))
            .reshape(shape![M, 1])
            .broadcast(shape![M, N])
            + (columns + column.broadcast(shape![N]))
                .reshape(shape![1, N])
                .broadcast(shape![M, N]);
        let base = pointer_to_tile(out);
        let base = base.reshape(shape![1, 1]);
        let base = base.broadcast(shape![M, N]);
        let destination = base.offset_tile(offsets);
        store_ptr_tko(
            destination,
            value,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            Latency::<1>,
        );
    }

    #[cutile::entry()]
    pub(super) fn last_row(
        out: &mut Tensor<bf16, { [1, POINTWISE_COLUMNS] }>,
        x: &Tensor<bf16, { [-1, -1] }>,
    ) {
        out.store(
            x.partition(shape![1, POINTWISE_COLUMNS])
                .load([x.shape()[0] - 1, program_id(1)]),
        );
    }

    #[cutile::entry()]
    pub(super) fn norm<const D: i32, const ZERO_CENTERED: bool>(
        out: &mut Tensor<bf16, { [1, D] }>,
        x: &Tensor<bf16, { [-1, D] }>,
        w: &Tensor<f32, { [D] }>,
        eps: f32,
    ) {
        let one: Tile<f32, { [1, D] }> = constant(1.0f32, shape![1, D]);
        let width: f32 = convert_scalar(D);
        let x: Tile<f32, { [1, D] }> =
            convert_tile(x.partition(shape![1, D]).load([program_id(0), 0]));
        let sum: Tile<f32, { [1] }> = reduce_sum(x * x, 1);
        let mean = sum.reshape(shape![1, 1]) / width.broadcast(shape![1, 1]);
        let mut weight = w.partition(shape![D]).load([0]).reshape(shape![1, D]);

        if ZERO_CENTERED {
            weight = weight + one;
        }
        let result: Tile<bf16, { [1, D] }> = convert_tile(
            x * rsqrt(mean + eps.broadcast(shape![1, 1]), ftz::Disabled).broadcast(shape![1, D])
                * weight,
        );
        out.store(result);
    }

    /// Keep the previous three raw projections; convolution precedes SiLU.
    #[cutile::entry()]
    pub(super) unsafe fn conv<T: ElementType, const C: i32>(
        out: *mut T,
        state: *mut T,
        x: &Tensor<T, { [-1, C] }>,
        weight: &Tensor<T, { [CONV_KERNEL_SIZE, C] }>,
    ) {
        let one: Tile<f32, { [1, POINTWISE_COLUMNS] }> =
            constant(1.0f32, shape![1, POINTWISE_COLUMNS]);
        let zero: Tile<f32, { [1, POINTWISE_COLUMNS] }> =
            constant(0.0f32, shape![1, POINTWISE_COLUMNS]);
        let block = program_id(1);
        let channels: Tile<i32, { [POINTWISE_COLUMNS] }> = iota(shape![POINTWISE_COLUMNS]);
        let base = pointer_to_tile(state);
        let base = base.reshape(shape![1]);
        let base = base.broadcast(shape![POINTWISE_COLUMNS]);
        let offsets = channels + (block * POINTWISE_COLUMNS).broadcast(shape![POINTWISE_COLUMNS]);
        let source = base.offset_tile(offsets);
        let (s0, _t): (Tile<T, { [POINTWISE_COLUMNS] }>, Token) = load_ptr_tko(
            source,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<1>,
        );
        let source = base.offset_tile(offsets + C.broadcast(shape![POINTWISE_COLUMNS]));
        let (s1, _t): (Tile<T, { [POINTWISE_COLUMNS] }>, Token) = load_ptr_tko(
            source,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<1>,
        );
        let source = base.offset_tile(offsets + (2 * C).broadcast(shape![POINTWISE_COLUMNS]));
        let (s2, _t): (Tile<T, { [POINTWISE_COLUMNS] }>, Token) = load_ptr_tko(
            source,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<1>,
        );
        let previous1 = s1.reshape(shape![1, POINTWISE_COLUMNS]);
        let previous2 = s2.reshape(shape![1, POINTWISE_COLUMNS]);
        let mut s0 = s0.reshape(shape![1, POINTWISE_COLUMNS]);
        let mut s1 = previous1;
        let mut s2 = previous2;
        let w0: Tile<f32, { [1, POINTWISE_COLUMNS] }> = convert_tile(
            weight
                .partition(shape![1, POINTWISE_COLUMNS])
                .load([0, block]),
        );
        let w1: Tile<f32, { [1, POINTWISE_COLUMNS] }> = convert_tile(
            weight
                .partition(shape![1, POINTWISE_COLUMNS])
                .load([1, block]),
        );
        let w2: Tile<f32, { [1, POINTWISE_COLUMNS] }> = convert_tile(
            weight
                .partition(shape![1, POINTWISE_COLUMNS])
                .load([2, block]),
        );
        let w3: Tile<f32, { [1, POINTWISE_COLUMNS] }> = convert_tile(
            weight
                .partition(shape![1, POINTWISE_COLUMNS])
                .load([CONV_HISTORY_LEN, block]),
        );
        // Build the raw suffix before shifting loop carries: cuTile 0.4 otherwise
        // saves the final carry into all three history planes.
        let count = x.shape()[0];
        let next0 = if count >= CONV_HISTORY_LEN {
            x.partition(shape![1, POINTWISE_COLUMNS])
                .load([count - CONV_HISTORY_LEN, block])
        } else if count == 2 {
            previous2
        } else {
            previous1
        };
        let next1 = if count >= 2 {
            x.partition(shape![1, POINTWISE_COLUMNS])
                .load([count - 2, block])
        } else {
            previous2
        };
        let next2 = x
            .partition(shape![1, POINTWISE_COLUMNS])
            .load([count - 1, block]);
        let planes: Tile<i32, { [CONV_STATE_TILE_ROWS] }> = iota(shape![CONV_STATE_TILE_ROWS]);
        let first = eq_tile(planes, 0i32.broadcast(shape![CONV_STATE_TILE_ROWS]))
            .reshape(shape![CONV_STATE_TILE_ROWS, 1])
            .broadcast(shape![CONV_STATE_TILE_ROWS, POINTWISE_COLUMNS]);
        let second = eq_tile(planes, 1i32.broadcast(shape![CONV_STATE_TILE_ROWS]))
            .reshape(shape![CONV_STATE_TILE_ROWS, 1])
            .broadcast(shape![CONV_STATE_TILE_ROWS, POINTWISE_COLUMNS]);
        let values = select(
            first,
            next0.broadcast(shape![CONV_STATE_TILE_ROWS, POINTWISE_COLUMNS]),
            select(
                second,
                next1.broadcast(shape![CONV_STATE_TILE_ROWS, POINTWISE_COLUMNS]),
                next2.broadcast(shape![CONV_STATE_TILE_ROWS, POINTWISE_COLUMNS]),
            ),
        );
        let offsets = (planes * C.broadcast(shape![CONV_STATE_TILE_ROWS]))
            .reshape(shape![CONV_STATE_TILE_ROWS, 1])
            .broadcast(shape![CONV_STATE_TILE_ROWS, POINTWISE_COLUMNS])
            + (channels + (block * POINTWISE_COLUMNS).broadcast(shape![POINTWISE_COLUMNS]))
                .reshape(shape![1, POINTWISE_COLUMNS])
                .broadcast(shape![CONV_STATE_TILE_ROWS, POINTWISE_COLUMNS]);
        let valid = lt_tile(
            planes,
            CONV_HISTORY_LEN.broadcast(shape![CONV_STATE_TILE_ROWS]),
        )
        .reshape(shape![CONV_STATE_TILE_ROWS, 1])
        .broadcast(shape![CONV_STATE_TILE_ROWS, POINTWISE_COLUMNS]);
        let destination = pointer_to_tile(state);
        let destination = destination.reshape(shape![1, 1]);
        let destination = destination.broadcast(shape![CONV_STATE_TILE_ROWS, POINTWISE_COLUMNS]);
        let destination = destination.offset_tile(offsets);
        for row in 0..x.shape()[0] {
            let current = x.partition(shape![1, POINTWISE_COLUMNS]).load([row, block]);
            let a: Tile<f32, { [1, POINTWISE_COLUMNS] }> = convert_tile(s0);
            let b: Tile<f32, { [1, POINTWISE_COLUMNS] }> = convert_tile(s1);
            let c: Tile<f32, { [1, POINTWISE_COLUMNS] }> = convert_tile(s2);
            let d: Tile<f32, { [1, POINTWISE_COLUMNS] }> = convert_tile(current);
            let y = a * w0 + b * w1 + c * w2 + d * w3;

            let result: Tile<T, { [1, POINTWISE_COLUMNS] }> =
                convert_tile(y / (one + exp(zero - y)));
            store_values::<T, 1, POINTWISE_COLUMNS>(out, result, row, block * POINTWISE_COLUMNS, C);
            s0 = s1;
            s1 = s2;
            s2 = current;
        }
        store_ptr_tko(
            destination,
            values,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(valid),
            None,
            Latency::<1>,
        );
    }

    /// One head/value tile owns its FP32 recurrent state across the token loop.
    /// shortcut: sequential recurrence for prefill, add chunk GDN after the correctness baseline.
    #[cutile::entry()]
    pub(super) unsafe fn delta<
        T: ElementType,
        const KH: i32,
        const VH: i32,
        const K: i32,
        const V: i32,
    >(
        out: *mut T,
        state: *mut f32,
        qkv: &Tensor<T, { [-1, -1] }>,
        a: &Tensor<T, { [-1, VH] }>,
        b: &Tensor<T, { [-1, VH] }>,
        a_log: &Tensor<f32, { [VH] }>,
        dt: &Tensor<f32, { [VH] }>,
    ) {
        let epsilon: Tile<f32, { [1] }> = constant(1e-6f32, shape![1]);
        let zero: Tile<f32, { [1] }> = constant(0.0f32, shape![1]);
        let one: Tile<f32, { [1] }> = constant(1.0f32, shape![1]);
        let width: f32 = convert_scalar(K);
        let head = program_id(0);
        let column = program_id(1);
        let key_head = head / (VH / KH);
        let keys: Tile<i32, { [K] }> = iota(shape![K]);
        let values: Tile<i32, { [DELTA_VALUE_COLUMNS] }> = iota(shape![DELTA_VALUE_COLUMNS]);
        let offsets = (keys * V.broadcast(shape![K]) + (head * K * V).broadcast(shape![K]))
            .reshape(shape![K, 1])
            .broadcast(shape![K, DELTA_VALUE_COLUMNS])
            + (values + (column * DELTA_VALUE_COLUMNS).broadcast(shape![DELTA_VALUE_COLUMNS]))
                .reshape(shape![1, DELTA_VALUE_COLUMNS])
                .broadcast(shape![K, DELTA_VALUE_COLUMNS]);
        let base = pointer_to_tile(state);
        let base = base.reshape(shape![1, 1]);
        let base = base.broadcast(shape![K, DELTA_VALUE_COLUMNS]);
        let source = base.offset_tile(offsets);
        let (mut s, _t): (Tile<f32, { [K, DELTA_VALUE_COLUMNS] }>, Token) = load_ptr_tko(
            source,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<1>,
        );
        let al = a_log.partition(shape![1]).load([head]).reshape(shape![1]);
        let bias = dt.partition(shape![1]).load([head]).reshape(shape![1]);
        for row in 0..qkv.shape()[0] {
            let q: Tile<f32, { [K] }> = convert_tile(
                qkv.partition(shape![1, K])
                    .load([row, key_head])
                    .reshape(shape![K]),
            );
            let k: Tile<f32, { [K] }> = convert_tile(
                qkv.partition(shape![1, K])
                    .load([row, KH + key_head])
                    .reshape(shape![K]),
            );
            let qsum: Tile<f32, { [1] }> = reduce_sum(q * q, 0);
            let qsum = qsum.reshape(shape![1]);
            let ksum: Tile<f32, { [1] }> = reduce_sum(k * k, 0);
            let ksum = ksum.reshape(shape![1]);

            let q = q * rsqrt(qsum + epsilon, ftz::Disabled).broadcast(shape![K])
                / sqrt(
                    width.broadcast(shape![K]),
                    rounding::NearestEven,
                    ftz::Disabled,
                );

            let k = k * rsqrt(ksum + epsilon, ftz::Disabled).broadcast(shape![K]);
            let av: Tile<f32, { [1] }> = convert_tile(
                a.partition(shape![1, 1])
                    .load([row, head])
                    .reshape(shape![1]),
            );
            let bv: Tile<f32, { [1] }> = convert_tile(
                b.partition(shape![1, 1])
                    .load([row, head])
                    .reshape(shape![1]),
            );
            let av = av + bias;

            let softplus = max_tile(av, zero) + log(one + exp(zero - absf(av)));

            let decay = exp(zero - exp(al) * softplus);

            let beta_bf: Tile<T, { [1] }> = convert_tile(one / (one + exp(zero - bv)));
            let beta: Tile<f32, { [1] }> = convert_tile(beta_bf);
            s = s * decay
                .reshape(shape![1, 1])
                .broadcast(shape![K, DELTA_VALUE_COLUMNS]);
            let memory: Tile<f32, { [DELTA_VALUE_COLUMNS] }> = reduce_sum(
                s * k
                    .reshape(shape![K, 1])
                    .broadcast(shape![K, DELTA_VALUE_COLUMNS]),
                0,
            );
            let v: Tile<f32, { [DELTA_VALUE_COLUMNS] }> = convert_tile(
                qkv.partition(shape![1, DELTA_VALUE_COLUMNS])
                    .load([
                        row,
                        2 * KH * K / DELTA_VALUE_COLUMNS + head * V / DELTA_VALUE_COLUMNS + column,
                    ])
                    .reshape(shape![DELTA_VALUE_COLUMNS]),
            );
            let update = (v - memory) * beta.broadcast(shape![DELTA_VALUE_COLUMNS]);
            s = s + k
                .reshape(shape![K, 1])
                .broadcast(shape![K, DELTA_VALUE_COLUMNS])
                * update
                    .reshape(shape![1, DELTA_VALUE_COLUMNS])
                    .broadcast(shape![K, DELTA_VALUE_COLUMNS]);
            let y: Tile<f32, { [DELTA_VALUE_COLUMNS] }> = reduce_sum(
                s * q
                    .reshape(shape![K, 1])
                    .broadcast(shape![K, DELTA_VALUE_COLUMNS]),
                0,
            );
            let result: Tile<T, { [1, DELTA_VALUE_COLUMNS] }> =
                convert_tile(y.reshape(shape![1, DELTA_VALUE_COLUMNS]));
            store_values::<T, 1, DELTA_VALUE_COLUMNS>(
                out,
                result,
                row,
                head * V + column * DELTA_VALUE_COLUMNS,
                VH * V,
            );
        }
        store_f32::<K, DELTA_VALUE_COLUMNS>(state, s, head * K, column * DELTA_VALUE_COLUMNS, V);
    }

    #[cutile::entry()]
    pub(super) fn gated_norm<const D: i32>(
        out: &mut Tensor<bf16, { [1, D] }>,
        x: &Tensor<bf16, { [-1, D] }>,
        z: &Tensor<bf16, { [-1, D] }>,
        w: &Tensor<f32, { [D] }>,
        eps: f32,
    ) {
        let one: Tile<f32, { [1, D] }> = constant(1.0f32, shape![1, D]);
        let zero: Tile<f32, { [1, D] }> = constant(0.0f32, shape![1, D]);
        let width: f32 = convert_scalar(D);
        let row = program_id(0);
        let x: Tile<f32, { [1, D] }> = convert_tile(x.partition(shape![1, D]).load([row, 0]));
        let sum: Tile<f32, { [1] }> = reduce_sum(x * x, 1);
        let mean = sum.reshape(shape![1, 1]) / width.broadcast(shape![1, 1]);
        let n: Tile<bf16, { [1, D] }> = convert_tile(
            x * rsqrt(mean + eps.broadcast(shape![1, 1]), ftz::Disabled).broadcast(shape![1, D]),
        );
        let n: Tile<f32, { [1, D] }> = convert_tile(n);
        let z: Tile<f32, { [1, D] }> = convert_tile(z.partition(shape![1, D]).load([row, 0]));

        let result: Tile<bf16, { [1, D] }> = convert_tile(
            n * w.partition(shape![D]).load([0]).reshape(shape![1, D]) * z / (one + exp(zero - z)),
        );
        out.store(result);
    }

    /// Text positions share all three mRoPE axes, reducing to partial split-half RoPE.
    #[cutile::entry()]
    pub(super) unsafe fn qk<T: ElementType, const D: i32, const R: i32, const GATED: bool>(
        out: *mut T,
        x: &Tensor<T, { [-1, -1] }>,
        w: &Tensor<f32, { [D] }>,
        frequencies: &Tensor<f32, { [R] }>,
        eps: f32,
        position: i32,
        heads: i32,
        destination: i32,
    ) {
        let one: Tile<f32, { [D] }> = constant(1.0f32, shape![D]);
        let negative_one: Tile<f32, { [D] }> = constant(-1.0f32, shape![D]);
        let width: f32 = convert_scalar(D);
        let row = program_id(0);
        let head = program_id(1);
        let stride = if GATED { 2 } else { 1 };
        let raw: Tile<f32, { [D] }> = convert_tile(
            x.partition(shape![1, D])
                .load([row, head * stride])
                .reshape(shape![D]),
        );
        let sum: Tile<f32, { [1] }> = reduce_sum(raw * raw, 0);
        let sum = sum.reshape(shape![1]);
        let mean = sum / width.broadcast(shape![1]);
        let inv = rsqrt(mean + eps.broadcast(shape![1]), ftz::Disabled).broadcast(shape![D]);

        let n: Tile<T, { [D] }> =
            convert_tile(raw * inv * (w.partition(shape![D]).load([0]) + one));
        let n: Tile<f32, { [D] }> = convert_tile(n);
        let ids: Tile<i32, { [D] }> = iota(shape![D]);
        let shifted = select(
            lt_tile(ids, (R / 2).broadcast(shape![D])),
            ids + (R / 2).broadcast(shape![D]),
            ids - (R / 2).broadcast(shape![D]),
        );
        let valid = lt_tile(ids, R.broadcast(shape![D]));
        let base = pointer_to_tile(x.as_ptr());
        let base = base.reshape(shape![1]);
        let base = base.broadcast(shape![D]);
        let offsets = shifted + (row * x.shape()[1] + head * stride * D).broadcast(shape![D]);
        let source = base.offset_tile(offsets);
        let (other, _t): (Tile<T, { [D] }>, Token) = load_ptr_tko(
            source,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<1>,
        );
        let wb = pointer_to_tile(w.as_ptr());
        let wb = wb.reshape(shape![1]);
        let wb = wb.broadcast(shape![D]);
        let source = wb.offset_tile(shifted);
        let (other_w, _t): (Tile<f32, { [D] }>, Token) = load_ptr_tko(
            source,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<1>,
        );
        let other: Tile<f32, { [D] }> = convert_tile(other);

        let other: Tile<T, { [D] }> = convert_tile(other * inv * (other_w + one));
        let other: Tile<f32, { [D] }> = convert_tile(other);
        let freq_ids = ids % (R / 2).broadcast(shape![D]);
        let fb = pointer_to_tile(frequencies.as_ptr());
        let fb = fb.reshape(shape![1]);
        let fb = fb.broadcast(shape![D]);
        let source = fb.offset_tile(freq_ids);
        let (freq, _t): (Tile<f32, { [D] }>, Token) = load_ptr_tko(
            source,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<1>,
        );
        let absolute_position: f32 = convert_scalar(position + row);
        let angle = freq * absolute_position.broadcast(shape![D]);
        // Match BF16 RoPE operands/products, including the final rounded sum.
        let c: Tile<T, { [D] }> = convert_tile(cos(angle));
        let sn: Tile<T, { [D] }> = convert_tile(sin(angle));
        let c: Tile<f32, { [D] }> = convert_tile(c);
        let sn: Tile<f32, { [D] }> = convert_tile(sn);

        let sign = select(
            lt_tile(ids, (R / 2).broadcast(shape![D])),
            negative_one,
            one,
        );
        let left: Tile<T, { [D] }> = convert_tile(n * c);
        let right: Tile<T, { [D] }> = convert_tile(other * sign * sn);
        let left: Tile<f32, { [D] }> = convert_tile(left);
        let right: Tile<f32, { [D] }> = convert_tile(right);
        let result: Tile<T, { [1, D] }> =
            convert_tile(select(valid, left + right, n).reshape(shape![1, D]));
        store_values::<T, 1, D>(out, result, destination + row, head * D, heads * D);
    }

    #[cutile::entry()]
    pub(super) unsafe fn cache_v<T: ElementType, const D: i32>(
        out: *mut T,
        x: &Tensor<T, { [-1, -1] }>,
        start: i32,
    ) {
        let p = get_tile_block_id();
        let value = x.partition(shape![1, D]).load([p.0, p.1]);
        store_values::<T, 1, D>(out, value, start + p.0, p.1 * D, x.shape()[1]);
    }

    #[cutile::entry()]
    pub(super) fn gqa_attention<const D: i32, const QH: i32, const KVH: i32>(
        out: &mut Tensor<bf16, { [1, D] }>,
        q: &Tensor<bf16, { [-1, -1] }>,
        k: &Tensor<bf16, { [-1, -1] }>,
        v: &Tensor<bf16, { [-1, -1] }>,
        q_gate: &Tensor<bf16, { [-1, -1] }>,
        start: i32,
    ) {
        let zero_vector: Tile<f32, { [D] }> = constant(0.0f32, shape![D]);
        let one: Tile<f32, { [D] }> = constant(1.0f32, shape![D]);
        let width: f32 = convert_scalar(D);
        let row = program_id(0);
        let h = program_id(1);
        let kh = h / (QH / KVH);
        let q: Tile<f32, { [D] }> =
            convert_tile(q.partition(shape![1, D]).load([row, h]).reshape(shape![D]));

        let mut m: Tile<f32, { [1] }> = constant(f32::NEG_INFINITY, shape![1]);

        let mut denom: Tile<f32, { [1] }> = constant(0.0f32, shape![1]);

        let mut acc: Tile<f32, { [D] }> = constant(0.0f32, shape![D]);
        for t in 0..(start + row + 1) {
            let key: Tile<f32, { [D] }> =
                convert_tile(k.partition(shape![1, D]).load([t, kh]).reshape(shape![D]));
            let value: Tile<f32, { [D] }> =
                convert_tile(v.partition(shape![1, D]).load([t, kh]).reshape(shape![D]));
            let dot: Tile<f32, { [1] }> = reduce_sum(q * key, 0);
            let dot = dot.reshape(shape![1]);
            let score = dot
                / sqrt(
                    width.broadcast(shape![1]),
                    rounding::NearestEven,
                    ftz::Disabled,
                );
            let next_m = max_tile(m, score);
            let alpha = exp(m - next_m);
            let p = exp(score - next_m);
            acc = acc * alpha.broadcast(shape![D]) + value * p.broadcast(shape![D]);
            denom = denom * alpha + p;
            m = next_m;
        }
        let y: Tile<bf16, { [D] }> = convert_tile(acc / denom.broadcast(shape![D]));
        let y: Tile<f32, { [D] }> = convert_tile(y);
        let gate: Tile<f32, { [D] }> = convert_tile(
            q_gate
                .partition(shape![1, D])
                .load([row, h * 2 + 1])
                .reshape(shape![D]),
        );

        let sigmoid: Tile<bf16, { [D] }> = convert_tile(one / (one + exp(zero_vector - gate)));
        let sigmoid: Tile<f32, { [D] }> = convert_tile(sigmoid);
        let result: Tile<bf16, { [1, D] }> = convert_tile((y * sigmoid).reshape(shape![1, D]));
        out.store(result);
    }

    #[cutile::entry()]
    pub(super) fn route<T: ElementType, const EXPERTS: i32, const TOP: i32>(
        ids: &mut Tensor<i32, { [1, TOP] }>,
        scores: &mut Tensor<T, { [1, TOP] }>,
        logits: &Tensor<T, { [-1, EXPERTS] }>,
    ) {
        let row = program_id(0);
        let original: Tile<f32, { [EXPERTS] }> = convert_tile(
            logits
                .partition(shape![1, EXPERTS])
                .load([row, 0])
                .reshape(shape![EXPERTS]),
        );
        let indices: Tile<i32, { [EXPERTS] }> = iota(shape![EXPERTS]);
        let ranks: Tile<i32, { [TOP] }> = iota(shape![TOP]);
        let negative_infinity: Tile<f32, { [EXPERTS] }> =
            constant(f32::NEG_INFINITY, shape![EXPERTS]);
        let max_index: Tile<i32, { [EXPERTS] }> = constant(2147483647i32, shape![EXPERTS]);
        let mut selected: Tile<i32, { [TOP] }> = constant(0i32, shape![TOP]);
        let mut values: Tile<f32, { [TOP] }> = constant(0.0f32, shape![TOP]);
        let maximum: Tile<f32, { [1] }> = reduce_max(original, 0);
        let maximum = maximum.reshape(shape![1]);
        let mut remaining = original;
        for rank in 0..TOP {
            let best: Tile<f32, { [1] }> = reduce_max(remaining, 0);
            let best = best.reshape(shape![1]);
            let candidates = select(
                eq_tile(remaining, best.broadcast(shape![EXPERTS])),
                indices,
                max_index,
            );
            let id: Tile<i32, { [1] }> = reduce_min(candidates, 0);
            let id = id.reshape(shape![1]);
            let lane = eq_tile(ranks, rank.broadcast(shape![TOP]));
            selected = select(lane, id.broadcast(shape![TOP]), selected);
            values = select(lane, exp(best - maximum).broadcast(shape![TOP]), values);
            remaining = select(
                eq_tile(indices, id.broadcast(shape![EXPERTS])),
                negative_infinity,
                remaining,
            );
        }
        let denominator: Tile<f32, { [1] }> = reduce_sum(values, 0);
        let denominator = denominator.reshape(shape![1]);
        let result: Tile<T, { [1, TOP] }> =
            convert_tile((values / denominator.broadcast(shape![TOP])).reshape(shape![1, TOP]));
        ids.store(selected.reshape(shape![1, TOP]));
        scores.store(result);
    }

    /// shortcut: direct selected-expert reductions, replace with grouped MMA when tuning MoE.
    #[cutile::entry()]
    pub(super) unsafe fn expert<T: ElementType, const K: i32, const N: i32, const TOP: i32>(
        out: &mut Tensor<T, { [1, EXPERT_OUTPUT_COLUMNS] }>,
        x: &Tensor<T, { [-1, K] }>,
        weight: &Tensor<T, { [-1, K] }>,
        ids: &Tensor<i32, { [-1, TOP] }>,
        input_repeat: i32,
    ) {
        let pid = get_tile_block_id();
        let base = pointer_to_tile(ids.as_ptr());
        let source = base.offset(pid.0);
        let source = source.reshape(shape![1]);
        let (id, _t): (Tile<i32, { [1] }>, Token) = load_ptr_tko(
            source,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<1>,
        );
        let id: i32 = tile_to_scalar(id.reshape(shape![]));
        let row = pid.0 / input_repeat;
        let columns: Tile<i32, { [EXPERT_OUTPUT_COLUMNS] }> = iota(shape![EXPERT_OUTPUT_COLUMNS]);
        let channels: Tile<i32, { [EXPERT_REDUCTION_COLUMNS] }> =
            iota(shape![EXPERT_REDUCTION_COLUMNS]);
        let xbase = pointer_to_tile(x.as_ptr());
        let xbase = xbase.reshape(shape![1]);
        let xbase = xbase.broadcast(shape![EXPERT_REDUCTION_COLUMNS]);
        let wbase = pointer_to_tile(weight.as_ptr());
        let wbase = wbase.reshape(shape![1, 1]);
        let wbase = wbase.broadcast(shape![EXPERT_OUTPUT_COLUMNS, EXPERT_REDUCTION_COLUMNS]);
        let mut acc: Tile<f32, { [EXPERT_OUTPUT_COLUMNS] }> =
            constant(0.0f32, shape![EXPERT_OUTPUT_COLUMNS]);
        for block in 0..(K / EXPERT_REDUCTION_COLUMNS) {
            let xoffset = channels
                + (row * K + block * EXPERT_REDUCTION_COLUMNS)
                    .broadcast(shape![EXPERT_REDUCTION_COLUMNS]);
            let source = xbase.offset_tile(xoffset);
            let (input, _t): (Tile<T, { [EXPERT_REDUCTION_COLUMNS] }>, Token) = load_ptr_tko(
                source,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<1>,
            );
            let input: Tile<f32, { [EXPERT_REDUCTION_COLUMNS] }> = convert_tile(input);
            let offsets = ((columns
                + (id * N + pid.1 * EXPERT_OUTPUT_COLUMNS)
                    .broadcast(shape![EXPERT_OUTPUT_COLUMNS]))
                * K.broadcast(shape![EXPERT_OUTPUT_COLUMNS]))
            .reshape(shape![EXPERT_OUTPUT_COLUMNS, 1])
            .broadcast(shape![EXPERT_OUTPUT_COLUMNS, EXPERT_REDUCTION_COLUMNS])
                + (channels
                    + (block * EXPERT_REDUCTION_COLUMNS)
                        .broadcast(shape![EXPERT_REDUCTION_COLUMNS]))
                .reshape(shape![1, EXPERT_REDUCTION_COLUMNS])
                .broadcast(shape![EXPERT_OUTPUT_COLUMNS, EXPERT_REDUCTION_COLUMNS]);
            let source = wbase.offset_tile(offsets);
            let (w, _t): (
                Tile<T, { [EXPERT_OUTPUT_COLUMNS, EXPERT_REDUCTION_COLUMNS] }>,
                Token,
            ) = load_ptr_tko(
                source,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<1>,
            );
            let w: Tile<f32, { [EXPERT_OUTPUT_COLUMNS, EXPERT_REDUCTION_COLUMNS] }> =
                convert_tile(w);
            let product = w * input
                .reshape(shape![1, EXPERT_REDUCTION_COLUMNS])
                .broadcast(shape![EXPERT_OUTPUT_COLUMNS, EXPERT_REDUCTION_COLUMNS]);
            let partial: Tile<f32, { [EXPERT_OUTPUT_COLUMNS] }> = reduce_sum(product, 1);
            acc = acc + partial;
        }
        let result: Tile<T, { [1, EXPERT_OUTPUT_COLUMNS] }> =
            convert_tile(acc.reshape(shape![1, EXPERT_OUTPUT_COLUMNS]));
        out.store(result);
    }

    #[cutile::entry()]
    pub(super) fn expert_swiglu<const I: i32>(
        out: &mut Tensor<bf16, { [1, POINTWISE_COLUMNS] }>,
        x: &Tensor<bf16, { [-1, -1] }>,
    ) {
        let one: Tile<f32, { [1, POINTWISE_COLUMNS] }> =
            constant(1.0f32, shape![1, POINTWISE_COLUMNS]);
        let zero: Tile<f32, { [1, POINTWISE_COLUMNS] }> =
            constant(0.0f32, shape![1, POINTWISE_COLUMNS]);
        let p = get_tile_block_id();
        let g: Tile<f32, { [1, POINTWISE_COLUMNS] }> =
            convert_tile(x.partition(shape![1, POINTWISE_COLUMNS]).load([p.0, p.1]));
        let u: Tile<f32, { [1, POINTWISE_COLUMNS] }> = convert_tile(
            x.partition(shape![1, POINTWISE_COLUMNS])
                .load([p.0, I / POINTWISE_COLUMNS + p.1]),
        );

        let activated: Tile<bf16, { [1, POINTWISE_COLUMNS] }> =
            convert_tile(g / (one + exp(zero - g)));
        let activated: Tile<f32, { [1, POINTWISE_COLUMNS] }> = convert_tile(activated);
        let result: Tile<bf16, { [1, POINTWISE_COLUMNS] }> = convert_tile(activated * u);
        out.store(result);
    }

    #[cutile::entry()]
    pub(super) fn moe_merge<const TOP: i32>(
        out: &mut Tensor<bf16, { [1, POINTWISE_COLUMNS] }>,
        routed: &Tensor<bf16, { [-1, -1] }>,
        scores: &Tensor<bf16, { [-1, TOP] }>,
        shared: &Tensor<bf16, { [-1, -1] }>,
        gate: &Tensor<bf16, { [-1, 1] }>,
    ) {
        let one: Tile<f32, { [1, 1] }> = constant(1.0f32, shape![1, 1]);
        let zero_vector: Tile<f32, { [1, 1] }> = constant(0.0f32, shape![1, 1]);
        let p = get_tile_block_id();

        let mut sum: Tile<f32, { [1, POINTWISE_COLUMNS] }> =
            constant(0.0f32, shape![1, POINTWISE_COLUMNS]);
        for rank in 0..TOP {
            let v: Tile<f32, { [1, POINTWISE_COLUMNS] }> = convert_tile(
                routed
                    .partition(shape![1, POINTWISE_COLUMNS])
                    .load([p.0 * TOP + rank, p.1]),
            );
            let score: Tile<f32, { [1, 1] }> =
                convert_tile(scores.partition(shape![1, 1]).load([p.0, rank]));
            let weighted: Tile<bf16, { [1, POINTWISE_COLUMNS] }> =
                convert_tile(v * score.broadcast(shape![1, POINTWISE_COLUMNS]));
            let weighted: Tile<f32, { [1, POINTWISE_COLUMNS] }> = convert_tile(weighted);
            sum = sum + weighted;
        }
        let sum: Tile<bf16, { [1, POINTWISE_COLUMNS] }> = convert_tile(sum);
        let sum: Tile<f32, { [1, POINTWISE_COLUMNS] }> = convert_tile(sum);
        let shared: Tile<f32, { [1, POINTWISE_COLUMNS] }> = convert_tile(
            shared
                .partition(shape![1, POINTWISE_COLUMNS])
                .load([p.0, p.1]),
        );
        let gate: Tile<f32, { [1, 1] }> = convert_tile(gate.partition(shape![1, 1]).load([p.0, 0]));

        let sigmoid: Tile<bf16, { [1, 1] }> = convert_tile(one / (one + exp(zero_vector - gate)));
        let sigmoid: Tile<f32, { [1, 1] }> = convert_tile(sigmoid);
        let shared: Tile<bf16, { [1, POINTWISE_COLUMNS] }> =
            convert_tile(shared * sigmoid.broadcast(shape![1, POINTWISE_COLUMNS]));
        let shared: Tile<f32, { [1, POINTWISE_COLUMNS] }> = convert_tile(shared);
        let result: Tile<bf16, { [1, POINTWISE_COLUMNS] }> = convert_tile(sum + shared);
        out.store(result);
    }
}
