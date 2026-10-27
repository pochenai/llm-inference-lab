pub(super) use ops::*;

/// Shared attention/position metadata is a flat i32 tensor: three integers per packed token, followed
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
/// The separate u64 embedding-source table is documented on embedding_gather.
#[cutile::module]
pub(super) mod ops {
    use cutile::core::*;

    /// Integers per packed token: context row, absolute position, history start.
    pub(in crate::runner::model::qwen2) const TOKEN_METADATA_WIDTH: i32 = 3;
    /// Words per packed embedding row: GPU token address, inline host token ID.
    pub(in crate::runner::model::qwen2) const EMBEDDING_SOURCE_WIDTH: i32 = 2;

    #[cutile::entry()]
    pub(super) fn embedding<T: ElementType>(
        out: &mut Tensor<T, { [1, 256] }>,
        ids: &Tensor<u32, { [-1] }>,
        w: &Tensor<T, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        // Validated token IDs fit i32; reinterpret bits without a numeric conversion.
        let id: Tile<i32, { [1] }> = bitcast(ids.partition(shape![1]).load([pid.0]));
        let id: i32 = tile_to_scalar(id.reshape(shape![]));
        out.store(w.partition(shape![1, 256]).load([id, pid.1]));
    }

    /// Embed a host suffix or selected GPU feedback row into its packed row range.
    /// Continuous batching can change request order: previous [A, B] produced GPU
    /// ids=[101, 202], but the current batch is [B, A]. For one token per request,
    /// use (source_start=1, destination_start=0) and (0, 1), respectively:
    /// out[0,:]=w[202,:] belongs to B; out[1,:]=w[101,:] belongs to A.
    /// This writes embeddings directly into the current packed layout, matching
    /// its attention metadata/KV contexts; it does not move previous hidden rows.
    /// Clamp invalid feedback so CPU error reporting cannot race an out-of-bounds load.
    #[cutile::entry()]
    pub(super) unsafe fn embedding_at<T: ElementType>(
        out: *mut T,
        ids: &Tensor<i32, { [-1] }>,
        w: &Tensor<T, { [-1, -1] }>,
        source_start: i32,
        destination_start: i32,
        width: i32,
        vocab_size: i32,
    ) {
        let pid = get_tile_block_id();
        let raw = ids.partition(shape![1]).load([source_start + pid.0]);
        let zero: Tile<i32, { [1] }> = constant(0i32, shape![1]);
        let last: Tile<i32, { [1] }> = (vocab_size - 1).broadcast(shape![1]);
        let id: i32 = tile_to_scalar(min_tile(max_tile(raw, zero), last).reshape(shape![]));
        let value = w
            .partition(shape![1, 256])
            .load([id, pid.1])
            .reshape(shape![256]);
        let columns: Tile<i32, { [256] }> = iota(shape![256]);
        let columns = columns + (pid.1 * 256).broadcast(shape![256]);
        let base: PointerTile<*mut T, { [] }> = pointer_to_tile(out);
        let base = base.offset((destination_start + pid.0) * width);
        let base: PointerTile<*mut T, { [1] }> = base.reshape(shape![1]);
        let base: PointerTile<*mut T, { [256] }> = base.broadcast(shape![256]);
        let destination: PointerTile<*mut T, { [256] }> = base.offset_tile(columns);
        let valid = lt_tile(columns, width.broadcast(shape![256]));
        unsafe {
            store_ptr_tko(
                destination,
                value,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(valid),
                None,
                Latency::<1>,
            );
        }
    }

    /// Embed mixed/reordered sources in one launch. Each packed row has u64
    /// [address, inline_id]; address=0 selects inline_id, otherwise load a GPU i32.
    /// Example: old GPU ids=[101(A),202(B)], current [B,A,C(prompt=[7,8])]:
    /// sources=[&ids[1],0, &ids[0],0, 0,7, 0,8] -> out=[w[202],w[101],w[7],w[8]].
    /// Caller retains every addressed GPU allocation through completion. Output
    /// rows follow current query_ranges; no intermediate token-gather tensor is needed.
    #[cutile::entry()]
    pub(super) unsafe fn embedding_gather<T: ElementType>(
        out: &mut Tensor<T, { [1, 256] }>,
        sources: &Tensor<u64, { [-1] }>,
        w: &Tensor<T, { [-1, -1] }>,
        vocab_size: i32,
    ) {
        let pid = get_tile_block_id();
        let address = sources
            .partition(shape![1])
            .load([pid.0 * EMBEDDING_SOURCE_WIDTH]);
        let inline_id = sources
            .partition(shape![1])
            .load([pid.0 * EMBEDDING_SOURCE_WIDTH + 1]);
        let source: u64 = tile_to_scalar(address.reshape(shape![]));
        // Inline host IDs are vocabulary-validated and fit in i32.
        let mut raw: Tile<i32, { [1] }> = trunci(inline_id, overflow::NoUnsignedWrap);
        if source != 0u64 {
            let pointer: PointerTile<*mut i32, { [1] }> = int_to_ptr(address);
            let (value, _token): (Tile<i32, { [1] }>, Token) = unsafe {
                load_ptr_tko(
                    pointer,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    None,
                    None,
                    Latency::<1>,
                )
            };
            raw = value;
        }
        // Preserve embedding_at's protection while CPU validation runs concurrently.
        let zero: Tile<i32, { [1] }> = constant(0i32, shape![1]);
        let last: Tile<i32, { [1] }> = (vocab_size - 1).broadcast(shape![1]);
        let id: i32 = tile_to_scalar(min_tile(max_tile(raw, zero), last).reshape(shape![]));
        out.store(w.partition(shape![1, 256]).load([id, pid.1]));
    }

    /// Storage uses T; MMA accumulates in FP32 and casts once to output O.
    /// PyTorch checkpoint weights are [out,in]; transpose each tile in registers.
    /// Optional projection bias is fused into the epilogue before the single store.
    /// BM token rows reuse weights; BN covers output channels independently of heads.
    #[cutile::entry(optimization_hints = (sm_120 = (occupancy = 2,),))]
    pub(super) fn linear<
        T: ElementType,
        O: ElementType,
        const K: i32,
        const HAS_BIAS: bool,
        const BM: i32,
        const BN: i32,
    >(
        out: &mut Tensor<O, { [BM, BN] }>,
        x: &Tensor<T, { [-1, K] }>,
        w: &Tensor<T, { [-1, K] }>,
        bias: &Tensor<T, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let px = x.partition(shape![BM, 32]);
        let pw = w.partition(shape![BN, 32]);
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        for i in 0i32..(K / 32) {
            acc = mma(px.load([pid.0, i]), pw.load([pid.1, i]).transpose(), acc);
        }
        if HAS_BIAS {
            let bias_tile: Tile<f32, { [BN] }> =
                convert_tile(bias.partition(shape![BN]).load([pid.1]));
            acc = acc + bias_tile.reshape(shape![1, BN]).broadcast(shape![BM, BN]);
        }
        let result: Tile<O, { [BM, BN] }> = convert_tile(acc);
        out.store(result);
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
    pub(super) fn rms_norm<T: ElementType, const B: i32, const GATHER: bool>(
        out: &mut Tensor<T, { [1, B] }>,
        x: &Tensor<T, { [-1, -1] }>,
        w: &Tensor<T, { [-1] }>,
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
        let value: Tile<f32, { [1, B] }> = convert_tile(x.partition(shape![1, B]).load([row, 0]));
        let sum: Tile<f32, { [1] }> = reduce_sum(value * value, 1);
        let width_f: f32 = convert_scalar(width);
        let mean = sum.reshape(shape![1, 1]) / width_f.broadcast(shape![1, 1]);
        let inv = rsqrt(mean + eps.broadcast(shape![1, 1]), ftz::Disabled);
        let normalized: Tile<T, { [1, B] }> = convert_tile(value * inv.broadcast(shape![1, B]));
        let normalized: Tile<f32, { [1, B] }> = convert_tile(normalized);
        let weight: Tile<f32, { [1, B] }> =
            convert_tile(w.partition(shape![B]).load([0]).reshape(shape![1, B]));
        let result: Tile<T, { [1, B] }> = convert_tile(normalized * weight);
        out.store(result);
    }

    /// Split-half RoPE; FP32 trigonometry with activation-dtype products.
    /// Q/K projections already include their bias.
    /// Frequencies remain FP32 even when activation storage uses BF16.
    #[cutile::entry()]
    pub(super) fn rope<T: ElementType>(
        out: &mut Tensor<T, { [1, 64] }>,
        x: &Tensor<T, { [-1, -1] }>,
        freq: &Tensor<f32, { [32] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let lo = x.partition(shape![1, 32]).load([pid.0, pid.1 * 2]);
        let hi = x.partition(shape![1, 32]).load([pid.0, pid.1 * 2 + 1]);
        let position: i32 = tile_to_scalar(
            metadata
                .partition(shape![1])
                .load([pid.0 * TOKEN_METADATA_WIDTH + 1])
                .reshape(shape![]),
        );
        let position: f32 = convert_scalar(position);
        let angles = freq.partition(shape![32]).load([0]).reshape(shape![1, 32])
            * position.broadcast(shape![1, 32]);
        // Trig uses FP32; operands/products follow the activation dtype, as in Qwen.
        let c: Tile<T, { [1, 32] }> = convert_tile(cos(angles));
        let s: Tile<T, { [1, 32] }> = convert_tile(sin(angles));
        let a = lo * c - hi * s;
        let b = hi * c + lo * s;
        let result: Tile<T, { [1, 64] }> = cat(a, b, 1);
        out.store(result);
    }

    /// Gather BK physical KV slots per iteration, then update online softmax by block.
    /// Mask the final block at each query's causal boundary; slots need not be contiguous.
    /// Target four resident CTAs per SM on SM120 to reduce register pressure.
    #[cutile::entry(optimization_hints = (sm_120 = (occupancy = 4,),))]
    pub(super) unsafe fn attention<T: ElementType, const BK: i32>(
        out: &mut Tensor<T, { [1, 64] }>,
        q: &Tensor<T, { [-1, -1] }>,
        k: &Tensor<T, { [-1, -1] }>,
        v: &Tensor<T, { [-1, -1] }>,
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
        let query: Tile<f32, { [1, 64] }> =
            convert_tile(q.partition(shape![1, 64]).load([pid.0, pid.1]));
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
        let kp: PointerTile<*const T, { [] }> = pointer_to_tile(k.as_ptr());
        let kp: PointerTile<*const T, { [1, 1] }> = kp.reshape(shape![1, 1]);
        let kp: PointerTile<*const T, { [BK, 64] }> = kp.broadcast(shape![BK, 64]);
        let vp: PointerTile<*const T, { [] }> = pointer_to_tile(v.as_ptr());
        let vp: PointerTile<*const T, { [1, 1] }> = vp.reshape(shape![1, 1]);
        let vp: PointerTile<*const T, { [BK, 64] }> = vp.broadcast(shape![BK, 64]);
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
            let key_ptrs: PointerTile<*const T, { [BK, 64] }> = kp.offset_tile(offsets);
            let (keys, _token): (Tile<T, { [BK, 64] }>, Token) = load_ptr_tko(
                key_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None::<T>,
                None,
                Latency::<1>,
            );
            // cuTile 0.4 scalar padding cannot broadcast to a 2D pointer-load result.
            let keys: Tile<f32, { [BK, 64] }> = convert_tile(keys);
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
            let value_ptrs: PointerTile<*const T, { [BK, 64] }> = vp.offset_tile(offsets);
            let (values, _token): (Tile<T, { [BK, 64] }>, Token) = load_ptr_tko(
                value_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None::<T>,
                None,
                Latency::<1>,
            );
            let values: Tile<f32, { [BK, 64] }> = convert_tile(values);
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
        let result: Tile<T, { [1, 64] }> =
            convert_tile(true_div(acc, denominator.broadcast(shape![1, 64])));
        out.store(result);
    }

    /// Multi-query online softmax: reuse each gathered K/V tile across BQ queries.
    /// BF16 operands feed MMA directly; scores, normalizers and accumulators stay FP32.
    /// A launch covers one context/head group, so tiles never cross request boundaries.
    #[cutile::entry(optimization_hints = (sm_120 = (occupancy = 2,),))]
    pub(super) unsafe fn attention_prefill<T: ElementType, const BQ: i32, const BK: i32>(
        out: *mut T,
        q: &Tensor<T, { [-1, -1] }>,
        k: &Tensor<T, { [-1, -1] }>,
        v: &Tensor<T, { [-1, -1] }>,
        table: &Tensor<i32, { [-1, -1] }>,
        context_row: i32,
        query_start: i32,
        query_count: i32,
        absolute_start: i32,
        history_start: i32,
        qheads: i32,
        kvheads: i32,
    ) {
        let pid = get_tile_block_id();
        let local: Tile<i32, { [BQ] }> = iota(shape![BQ]);
        let local = local + (pid.0 * BQ).broadcast(shape![BQ]);
        let valid_q = lt_tile(local, query_count.broadcast(shape![BQ]));
        let positions_q = local + absolute_start.broadcast(shape![BQ]);
        let packed_q = local + query_start.broadcast(shape![BQ]);
        let columns: Tile<i32, { [64] }> = iota(shape![64]);
        let q_columns = columns + (pid.1 * 64).broadcast(shape![64]);
        let q_offsets = (packed_q * q.shape()[1].broadcast(shape![BQ]))
            .reshape(shape![BQ, 1])
            .broadcast(shape![BQ, 64])
            + q_columns.reshape(shape![1, 64]).broadcast(shape![BQ, 64]);
        let q_base: PointerTile<*const T, { [] }> = pointer_to_tile(q.as_ptr());
        let q_base: PointerTile<*const T, { [1, 1] }> = q_base.reshape(shape![1, 1]);
        let q_base: PointerTile<*const T, { [BQ, 64] }> = q_base.broadcast(shape![BQ, 64]);
        let q_ptrs: PointerTile<*const T, { [BQ, 64] }> = q_base.offset_tile(q_offsets);
        let q_mask = valid_q.reshape(shape![BQ, 1]).broadcast(shape![BQ, 64]);
        let zero_q_f32: Tile<f32, { [BQ, 64] }> = constant(0.0f32, shape![BQ, 64]);
        let zero_q: Tile<T, { [BQ, 64] }> = convert_tile(zero_q_f32);
        let (query, _token): (Tile<T, { [BQ, 64] }>, Token) = load_ptr_tko(
            q_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(q_mask),
            None::<T>,
            None,
            Latency::<1>,
        );
        let query = select(q_mask, query, zero_q);
        let mut maximum: Tile<f32, { [BQ, 1] }> = constant(f32::NEG_INFINITY, shape![BQ, 1]);
        let mut denominator: Tile<f32, { [BQ, 1] }> = constant(0.0f32, shape![BQ, 1]);
        let mut acc: Tile<f32, { [BQ, 64] }> = constant(0.0f32, shape![BQ, 64]);
        let zero_scores: Tile<f32, { [BQ, BK] }> = constant(0.0f32, shape![BQ, BK]);
        let negative: Tile<f32, { [BQ, BK] }> = constant(f32::NEG_INFINITY, shape![BQ, BK]);
        let zero_kv_f32: Tile<f32, { [BK, 64] }> = constant(0.0f32, shape![BK, 64]);
        let zero_kv: Tile<T, { [BK, 64] }> = convert_tile(zero_kv_f32);
        let zero_rows: Tile<f32, { [BQ, 1] }> = constant(0.0f32, shape![BQ, 1]);
        let valid_rows = valid_q.reshape(shape![BQ, 1]);
        let causal_q = (positions_q + 1i32.broadcast(shape![BQ]))
            .reshape(shape![BQ, 1])
            .broadcast(shape![BQ, BK]);
        let kv_head = pid.1 / (qheads / kvheads);
        let end = (pid.0 + 1) * BQ;
        let end = if end < query_count { end } else { query_count };
        let visible_end = absolute_start + end;
        let offsets_k: Tile<i32, { [BK] }> = iota(shape![BK]);
        let table_base: PointerTile<*const i32, { [] }> = pointer_to_tile(table.as_ptr());
        let table_base: PointerTile<*const i32, { [] }> =
            table_base.offset(context_row * table.shape()[1]);
        let table_base: PointerTile<*const i32, { [1] }> = table_base.reshape(shape![1]);
        let table_base: PointerTile<*const i32, { [BK] }> = table_base.broadcast(shape![BK]);
        let k_base: PointerTile<*const T, { [] }> = pointer_to_tile(k.as_ptr());
        let v_base: PointerTile<*const T, { [] }> = pointer_to_tile(v.as_ptr());
        for block in 0i32..((visible_end - history_start + BK - 1) / BK) {
            let positions_k = offsets_k + (history_start + block * BK).broadcast(shape![BK]);
            let valid_k = lt_tile(positions_k, visible_end.broadcast(shape![BK]));
            let table_ptrs: PointerTile<*const i32, { [BK] }> = table_base.offset_tile(positions_k);
            let (slots, _token): (Tile<i32, { [BK] }>, Token) = load_ptr_tko(
                table_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(valid_k),
                Some(0i32),
                None,
                Latency::<1>,
            );
            let offsets =
                slots * (kvheads * 64).broadcast(shape![BK]) + (kv_head * 64).broadcast(shape![BK]);
            let offsets = offsets.reshape(shape![BK, 1]).broadcast(shape![BK, 64])
                + columns.reshape(shape![1, 64]).broadcast(shape![BK, 64]);
            let k_base_2d: PointerTile<*const T, { [1, 1] }> = k_base.reshape(shape![1, 1]);
            let k_base_2d: PointerTile<*const T, { [BK, 64] }> =
                k_base_2d.broadcast(shape![BK, 64]);
            let k_ptrs: PointerTile<*const T, { [BK, 64] }> = k_base_2d.offset_tile(offsets);
            let v_base_2d: PointerTile<*const T, { [1, 1] }> = v_base.reshape(shape![1, 1]);
            let v_base_2d: PointerTile<*const T, { [BK, 64] }> =
                v_base_2d.broadcast(shape![BK, 64]);
            let v_ptrs: PointerTile<*const T, { [BK, 64] }> = v_base_2d.offset_tile(offsets);
            let kv_mask = valid_k.reshape(shape![BK, 1]).broadcast(shape![BK, 64]);
            let (keys, _token): (Tile<T, { [BK, 64] }>, Token) = load_ptr_tko(
                k_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(kv_mask),
                None::<T>,
                None,
                Latency::<1>,
            );
            let keys = select(kv_mask, keys, zero_kv);
            let scores: Tile<f32, { [BQ, BK] }> = mma(query, keys.transpose(), zero_scores);
            let causal = lt_tile(
                positions_k.reshape(shape![1, BK]).broadcast(shape![BQ, BK]),
                causal_q,
            ) & valid_q.reshape(shape![BQ, 1]).broadcast(shape![BQ, BK]);
            let scores = select(
                causal,
                scores * 0.125f32.broadcast(shape![BQ, BK]),
                negative,
            );
            let block_max: Tile<f32, { [BQ] }> = reduce_max(scores, 1);
            let next = max_tile(maximum, block_max.reshape(shape![BQ, 1]));
            // Padded query rows must not introduce -inf - -inf into the MMA operands.
            let next = select(valid_rows, next, zero_rows);
            let alpha = exp(maximum - next);
            let probabilities = exp(scores - next.broadcast(shape![BQ, BK]));
            let sum: Tile<f32, { [BQ] }> = reduce_sum(probabilities, 1);
            denominator = denominator * alpha + sum.reshape(shape![BQ, 1]);
            let p: Tile<T, { [BQ, BK] }> = convert_tile(probabilities);
            let (values, _token): (Tile<T, { [BK, 64] }>, Token) = load_ptr_tko(
                v_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(kv_mask),
                None::<T>,
                None,
                Latency::<1>,
            );
            let values = select(kv_mask, values, zero_kv);
            acc = mma(p, values, acc * alpha.broadcast(shape![BQ, 64]));
            maximum = next;
        }
        let result: Tile<T, { [BQ, 64] }> =
            convert_tile(true_div(acc, denominator.broadcast(shape![BQ, 64])));
        let out_base: PointerTile<*mut T, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut T, { [1, 1] }> = out_base.reshape(shape![1, 1]);
        let out_base: PointerTile<*mut T, { [BQ, 64] }> = out_base.broadcast(shape![BQ, 64]);
        let out_ptrs: PointerTile<*mut T, { [BQ, 64] }> = out_base.offset_tile(q_offsets);
        store_ptr_tko(
            out_ptrs,
            result,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(q_mask),
            None,
            Latency::<1>,
        );
    }

    #[cutile::entry()]
    pub(super) fn add<T: ElementType>(
        out: &mut Tensor<T, { [1, 256] }>,
        x: &Tensor<T, { [-1, -1] }>,
        y: &Tensor<T, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let x: Tile<f32, { [1, 256] }> =
            convert_tile(x.partition(shape![1, 256]).load([pid.0, pid.1]));
        let y: Tile<f32, { [1, 256] }> =
            convert_tile(y.partition(shape![1, 256]).load([pid.0, pid.1]));
        let result: Tile<T, { [1, 256] }> = convert_tile(x + y);
        out.store(result);
    }

    #[cutile::entry()]
    pub(super) fn swiglu<T: ElementType>(
        out: &mut Tensor<T, { [1, 256] }>,
        gate: &Tensor<T, { [-1, -1] }>,
        up: &Tensor<T, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let g: Tile<f32, { [1, 256] }> =
            convert_tile(gate.partition(shape![1, 256]).load([pid.0, pid.1]));
        let one: Tile<f32, { [1, 256] }> = constant(1.0f32, shape![1, 256]);
        let zero: Tile<f32, { [1, 256] }> = constant(0.0f32, shape![1, 256]);
        let activated: Tile<T, { [1, 256] }> = convert_tile(true_div(g, one + exp(zero - g)));
        let activated: Tile<f32, { [1, 256] }> = convert_tile(activated);
        let up: Tile<f32, { [1, 256] }> =
            convert_tile(up.partition(shape![1, 256]).load([pid.0, pid.1]));
        let result: Tile<T, { [1, 256] }> = convert_tile(activated * up);
        out.store(result);
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
