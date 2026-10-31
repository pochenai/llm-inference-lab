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

    /// Head-vector width, independent of the query/key tile sizes BQ/BK.
    const HEAD_DIM: i32 = 64;

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
        let base = pointer_to_tile(out);
        let base = base.offset((destination_start + pid.0) * width);
        let base = base.reshape(shape![1]);
        let base = base.broadcast(shape![256]);
        let destination = base.offset_tile(columns);
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

    /// Shared projection math; epilogues retain their own activation-dtype rounding.
    /// Callers reshape concrete results to their symbolic tile shape for cuTile 0.4.
    fn project_tile<T: ElementType, const K: i32, const BM: i32, const BN: i32, const BK: i32>(
        x: &Tensor<T, { [-1, K] }>,
        w: &Tensor<T, { [-1, K] }>,
        row: i32,
        column: i32,
    ) -> Tile<f32, { [BM, BN] }> {
        let px = x.partition(shape![BM, BK]);
        let pw = w.partition(shape![BN, BK]);
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        for i in 0i32..((K + BK - 1) / BK) {
            acc = mma(px.load([row, i]), pw.load([column, i]).transpose(), acc);
        }
        acc
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
        const BK: i32,
    >(
        out: &mut Tensor<O, { [BM, BN] }>,
        x: &Tensor<T, { [-1, K] }>,
        w: &Tensor<T, { [-1, K] }>,
        bias: &Tensor<T, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let mut acc: Tile<f32, { [BM, BN] }> =
            project_tile::<T, K, BM, BN, BK>(x, w, pid.0, pid.1).reshape(shape![BM, BN]);
        if HAS_BIAS {
            let bias_tile: Tile<f32, { [BN] }> =
                convert_tile(bias.partition(shape![BN]).load([pid.1]));
            acc = acc + bias_tile.reshape(shape![1, BN]).broadcast(shape![BM, BN]);
        }
        let result: Tile<O, { [BM, BN] }> = convert_tile(acc);
        out.store(result);
    }

    /// Fuse the up projection with SiLU(gate) * up, avoiding a global up temporary.
    /// Round both up and SiLU to T before multiplying, matching the unfused kernels.
    #[cutile::entry(optimization_hints = (sm_120 = (occupancy = 2,),))]
    pub(super) fn linear_swiglu<
        T: ElementType,
        const K: i32,
        const BM: i32,
        const BN: i32,
        const BK: i32,
    >(
        out: &mut Tensor<T, { [BM, BN] }>,
        x: &Tensor<T, { [-1, K] }>,
        w: &Tensor<T, { [-1, K] }>,
        gate: &Tensor<T, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let acc: Tile<f32, { [BM, BN] }> =
            project_tile::<T, K, BM, BN, BK>(x, w, pid.0, pid.1).reshape(shape![BM, BN]);
        let up: Tile<T, { [BM, BN] }> = convert_tile(acc);
        let up: Tile<f32, { [BM, BN] }> = convert_tile(up);
        let g: Tile<f32, { [BM, BN] }> =
            convert_tile(gate.partition(shape![BM, BN]).load([pid.0, pid.1]));
        let one: Tile<f32, { [BM, BN] }> = constant(1.0f32, shape![BM, BN]);
        let zero: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        let activated: Tile<T, { [BM, BN] }> = convert_tile(true_div(g, one + exp(zero - g)));
        let activated: Tile<f32, { [BM, BN] }> = convert_tile(activated);
        let result: Tile<T, { [BM, BN] }> = convert_tile(activated * up);
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
    /// With GATHER=false, normalize ROWS adjacent rows per CTA; ignore last_rows_offset.
    /// GATHER=true uses ROWS=1 so arbitrary request endpoints remain independent.
    #[cutile::entry()]
    pub(super) fn rms_norm<T: ElementType, const B: i32, const GATHER: bool, const ROWS: i32>(
        out: &mut Tensor<T, { [ROWS, B] }>,
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
        let value: Tile<f32, { [ROWS, B] }> =
            convert_tile(x.partition(shape![ROWS, B]).load([row, 0]));
        let sum: Tile<f32, { [ROWS] }> = reduce_sum(value * value, 1);
        let width_f: f32 = convert_scalar(width);
        let mean = sum.reshape(shape![ROWS, 1]) / width_f.broadcast(shape![ROWS, 1]);
        let inv = rsqrt(mean + eps.broadcast(shape![ROWS, 1]), ftz::Disabled);
        let normalized: Tile<T, { [ROWS, B] }> =
            convert_tile(value * inv.broadcast(shape![ROWS, B]));
        let normalized: Tile<f32, { [ROWS, B] }> = convert_tile(normalized);
        let weight: Tile<f32, { [ROWS, B] }> = convert_tile(
            w.partition(shape![B])
                .load([0])
                .reshape(shape![1, B])
                .broadcast(shape![ROWS, B]),
        );
        let result: Tile<T, { [ROWS, B] }> = convert_tile(normalized * weight);
        out.store(result);
    }

    /// Split-half RoPE; FP32 trigonometry with activation-dtype products.
    /// Q/K projections already include their bias.
    /// Frequencies remain FP32 even when activation storage uses BF16.
    #[cutile::entry()]
    pub(super) unsafe fn rope<T: ElementType, const ROWS: i32>(
        out: &mut Tensor<T, { [ROWS, 64] }>,
        x: &Tensor<T, { [-1, -1] }>,
        freq: &Tensor<f32, { [32] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let lo = x.partition(shape![ROWS, 32]).load([pid.0, pid.1 * 2]);
        let hi = x.partition(shape![ROWS, 32]).load([pid.0, pid.1 * 2 + 1]);
        let rows: Tile<i32, { [ROWS] }> = iota(shape![ROWS]);
        let rows = rows + (pid.0 * ROWS).broadcast(shape![ROWS]);
        let valid = lt_tile(rows, x.shape()[0].broadcast(shape![ROWS]));
        let base = pointer_to_tile(metadata.as_ptr());
        let base = base.reshape(shape![1]);
        let base = base.broadcast(shape![ROWS]);
        let offsets =
            rows * TOKEN_METADATA_WIDTH.broadcast(shape![ROWS]) + 1i32.broadcast(shape![ROWS]);
        let pointers = base.offset_tile(offsets);
        let (positions, _token): (Tile<i32, { [ROWS] }>, Token) = load_ptr_tko(
            pointers,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(valid),
            Some(0i32),
            None,
            Latency::<1>,
        );
        let positions: Tile<f32, { [ROWS] }> = convert_tile(positions);
        let angles = freq
            .partition(shape![32])
            .load([0])
            .reshape(shape![1, 32])
            .broadcast(shape![ROWS, 32])
            * positions
                .reshape(shape![ROWS, 1])
                .broadcast(shape![ROWS, 32]);
        // Trig uses FP32; operands/products follow the activation dtype, as in Qwen.
        let c: Tile<T, { [ROWS, 32] }> = convert_tile(cos(angles));
        let s: Tile<T, { [ROWS, 32] }> = convert_tile(sin(angles));
        let a = lo * c - hi * s;
        let b = hi * c + lo * s;
        let result: Tile<T, { [ROWS, 64] }> = cat(a, b, 1);
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
        let table_base = pointer_to_tile(table.as_ptr());
        let table_base = table_base.offset(row * table.shape()[1]);
        let table_base = table_base.reshape(shape![1]);
        let table_base = table_base.broadcast(shape![BK]);
        let kp = pointer_to_tile(k.as_ptr());
        let kp = kp.reshape(shape![1, 1]);
        let kp = kp.broadcast(shape![BK, 64]);
        let vp = pointer_to_tile(v.as_ptr());
        let vp = vp.reshape(shape![1, 1]);
        let vp = vp.broadcast(shape![BK, 64]);
        for block in 0i32..((position - history_start + BK) / BK) {
            let positions = local + (history_start + block * BK).broadcast(shape![BK]);
            let valid = lt_tile(positions, (position + 1).broadcast(shape![BK]));
            // Read only mapped, visible positions; masked lanes never dereference a slot.
            let table_ptrs = table_base.offset_tile(positions);
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
            let key_ptrs = kp.offset_tile(offsets);
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
            let value_ptrs = vp.offset_tile(offsets);
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

    /// Causal prefill attention for one context, over the visible key/value history.
    /// https://developer.nvidia.com/blog/tuning-flash-attention-for-peak-performance-in-nvidia-cuda-tile/
    /// ```text
    /// kv_head = query_heads / (qheads / kvheads)
    /// causal  = valid_q && (history_start <= key_positions <= query_positions)
    ///
    /// scores = query * keys^T / sqrt(HEAD_DIM)
    /// scores = where(causal, scores, -infinity)
    /// result = cast_T(softmax_rows(scores) * values)
    /// ```
    /// Q/K/V projections are already complete; the causal mask applies to scores only.
    ///
    /// The implementation streams key/value blocks with these tile shapes:
    /// ```text
    /// query        : [BQ, HEAD_DIM]
    /// keys, values : [BK, HEAD_DIM]
    /// scores       : [BQ, BK]
    /// acc          : [BQ, HEAD_DIM]
    ///
    /// (qheads / kvheads) % GROUP = 0
    /// ```
    /// Single-head groups pack query tokens; larger groups also pack Q heads sharing K/V.
    /// Every loaded K/V tile serves all packed query rows within this context.
    /// MMA operands/weights use T; online maxima, sums and accumulation use FP32.
    ///
    /// Running state for each valid query row, over processed keys:
    /// ```text
    /// maximum     = max(scores)
    /// denominator = sum(exp(scores - maximum))
    /// acc         = sum(exp(scores - maximum) * values)
    /// result      = cast_T(acc / denominator)
    /// ```
    #[cutile::entry(optimization_hints = (sm_120 = (occupancy = 2,),))]
    pub(super) unsafe fn attention_prefill<
        T: ElementType,
        const BQ: i32,
        const BK: i32,
        const GROUP: i32,
    >(
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
        // Map low CTA IDs to longer causal histories to reduce the execution tail.
        // This is a scheduling heuristic; correctness never depends on CTA order. 
        // ~ 2% improvement.
        let query_tiles = (query_count * GROUP + BQ - 1) / BQ;
        let query_tile = query_tiles - 1 - pid.0;
        // query_head_rows enumerates (token, head-in-group) pairs in this append.
        // query_indices indexes its new tokens; query_heads identifies model Q heads.
        // ```text
        // query_head_rows = iota(BQ) + query_tile * BQ
        // query_indices   = query_head_rows / GROUP
        // query_heads     = pid.1 * GROUP + query_head_rows % GROUP
        //
        // GROUP=7, query_tile=0, pid.1=0:
        // query_head_rows : 0  1  2  3  4  5  6  7 ...
        // query_indices   : 0  0  0  0  0  0  0  1 ...
        // query_heads     : 0  1  2  3  4  5  6  0 ...
        // ```
        let query_head_rows: Tile<i32, { [BQ] }> = iota(shape![BQ]);
        let query_head_rows = query_head_rows + (query_tile * BQ).broadcast(shape![BQ]);
        let query_heads =
            query_head_rows % GROUP.broadcast(shape![BQ]) + (pid.1 * GROUP).broadcast(shape![BQ]);
        let query_indices = query_head_rows / GROUP.broadcast(shape![BQ]);
        let valid_q = lt_tile(query_indices, query_count.broadcast(shape![BQ]));
        // Distinguish absolute history positions from rows in the packed batch tensors.
        // ```text
        // query_positions = absolute_start + query_indices
        // query_rows      = query_start + query_indices
        // ```
        let query_positions = query_indices + absolute_start.broadcast(shape![BQ]);
        let query_rows = query_indices + query_start.broadcast(shape![BQ]);
        let head_columns: Tile<i32, { [HEAD_DIM] }> = iota(shape![HEAD_DIM]);

        // Flatten the logical query coordinates.
        // ```text
        // q_offsets = query_rows * q.shape()[1]
        //             + query_heads * HEAD_DIM + head_columns
        // ```
        let q_offsets = (query_rows * q.shape()[1].broadcast(shape![BQ]))
            .reshape(shape![BQ, 1])
            .broadcast(shape![BQ, HEAD_DIM])
            + (query_heads * HEAD_DIM.broadcast(shape![BQ]))
                .reshape(shape![BQ, 1])
                .broadcast(shape![BQ, HEAD_DIM])
            + head_columns
                .reshape(shape![1, HEAD_DIM])
                .broadcast(shape![BQ, HEAD_DIM]);
        let q_base = pointer_to_tile(q.as_ptr());
        let q_base = q_base.reshape(shape![1, 1]);
        let q_base = q_base.broadcast(shape![BQ, HEAD_DIM]);
        let q_ptrs = q_base.offset_tile(q_offsets);
        let q_mask = valid_q
            .reshape(shape![BQ, 1])
            .broadcast(shape![BQ, HEAD_DIM]);
        let zero_q_f32: Tile<f32, { [BQ, HEAD_DIM] }> = constant(0.0f32, shape![BQ, HEAD_DIM]);
        let zero_q: Tile<T, { [BQ, HEAD_DIM] }> = convert_tile(zero_q_f32);
        let (query, _token): (Tile<T, { [BQ, HEAD_DIM] }>, Token) = load_ptr_tko(
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
        let mut acc: Tile<f32, { [BQ, HEAD_DIM] }> = constant(0.0f32, shape![BQ, HEAD_DIM]);
        let zero_scores: Tile<f32, { [BQ, BK] }> = constant(0.0f32, shape![BQ, BK]);
        let negative: Tile<f32, { [BQ, BK] }> = constant(f32::NEG_INFINITY, shape![BQ, BK]);
        let zero_kv_f32: Tile<f32, { [BK, HEAD_DIM] }> = constant(0.0f32, shape![BK, HEAD_DIM]);
        let zero_kv: Tile<T, { [BK, HEAD_DIM] }> = convert_tile(zero_kv_f32);
        let zero_rows: Tile<f32, { [BQ, 1] }> = constant(0.0f32, shape![BQ, 1]);
        let valid_rows = valid_q.reshape(shape![BQ, 1]);
        let head_dim: f32 = convert_scalar(HEAD_DIM);
        let scale: Tile<f32, { [BQ, BK] }> =
            rsqrt(head_dim.broadcast(shape![BQ, BK]), ftz::Disabled);
        let causal_q = (query_positions + 1i32.broadcast(shape![BQ]))
            .reshape(shape![BQ, 1])
            .broadcast(shape![BQ, BK]);
        let kv_head = (pid.1 * GROUP) / (qheads / kvheads);
        let end = ((query_tile + 1) * BQ + GROUP - 1) / GROUP;
        let end = if end < query_count { end } else { query_count };
        let visible_end = absolute_start + end;
        let key_indices: Tile<i32, { [BK] }> = iota(shape![BK]);
        let table_base = pointer_to_tile(table.as_ptr());
        let table_base = table_base.offset(context_row * table.shape()[1]);
        let table_base = table_base.reshape(shape![1]);
        let table_base = table_base.broadcast(shape![BK]);
        let k_base = pointer_to_tile(k.as_ptr());
        let v_base = pointer_to_tile(v.as_ptr());
        // Reuse each key/value block across packed query rows; never materialize
        // the full attention score matrix.
        for block in 0i32..((visible_end - history_start + BK - 1) / BK) {
            let key_positions = key_indices + (history_start + block * BK).broadcast(shape![BK]);
            let valid_k = lt_tile(key_positions, visible_end.broadcast(shape![BK]));
            let table_ptrs = table_base.offset_tile(key_positions);
            let (slots, _token): (Tile<i32, { [BK] }>, Token) = load_ptr_tko(
                table_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(valid_k),
                Some(0i32),
                None,
                Latency::<1>,
            );
            // Resolve logical key positions to physical cache coordinates.
            // ```text
            // slots      = table[context_row, key_positions]
            // kv_offsets = (slots * kvheads + kv_head) * HEAD_DIM + head_columns
            // ```
            let kv_offsets = slots * (kvheads * HEAD_DIM).broadcast(shape![BK])
                + (kv_head * HEAD_DIM).broadcast(shape![BK]);
            let kv_offsets = kv_offsets
                .reshape(shape![BK, 1])
                .broadcast(shape![BK, HEAD_DIM])
                + head_columns
                    .reshape(shape![1, HEAD_DIM])
                    .broadcast(shape![BK, HEAD_DIM]);
            let k_base_2d = k_base.reshape(shape![1, 1]);
            let k_base_2d = k_base_2d.broadcast(shape![BK, HEAD_DIM]);
            let k_ptrs = k_base_2d.offset_tile(kv_offsets);
            let v_base_2d = v_base.reshape(shape![1, 1]);
            let v_base_2d = v_base_2d.broadcast(shape![BK, HEAD_DIM]);
            let v_ptrs = v_base_2d.offset_tile(kv_offsets);
            let kv_mask = valid_k
                .reshape(shape![BK, 1])
                .broadcast(shape![BK, HEAD_DIM]);
            let (keys, _token): (Tile<T, { [BK, HEAD_DIM] }>, Token) = load_ptr_tko(
                k_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(kv_mask),
                None::<T>,
                None,
                Latency::<1>,
            );
            let keys = select(kv_mask, keys, zero_kv);
            // Block dot products.
            // ```text
            // scores = query * keys^T
            // ```
            let scores: Tile<f32, { [BQ, BK] }> = mma(query, keys.transpose(), zero_scores);
            let causal = lt_tile(
                key_positions
                    .reshape(shape![1, BK])
                    .broadcast(shape![BQ, BK]),
                causal_q,
            ) & valid_q.reshape(shape![BQ, 1]).broadcast(shape![BQ, BK]);
            // Scale and apply the row-specific causal mask.
            // ```text
            // scores = where(causal, scores * scale, -infinity)
            // ```
            let scores = select(causal, scores * scale, negative);
            // Update the running row maximum.
            // ```text
            // block_max    = rowmax(scores)
            // next_maximum = max(maximum, block_max)
            // ```
            let block_max: Tile<f32, { [BQ] }> = reduce_max(scores, 1);
            let next_maximum = max_tile(maximum, block_max.reshape(shape![BQ, 1]));
            // Padded query rows must not introduce -inf - -inf into the MMA operands.
            let next_maximum = select(valid_rows, next_maximum, zero_rows);
            // Rebase earlier blocks and form unnormalized weights for this block.
            // ```text
            // alpha         = exp(maximum - next_maximum)
            // block_weights = exp(scores - next_maximum)
            // ```
            let alpha = exp(maximum - next_maximum);
            let block_weights = exp(scores - next_maximum.broadcast(shape![BQ, BK]));
            let block_sum: Tile<f32, { [BQ] }> = reduce_sum(block_weights, 1);
            // Accumulate the FP32 normalizer.
            // ```text
            // block_sum   = rowsum(block_weights)
            // denominator = alpha * denominator + block_sum
            // ```
            denominator = denominator * alpha + block_sum.reshape(shape![BQ, 1]);
            let mma_weights: Tile<T, { [BQ, BK] }> = convert_tile(block_weights);
            let (values, _token): (Tile<T, { [BK, HEAD_DIM] }>, Token) = load_ptr_tko(
                v_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(kv_mask),
                None::<T>,
                None,
                Latency::<1>,
            );
            let values = select(kv_mask, values, zero_kv);
            // Accumulate the weighted values using MMA.
            // ```text
            // mma_weights = cast_T(block_weights)
            // acc         = alpha * acc + mma_weights * values
            // maximum     = next_maximum
            // ```
            acc = mma(
                mma_weights,
                values,
                acc * alpha.broadcast(shape![BQ, HEAD_DIM]),
            );
            maximum = next_maximum;
        }
        // Normalize once after all key blocks.
        // ```text
        // result = cast_T(acc / denominator)
        // ```
        let result: Tile<T, { [BQ, HEAD_DIM] }> =
            convert_tile(true_div(acc, denominator.broadcast(shape![BQ, HEAD_DIM])));
        let out_base = pointer_to_tile(out);
        let out_base = out_base.reshape(shape![1, 1]);
        let out_base = out_base.broadcast(shape![BQ, HEAD_DIM]);
        let out_ptrs = out_base.offset_tile(q_offsets);
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
    pub(super) fn add<T: ElementType, const ROWS: i32>(
        out: &mut Tensor<T, { [ROWS, 256] }>,
        x: &Tensor<T, { [-1, -1] }>,
        y: &Tensor<T, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let x: Tile<f32, { [ROWS, 256] }> =
            convert_tile(x.partition(shape![ROWS, 256]).load([pid.0, pid.1]));
        let y: Tile<f32, { [ROWS, 256] }> =
            convert_tile(y.partition(shape![ROWS, 256]).load([pid.0, pid.1]));
        let result: Tile<T, { [ROWS, 256] }> = convert_tile(x + y);
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
