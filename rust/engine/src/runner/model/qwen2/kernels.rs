pub(in crate::runner::model) use ops::*;

/// Amortize reduction overhead; Qwen's vocabulary needs 38 partials/request.
/// Warm 4K B1/B4: ~76%/83% less argmax GPU time in separate Nsight captures.
pub(super) const ARGMAX_TILE: usize = 4096;
/// Covers the supported vocabulary limit with ARGMAX_TILE-sized partials.
pub(super) const ARGMAX_FINAL_TILE: usize = 64;

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
    /// Rounded f32 log2(e); cuTile 0.4 cannot lower the std constant path here.
    const LOG2_E: f32 = 1.442_695;

    /// Integers per packed token: context row, absolute position, history start.
    pub(in crate::runner::model::qwen2) const TOKEN_METADATA_WIDTH: i32 = 3;
    /// Words per packed embedding row: GPU token address, inline host token ID.
    pub(in crate::runner::model::qwen2) const EMBEDDING_SOURCE_WIDTH: i32 = 2;

    #[cutile::entry()]
    pub(in crate::runner::model) fn embedding<T: ElementType>(
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
    /// Partition loads zero-pad out-of-bounds rows: BM=16 can compute four valid rows
    /// without expanding the input Tensor; each epilogue masks invalid output rows.
    fn project_tile<
        T: ElementType,
        const K: i32,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const SPLITS: i32,
    >(
        x: &Tensor<T, { [-1, K] }>,
        w: &Tensor<T, { [-1, K] }>,
        row: i32,
        column: i32,
        split: i32,
    ) -> Tile<f32, { [BM, BN] }> {
        let px = x.partition(shape![BM, BK]);
        let pw = w.partition(shape![BN, BK]);
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        let blocks = (K + BK - 1) / BK;
        for i in (blocks * split / SPLITS)..(blocks * (split + 1) / SPLITS) {
            acc = mma(px.load([row, i]), pw.load([column, i]).transpose(), acc);
        }
        acc
    }

    /// Project one output tile with the ordinary MMA/bias math and a masked scatter.
    unsafe fn project_decode_tile<
        T: ElementType,
        const K: i32,
        const BM: i32,
        const BN: i32,
        const BK: i32,
    >(
        out: *mut T,
        x: &Tensor<T, { [-1, K] }>,
        w: &Tensor<T, { [-1, K] }>,
        bias: &Tensor<T, { [-1] }>,
        row: i32,
        column: i32,
    ) {
        let acc = project_tile::<T, K, BM, BN, BK, 1>(x, w, row, column, 0).reshape(shape![BM, BN]);
        let bias: Tile<f32, { [BN] }> = convert_tile(bias.partition(shape![BN]).load([column]));
        let result: Tile<T, { [BM, BN] }> =
            convert_tile(acc + bias.reshape(shape![1, BN]).broadcast(shape![BM, BN]));
        let rows: Tile<i32, { [BM] }> = iota(shape![BM]);
        let rows = rows + (row * BM).broadcast(shape![BM]);
        let columns: Tile<i32, { [BN] }> = iota(shape![BN]);
        let columns = columns + (column * BN).broadcast(shape![BN]);
        let offsets = (rows * w.shape()[0].broadcast(shape![BM]))
            .reshape(shape![BM, 1])
            .broadcast(shape![BM, BN])
            + columns.reshape(shape![1, BN]).broadcast(shape![BM, BN]);
        let mask = lt_tile(rows, x.shape()[0].broadcast(shape![BM]))
            .reshape(shape![BM, 1])
            .broadcast(shape![BM, BN])
            & lt_tile(columns, w.shape()[0].broadcast(shape![BN]))
                .reshape(shape![1, BN])
                .broadcast(shape![BM, BN]);
        let base = pointer_to_tile(out);
        let base = base.reshape(shape![1, 1]);
        let base = base.broadcast(shape![BM, BN]);
        let dst = base.offset_tile(offsets);
        store_ptr_tko(
            dst,
            result,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<1>,
        );
    }

    /// One skinny QKV launch schedules all output tiles together without concatenating weights.
    /// Qwen decode: 28 Q + 4 K + 4 V CTAs share one grid, instead of three underfilled grids.
    /// For small BF16 batches, [BM,BN,BK]=[16,32,64] generates HMMA; the former
    /// [4,32,32] generated SIMT here. Zero-padded loads and masked stores keep batch=4.
    /// 4K B4: ~34% QKV GPU-time gain; QKV+down tile changes together give ~5.1% decode wall-time gain.
    /// HMMA/BF16 instruction shapes and FP32 accumulation:
    /// https://docs.nvidia.com/cuda/archive/12.4.0/ampere-tuning-guide/index.html#improved-tensor-core-operations
    /// SGLang v0.5.9 QKV -> ColumnParallelLinear -> unquantized F.linear:
    /// https://github.com/sgl-project/sglang/blob/v0.5.9/python/sglang/srt/models/qwen2.py#L179-L190
    /// https://github.com/sgl-project/sglang/blob/v0.5.9/python/sglang/srt/layers/linear.py#L442-L452
    /// https://github.com/sgl-project/sglang/blob/v0.5.9/python/sglang/srt/layers/quantization/unquant.py#L132-L152
    /// Those calls delegate kernel selection; HMMA was verified in our local NCU/SASS capture.
    #[cutile::entry(optimization_hints = (sm_120 = (occupancy = 2,),))]
    pub(super) unsafe fn qkv_decode<
        T: ElementType,
        const K: i32,
        const BM: i32,
        const BN: i32,
        const BK: i32,
    >(
        q: *mut T,
        k: *mut T,
        v: *mut T,
        x: &Tensor<T, { [-1, K] }>,
        wq: &Tensor<T, { [-1, K] }>,
        wk: &Tensor<T, { [-1, K] }>,
        wv: &Tensor<T, { [-1, K] }>,
        bq: &Tensor<T, { [-1] }>,
        bk: &Tensor<T, { [-1] }>,
        bv: &Tensor<T, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let q_blocks = (wq.shape()[0] + BN - 1) / BN;
        let k_blocks = (wk.shape()[0] + BN - 1) / BN;
        // Keep branch-local indices bounded independently of the combined launch grid.
        if pid.1 < q_blocks {
            project_decode_tile::<T, K, BM, BN, BK>(q, x, wq, bq, pid.0, pid.1 % q_blocks);
        } else if pid.1 < q_blocks + k_blocks {
            project_decode_tile::<T, K, BM, BN, BK>(
                k,
                x,
                wk,
                bk,
                pid.0,
                (pid.1 - q_blocks) % k_blocks,
            );
        } else {
            project_decode_tile::<T, K, BM, BN, BK>(
                v,
                x,
                wv,
                bv,
                pid.0,
                (pid.1 - q_blocks - k_blocks) % ((wv.shape()[0] + BN - 1) / BN),
            );
        }
    }

    /// Split a skinny projection's K reduction across CTAs; share the ordinary MMA tile.
    /// The small-BF16 [16,32,64] tile shortens the K loop while preserving split buckets
    /// and the existing FP32 partial buffer; padded token rows are masked on writeback.
    /// 4K B4: ~15% down+merge GPU-time gain; QKV+down tiles together give ~5.1% decode wall-time gain.
    /// HMMA/BF16 instruction shapes and FP32 accumulation:
    /// https://docs.nvidia.com/cuda/archive/12.4.0/ampere-tuning-guide/index.html#improved-tensor-core-operations
    /// SGLang v0.5.9 down -> RowParallelLinear -> unquantized F.linear:
    /// https://github.com/sgl-project/sglang/blob/v0.5.9/python/sglang/srt/models/qwen2.py#L95-L100
    /// https://github.com/sgl-project/sglang/blob/v0.5.9/python/sglang/srt/layers/linear.py#L1468-L1479
    /// https://github.com/sgl-project/sglang/blob/v0.5.9/python/sglang/srt/layers/quantization/unquant.py#L132-L152
    /// ```text
    /// grid = [ceil(M/BM), ceil(N/BN), SPLITS]
    /// partials[split, row, column] = X[row, K_split] * W[column, K_split]^T
    /// ```
    #[cutile::entry(optimization_hints = (sm_120 = (occupancy = 2,),))]
    pub(super) unsafe fn linear_split_k<
        T: ElementType,
        const K: i32,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const SPLITS: i32,
    >(
        partials: *mut f32,
        x: &Tensor<T, { [-1, K] }>,
        w: &Tensor<T, { [-1, K] }>,
    ) {
        let pid = get_tile_block_id();
        let acc = project_tile::<T, K, BM, BN, BK, SPLITS>(x, w, pid.0, pid.1, pid.2)
            .reshape(shape![BM, BN]);
        let rows: Tile<i32, { [BM] }> = iota(shape![BM]);
        let rows = rows + (pid.0 * BM).broadcast(shape![BM]);
        let columns: Tile<i32, { [BN] }> = iota(shape![BN]);
        let columns = columns + (pid.1 * BN).broadcast(shape![BN]);
        let offsets = ((pid.2 * x.shape()[0]).broadcast(shape![BM]) + rows)
            * w.shape()[0].broadcast(shape![BM]);
        let offsets = offsets.reshape(shape![BM, 1]).broadcast(shape![BM, BN])
            + columns.reshape(shape![1, BN]).broadcast(shape![BM, BN]);
        let valid = lt_tile(rows, x.shape()[0].broadcast(shape![BM]))
            .reshape(shape![BM, 1])
            .broadcast(shape![BM, BN])
            & lt_tile(columns, w.shape()[0].broadcast(shape![BN]))
                .reshape(shape![1, BN])
                .broadcast(shape![BM, BN]);
        let base = pointer_to_tile(partials);
        let base = base.reshape(shape![1, 1]);
        let base = base.broadcast(shape![BM, BN]);
        let dst = base.offset_tile(offsets);
        store_ptr_tko(
            dst,
            acc,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(valid),
            None,
            Latency::<1>,
        );
    }

    /// Sum FP32 projection partials, then cast once; no intermediate BF16 rounding.
    #[cutile::entry()]
    pub(super) unsafe fn linear_merge_split_k<T: ElementType, const SPLITS: i32, const BN: i32>(
        out: &mut Tensor<T, { [1, BN] }>,
        partials: &Tensor<f32, { [-1, -1] }>,
        rows: i32,
        columns: i32,
    ) {
        let pid = get_tile_block_id();
        let split_ids: Tile<i32, { [SPLITS] }> = iota(shape![SPLITS]);
        let channels: Tile<i32, { [BN] }> = iota(shape![BN]);
        let channels = channels + (pid.1 * BN).broadcast(shape![BN]);
        let offsets = (split_ids * rows.broadcast(shape![SPLITS])
            + pid.0.broadcast(shape![SPLITS]))
            * columns.broadcast(shape![SPLITS]);
        let offsets = offsets
            .reshape(shape![SPLITS, 1])
            .broadcast(shape![SPLITS, BN])
            + channels
                .reshape(shape![1, BN])
                .broadcast(shape![SPLITS, BN]);
        let mask = lt_tile(channels, columns.broadcast(shape![BN]))
            .reshape(shape![1, BN])
            .broadcast(shape![SPLITS, BN]);
        let base = pointer_to_tile(partials.as_ptr());
        let base = base.reshape(shape![1, 1]);
        let base = base.broadcast(shape![SPLITS, BN]);
        let src = base.offset_tile(offsets);
        let (values, _token): (Tile<f32, { [SPLITS, BN] }>, Token) = load_ptr_tko(
            src,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None::<f32>,
            None,
            Latency::<1>,
        );
        let zero: Tile<f32, { [SPLITS, BN] }> = constant(0.0f32, shape![SPLITS, BN]);
        let values = select(mask, values, zero);
        let result: Tile<f32, { [BN] }> = reduce_sum(values, 0);
        let result: Tile<T, { [1, BN] }> = convert_tile(result.reshape(shape![1, BN]));
        out.store(result);
    }

    /// Storage uses T; MMA accumulates in FP32 and casts once to output O.
    /// PyTorch checkpoint weights are [out,in]; transpose each tile in registers.
    /// Optional projection bias is fused into the epilogue before the single store.
    /// BM token rows reuse weights; BN covers output channels independently of heads.
    #[cutile::entry(optimization_hints = (sm_120 = (occupancy = 2,),))]
    pub(in crate::runner::model) fn linear<
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
            project_tile::<T, K, BM, BN, BK, 1>(x, w, pid.0, pid.1, 0).reshape(shape![BM, BN]);
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
            project_tile::<T, K, BM, BN, BK, 1>(x, w, pid.0, pid.1, 0).reshape(shape![BM, BN]);
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

    /// Group nearby M tiles to reuse X and weights in L2; writes use mapped coordinates.
    /// https://developer.nvidia.com/blog/how-to-write-high-performance-matrix-multiply-in-nvidia-cuda-tile/
    /// Storage uses T; MMA accumulates in FP32 and casts once to output O.
    /// PyTorch checkpoint weights are [out,in]; transpose each tile in registers.
    /// Optional projection bias is fused into the epilogue before the single store.
    /// BM token rows reuse weights; BN covers output channels independently of heads.
    /// 4K B2/B4 down projection: ~2.8-4.3% kernel GPU-time reduction (Nsight).
    /// Swizzle policy overall: ~1.7-2.1% prefill e2e reduction; other linears vary.
    #[cutile::entry(optimization_hints = (sm_120 = (occupancy = 2,),))]
    pub(super) fn linear_swizzled<
        T: ElementType,
        O: ElementType,
        const K: i32,
        const HAS_BIAS: bool,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const MAP_SHAPE: [i32; 2],
    >(
        mut out: MappedPartitionMut<O, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<T, { [-1, K] }>,
        w: &Tensor<T, { [-1, K] }>,
        bias: &Tensor<T, { [-1] }>,
    ) {
        for index in out.iter_indices() {
            let (row, column) = index.components();
            let mut acc: Tile<f32, { [BM, BN] }> =
                project_tile::<T, K, BM, BN, BK, 1>(x, w, row, column, 0).reshape(shape![BM, BN]);
            if HAS_BIAS {
                let bias_tile: Tile<f32, { [BN] }> =
                    convert_tile(bias.partition(shape![BN]).load([column]));
                acc = acc + bias_tile.reshape(shape![1, BN]).broadcast(shape![BM, BN]);
            }
            let result: Tile<O, { [BM, BN] }> = convert_tile(acc);
            out.store(result, index);
        }
    }

    /// Fuse gate/up projections and SwiGLU; reuse each X tile and store only the final result.
    /// Mapped launches group nearby tiles; group size one also handles narrow projections.
    /// Round gate, up and SiLU to T to match the separate kernels.
    /// 4K B2/B4: ~3.0-3.2% combined-FFN GPU-time gain; ~1.1-1.8% incremental prefill e2e gain.
    /// https://developer.nvidia.com/blog/accelerating-mamba-2-with-high-performance-forward-pass-kernels-in-nvidia-cuda-tile/
    #[cutile::entry(optimization_hints = (sm_120 = (occupancy = 2,),))]
    pub(super) fn linear_gate_up_swiglu_swizzled<
        T: ElementType,
        const K: i32,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const MAP_SHAPE: [i32; 2],
    >(
        mut out: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<T, { [-1, K] }>,
        gate_w: &Tensor<T, { [-1, K] }>,
        up_w: &Tensor<T, { [-1, K] }>,
    ) {
        for index in out.iter_indices() {
            let (row, column) = index.components();
            let px = x.partition(shape![BM, BK]);
            let pg = gate_w.partition(shape![BN, BK]);
            let pu = up_w.partition(shape![BN, BK]);
            let mut gate: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
            let mut up: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
            for block in 0i32..((K + BK - 1) / BK) {
                let input = px.load([row, block]);
                gate = mma(input, pg.load([column, block]).transpose(), gate);
                up = mma(input, pu.load([column, block]).transpose(), up);
            }
            let gate: Tile<T, { [BM, BN] }> = convert_tile(gate);
            let gate: Tile<f32, { [BM, BN] }> = convert_tile(gate);
            let up: Tile<T, { [BM, BN] }> = convert_tile(up);
            let up: Tile<f32, { [BM, BN] }> = convert_tile(up);
            let one: Tile<f32, { [BM, BN] }> = constant(1.0f32, shape![BM, BN]);
            let zero: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
            let activated: Tile<T, { [BM, BN] }> =
                convert_tile(true_div(gate, one + exp(zero - gate)));
            let activated: Tile<f32, { [BM, BN] }> = convert_tile(activated);
            let result: Tile<T, { [BM, BN] }> = convert_tile(activated * up);
            out.store(result, index);
        }
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

    /// Graph replay gets context row/position from metadata, never capture-time CPU scalars.
    /// Each row writes only its uniquely reserved current KV slot; caller retains those states.
    #[cutile::entry()]
    pub(super) unsafe fn store_decode<T: ElementType>(
        cache_k: *mut T,
        cache_v: *mut T,
        table: &Tensor<i32, { [-1, -1] }>,
        metadata: &Tensor<i32, { [-1] }>,
        k: &Tensor<T, { [-1, -1] }>,
        v: &Tensor<T, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let row = metadata
            .partition(shape![1])
            .load([pid.0 * TOKEN_METADATA_WIDTH])
            .reshape(shape![]);
        let position = metadata
            .partition(shape![1])
            .load([pid.0 * TOKEN_METADATA_WIDTH + 1])
            .reshape(shape![]);
        let row: i32 = tile_to_scalar(row);
        let position: i32 = tile_to_scalar(position);
        let slot = table
            .partition(shape![1, 1])
            .load([row, position])
            .reshape(shape![]);
        let slot: i32 = tile_to_scalar(slot);
        let offset = slot * k.shape()[1] + pid.1 * HEAD_DIM;
        let columns: Tile<i32, { [HEAD_DIM] }> = iota(shape![HEAD_DIM]);
        let kp: PointerTile<*mut T, { [] }> = pointer_to_tile(cache_k);
        let kp = kp.offset(offset);
        let kp = kp.reshape(shape![1]);
        let kp = kp.broadcast(shape![HEAD_DIM]);
        let vp: PointerTile<*mut T, { [] }> = pointer_to_tile(cache_v);
        let vp = vp.offset(offset);
        let vp = vp.reshape(shape![1]);
        let vp = vp.broadcast(shape![HEAD_DIM]);
        let keys = k
            .partition(shape![1, HEAD_DIM])
            .load([pid.0, pid.1])
            .reshape(shape![HEAD_DIM]);
        let values = v
            .partition(shape![1, HEAD_DIM])
            .load([pid.0, pid.1])
            .reshape(shape![HEAD_DIM]);
        let kp: PointerTile<*mut T, { [HEAD_DIM] }> = kp.offset_tile(columns);
        let vp: PointerTile<*mut T, { [HEAD_DIM] }> = vp.offset_tile(columns);
        store_ptr_tko(
            kp,
            keys,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            Latency::<1>,
        );
        store_ptr_tko(
            vp,
            values,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            Latency::<1>,
        );
    }

    /// Stream one disjoint KV segment; empty segments return (0, -inf, 0).
    /// Keep raw FP32 (weighted V, maximum, denominator) until the final normalization.
    unsafe fn attention_segment<T: ElementType, const BK: i32, const SPLITS: i32>(
        q: &Tensor<T, { [-1, -1] }>,
        k: &Tensor<T, { [-1, -1] }>,
        v: &Tensor<T, { [-1, -1] }>,
        table: &Tensor<i32, { [-1, -1] }>,
        metadata: &Tensor<i32, { [-1] }>,
        qheads: i32,
        kvheads: i32,
        query_index: i32,
        query_head: i32,
        split: i32,
    ) -> (
        Tile<f32, { [1, 64] }>,
        Tile<f32, { [1, 1] }>,
        Tile<f32, { [1, 1] }>,
    ) {
        let row: i32 = tile_to_scalar(
            metadata
                .partition(shape![1])
                .load([query_index * TOKEN_METADATA_WIDTH])
                .reshape(shape![]),
        );
        let position: i32 = tile_to_scalar(
            metadata
                .partition(shape![1])
                .load([query_index * TOKEN_METADATA_WIDTH + 1])
                .reshape(shape![]),
        );
        let history_start: i32 = tile_to_scalar(
            metadata
                .partition(shape![1])
                .load([query_index * TOKEN_METADATA_WIDTH + 2])
                .reshape(shape![]),
        );
        let head = query_head / (qheads / kvheads);
        let query: Tile<f32, { [1, 64] }> =
            convert_tile(q.partition(shape![1, 64]).load([query_index, query_head]));
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
        let total_blocks = (position - history_start + BK) / BK;
        // Partition whole KV blocks evenly; adjacent splits meet without overlap.
        let first_block = total_blocks * split / SPLITS;
        let end_block = total_blocks * (split + 1) / SPLITS;
        for block in first_block..end_block {
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
        (acc, maximum, denominator)
    }

    /// One CTA per query/head, retaining the original short-history path.
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
        let (acc, _maximum, denominator) = attention_segment::<T, BK, 1>(
            q, k, v, table, metadata, qheads, kvheads, pid.0, pid.1, 0,
        );
        let acc = acc.reshape(shape![1, 64]);
        let denominator = denominator.reshape(shape![1, 1]);
        let result: Tile<T, { [1, 64] }> =
            convert_tile(true_div(acc, denominator.broadcast(shape![1, 64])));
        out.store(result);
    }

    /// Parallel KV segments, with one CTA per (query/head, split).
    /// https://github.com/NVIDIA/TileGym/blob/main/src/tilegym/ops/cutile/flash_decode.py
    /// partials[(query * qheads + head) * SPLITS + split, :] = [weighted_V[64], maximum, denominator].
    /// The 128-wide store masks its padding beyond the 66 valid packet elements.
    #[cutile::entry(optimization_hints = (sm_120 = (occupancy = 4,),))]
    pub(super) unsafe fn attention_split_kv<T: ElementType, const BK: i32, const SPLITS: i32>(
        partials: &mut Tensor<f32, { [1, 128] }>,
        q: &Tensor<T, { [-1, -1] }>,
        k: &Tensor<T, { [-1, -1] }>,
        v: &Tensor<T, { [-1, -1] }>,
        table: &Tensor<i32, { [-1, -1] }>,
        metadata: &Tensor<i32, { [-1] }>,
        qheads: i32,
        kvheads: i32,
    ) {
        let pid = get_tile_block_id();
        let (acc, maximum, denominator) = attention_segment::<T, BK, SPLITS>(
            q,
            k,
            v,
            table,
            metadata,
            qheads,
            kvheads,
            pid.0 / (qheads * SPLITS),
            (pid.0 / SPLITS) % qheads,
            pid.0 % SPLITS,
        );
        let acc = acc.reshape(shape![1, 64]);
        let maximum = maximum.reshape(shape![1, 1]);
        let denominator = denominator.reshape(shape![1, 1]);
        let columns: Tile<i32, { [64] }> = iota(shape![64]);
        let columns = columns.reshape(shape![1, 64]);
        let first_column: Tile<i32, { [1, 64] }> = constant(0i32, shape![1, 64]);
        let stats = select(
            eq_tile(columns, first_column),
            maximum.broadcast(shape![1, 64]),
            denominator.broadcast(shape![1, 64]),
        );
        let packet: Tile<f32, { [1, 128] }> = cat(acc, stats, 1);
        partials.store(packet);
    }

    /// Stable merge of raw FP32 split states; normalize and cast only once.
    /// https://github.com/NVIDIA/TileGym/blob/main/src/tilegym/ops/cutile/splitk_reduce.py
    /// ```text
    /// global_max = max(split_max)
    /// weights    = exp(split_max - global_max)
    /// output     = sum(weights * split_weighted_V) / sum(weights * split_denominator)
    /// ```
    #[cutile::entry()]
    pub(super) fn attention_merge_splits<T: ElementType, const SPLITS: i32>(
        out: &mut Tensor<T, { [1, 64] }>,
        partials: &Tensor<f32, { [-1, -1] }>,
        qheads: i32,
    ) {
        let pid = get_tile_block_id();
        let index = pid.0 * qheads + pid.1;
        let maxima = partials
            .partition(shape![SPLITS, 1])
            .load([index, 64])
            .reshape(shape![SPLITS]);
        let denominators = partials
            .partition(shape![SPLITS, 1])
            .load([index, 65])
            .reshape(shape![SPLITS]);
        let values = partials
            .partition(shape![SPLITS, 64])
            .load([index, 0])
            .reshape(shape![SPLITS, 64]);
        let maximum: Tile<f32, { [] }> = reduce_max(maxima, 0);
        let maximum = maximum.reshape(shape![1]).broadcast(shape![SPLITS]);
        // Split maxima use natural-log units; convert their difference before exp2.
        let log2_e: Tile<f32, { [SPLITS] }> = constant(LOG2_E, shape![SPLITS]);
        let weights = exp2((maxima - maximum) * log2_e, ftz::Enabled);
        let numerator: Tile<f32, { [64] }> = reduce_sum(
            values
                * weights
                    .reshape(shape![SPLITS, 1])
                    .broadcast(shape![SPLITS, 64]),
            0,
        );
        let numerator = numerator.reshape(shape![1, 64]);
        let denominator: Tile<f32, { [] }> = reduce_sum(weights * denominators, 0);
        let denominator = denominator.reshape(shape![1, 1]).broadcast(shape![1, 64]);
        let result: Tile<T, { [1, 64] }> = convert_tile(true_div(numerator, denominator));
        out.store(result);
    }

    /// Shared prefill/decode tile: QK MMA, online exp2 softmax, then probability/V MMA.
    /// Returns unnormalized FP32 values, base-two maxima and denominators for split merging.
    unsafe fn attention_tile<T: ElementType, const BQ: i32, const BK: i32>(
        query: Tile<T, { [BQ, HEAD_DIM] }>,
        query_positions: Tile<i32, { [BQ] }>,
        valid_q: Tile<bool, { [BQ] }>,
        k: &Tensor<T, { [-1, -1] }>,
        v: &Tensor<T, { [-1, -1] }>,
        table: &Tensor<i32, { [-1, -1] }>,
        context_row: i32,
        kv_head: i32,
        kvheads: i32,
        history_start: i32,
        visible_end: i32,
        first_block: i32,
        end_block: i32,
        unmasked_blocks: i32,
    ) -> (
        Tile<f32, { [BQ, HEAD_DIM] }>,
        Tile<f32, { [BQ, 1] }>,
        Tile<f32, { [BQ, 1] }>,
    ) {
        let head_columns: Tile<i32, { [HEAD_DIM] }> = iota(shape![HEAD_DIM]);
        let mut maximum: Tile<f32, { [BQ, 1] }> = constant(f32::NEG_INFINITY, shape![BQ, 1]);
        let mut denominator: Tile<f32, { [BQ, 1] }> = constant(0.0f32, shape![BQ, 1]);
        let mut acc: Tile<f32, { [BQ, HEAD_DIM] }> = constant(0.0f32, shape![BQ, HEAD_DIM]);
        let zero_scores: Tile<f32, { [BQ, BK] }> = constant(0.0f32, shape![BQ, BK]);
        let negative: Tile<f32, { [BQ, BK] }> = constant(f32::NEG_INFINITY, shape![BQ, BK]);
        let zero_kv_f32: Tile<f32, { [BK, HEAD_DIM] }> = constant(0.0f32, shape![BK, HEAD_DIM]);
        let zero_kv: Tile<T, { [BK, HEAD_DIM] }> = convert_tile(zero_kv_f32);
        let zero_rows: Tile<f32, { [BQ, 1] }> = constant(0.0f32, shape![BQ, 1]);
        let valid_rows = valid_q.reshape(shape![BQ, 1]);
        let valid_scores = valid_rows.broadcast(shape![BQ, BK]);
        let head_dim: f32 = convert_scalar(HEAD_DIM);
        // exp(x) = exp2(x * log2(e)); fold the conversion into the score scale.
        // ~2% prefill e2e improvement; ~8-9% attention-kernel estimate.
        let scale: Tile<f32, { [BQ, BK] }> =
            rsqrt(head_dim.broadcast(shape![BQ, BK]), ftz::Disabled)
                * constant(LOG2_E, shape![BQ, BK]);
        let causal_q = (query_positions + 1i32.broadcast(shape![BQ]))
            .reshape(shape![BQ, 1])
            .broadcast(shape![BQ, BK]);
        let key_indices: Tile<i32, { [BK] }> = iota(shape![BK]);
        let table_base = pointer_to_tile(table.as_ptr());
        let table_base = table_base.offset(context_row * table.shape()[1]);
        let table_base = table_base.reshape(shape![1]);
        let table_base = table_base.broadcast(shape![BK]);
        let k_base = pointer_to_tile(k.as_ptr());
        let k_base = k_base.reshape(shape![1, 1]);
        let k_base = k_base.broadcast(shape![BK, HEAD_DIM]);
        let v_base = pointer_to_tile(v.as_ptr());
        let v_base = v_base.reshape(shape![1, 1]);
        let v_base = v_base.broadcast(shape![BK, HEAD_DIM]);
        let kv_slot_stride = (kvheads * HEAD_DIM).broadcast(shape![BK]);
        let kv_head_offset = (kv_head * HEAD_DIM).broadcast(shape![BK]);
        let kv_columns = head_columns
            .reshape(shape![1, HEAD_DIM])
            .broadcast(shape![BK, HEAD_DIM]);
        let visible_keys = visible_end.broadcast(shape![BK]);
        // Reuse each key/value block across packed query rows; never materialize
        // the full attention score matrix.
        for block in first_block..end_block {
            let key_positions = key_indices + (history_start + block * BK).broadcast(shape![BK]);
            let valid_k = lt_tile(key_positions, visible_keys);
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
            let kv_offsets = slots * kv_slot_stride + kv_head_offset;
            let kv_offsets = kv_offsets
                .reshape(shape![BK, 1])
                .broadcast(shape![BK, HEAD_DIM])
                + kv_columns;
            let k_ptrs = k_base.offset_tile(kv_offsets);
            let v_ptrs = v_base.offset_tile(kv_offsets);
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
            // Make V available before QK/softmax to expose memory/compute overlap.
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
            // Block dot products.
            // ```text
            // scores = query * keys^T
            // ```
            let scores: Tile<f32, { [BQ, BK] }> = mma(query, keys.transpose(), zero_scores);
            // Earlier full blocks only need the padded-query guard; diagonal
            // blocks retain row-specific causal masking. Pointer guards stay intact.
            // ```text
            // block < unmasked_blocks: every key_position <= first_query_position
            // otherwise: scores = where(causal, scores * scale, -infinity)
            // ```
            // ~1% prefill e2e ( ~4-5% kernel only) improvement
            let scores = if block < unmasked_blocks {
                select(valid_scores, scores * scale, negative)
            } else {
                let causal = lt_tile(
                    key_positions
                        .reshape(shape![1, BK])
                        .broadcast(shape![BQ, BK]),
                    causal_q,
                ) & valid_scores;
                select(causal, scores * scale, negative)
            };
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
            // alpha         = exp2(maximum - next_maximum)
            // block_weights = exp2(scores - next_maximum)
            // ```
            let alpha = exp2(maximum - next_maximum, ftz::Enabled);
            let block_weights = exp2(
                scores - next_maximum.broadcast(shape![BQ, BK]),
                ftz::Enabled,
            );
            let block_sum: Tile<f32, { [BQ] }> = reduce_sum(block_weights, 1);
            // Accumulate the FP32 normalizer.
            // ```text
            // block_sum   = rowsum(block_weights)
            // denominator = alpha * denominator + block_sum
            // ```
            denominator = fma(
                denominator,
                alpha,
                block_sum.reshape(shape![BQ, 1]),
                rounding::NearestEven,
                ftz::Disabled,
            );
            let mma_weights: Tile<T, { [BQ, BK] }> = convert_tile(block_weights);
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
        (acc, maximum, denominator)
    }

    /// Pack Q heads sharing one KV head; reuse the prefill MMA/softmax tile for each split.
    /// ```text
    /// grid = [packed query row, KV head, split]
    /// Q [GROUP, HEAD_DIM] * K^T [HEAD_DIM, BK] -> scores [GROUP, BK]
    /// probabilities [GROUP, BK] * V [BK, HEAD_DIM] -> partial values [GROUP, HEAD_DIM]
    /// ```
    /// Padded Q rows never access output storage. Packets retain the scalar path's
    /// natural-log maximum so both paths share the same stable split merge.
    #[cutile::entry(optimization_hints = (sm_120 = (occupancy = 2,),))]
    pub(super) unsafe fn attention_decode_mma<
        T: ElementType,
        const BQ: i32,
        const BK: i32,
        const GROUP: i32,
        const SPLITS: i32,
        const PARTIAL_WIDTH: i32,
    >(
        partials: *mut f32,
        q: &Tensor<T, { [-1, -1] }>,
        k: &Tensor<T, { [-1, -1] }>,
        v: &Tensor<T, { [-1, -1] }>,
        table: &Tensor<i32, { [-1, -1] }>,
        metadata: &Tensor<i32, { [-1] }>,
        qheads: i32,
        kvheads: i32,
    ) {
        let pid = get_tile_block_id();
        let query_row = pid.0;
        let kv_head = pid.1;
        let split = pid.2;
        let row = metadata
            .partition(shape![1])
            .load([query_row * TOKEN_METADATA_WIDTH]);
        let context_row: i32 = tile_to_scalar(row.reshape(shape![]));
        let position = metadata
            .partition(shape![1])
            .load([query_row * TOKEN_METADATA_WIDTH + 1]);
        let position: i32 = tile_to_scalar(position.reshape(shape![]));
        let start = metadata
            .partition(shape![1])
            .load([query_row * TOKEN_METADATA_WIDTH + 2]);
        let history_start: i32 = tile_to_scalar(start.reshape(shape![]));
        let heads: Tile<i32, { [BQ] }> = iota(shape![BQ]);
        let valid_q = lt_tile(heads, GROUP.broadcast(shape![BQ]));
        let query_heads = heads + (kv_head * GROUP).broadcast(shape![BQ]);
        let columns: Tile<i32, { [HEAD_DIM] }> = iota(shape![HEAD_DIM]);
        let q_offsets = ((query_row * q.shape()[1]).broadcast(shape![BQ])
            + query_heads * HEAD_DIM.broadcast(shape![BQ]))
        .reshape(shape![BQ, 1])
        .broadcast(shape![BQ, HEAD_DIM])
            + columns
                .reshape(shape![1, HEAD_DIM])
                .broadcast(shape![BQ, HEAD_DIM]);
        let q_base = pointer_to_tile(q.as_ptr());
        let q_base = q_base.reshape(shape![1, 1]);
        let q_base = q_base.broadcast(shape![BQ, HEAD_DIM]);
        let q_mask = valid_q
            .reshape(shape![BQ, 1])
            .broadcast(shape![BQ, HEAD_DIM]);
        let q_ptrs = q_base.offset_tile(q_offsets);
        let (query, _token): (Tile<T, { [BQ, HEAD_DIM] }>, Token) = load_ptr_tko(
            q_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(q_mask),
            None::<T>,
            None,
            Latency::<1>,
        );
        let zero: Tile<f32, { [BQ, HEAD_DIM] }> = constant(0.0f32, shape![BQ, HEAD_DIM]);
        let zero: Tile<T, { [BQ, HEAD_DIM] }> = convert_tile(zero);
        let query = select(q_mask, query, zero);
        let positions = position.broadcast(shape![BQ]);
        let total_blocks = (position - history_start + BK) / BK;
        let (acc, maximum, denominator) = attention_tile::<T, BQ, BK>(
            query,
            positions,
            valid_q,
            k,
            v,
            table,
            context_row,
            kv_head,
            kvheads,
            history_start,
            position + 1,
            total_blocks * split / SPLITS,
            total_blocks * (split + 1) / SPLITS,
            (position - history_start + 1) / BK,
        );
        let acc = acc.reshape(shape![BQ, HEAD_DIM]);
        let maximum = maximum.reshape(shape![BQ, 1]);
        let denominator = denominator.reshape(shape![BQ, 1]);
        let log2_e: Tile<f32, { [BQ, 1] }> = constant(LOG2_E, shape![BQ, 1]);
        let maximum = true_div(maximum, log2_e);
        let columns: Tile<i32, { [128] }> = iota(shape![128]);
        let stat_columns: Tile<i32, { [HEAD_DIM] }> = iota(shape![HEAD_DIM]);
        let stat_columns = stat_columns
            .reshape(shape![1, HEAD_DIM])
            .broadcast(shape![BQ, HEAD_DIM]);
        let zero_columns: Tile<i32, { [BQ, HEAD_DIM] }> = constant(0i32, shape![BQ, HEAD_DIM]);
        let stats = select(
            eq_tile(stat_columns, zero_columns),
            maximum.broadcast(shape![BQ, HEAD_DIM]),
            denominator.broadcast(shape![BQ, HEAD_DIM]),
        );
        let packet: Tile<f32, { [BQ, 128] }> = cat(acc, stats, 1);
        let offsets = (((query_row * qheads).broadcast(shape![BQ]) + query_heads)
            * SPLITS.broadcast(shape![BQ])
            + split.broadcast(shape![BQ]))
            * PARTIAL_WIDTH.broadcast(shape![BQ]);
        let offsets = offsets.reshape(shape![BQ, 1]).broadcast(shape![BQ, 128])
            + columns.reshape(shape![1, 128]).broadcast(shape![BQ, 128]);
        let mask = valid_q.reshape(shape![BQ, 1]).broadcast(shape![BQ, 128])
            & lt_tile(columns, PARTIAL_WIDTH.broadcast(shape![128]))
                .reshape(shape![1, 128])
                .broadcast(shape![BQ, 128]);
        let base = pointer_to_tile(partials);
        let base = base.reshape(shape![1, 1]);
        let base = base.broadcast(shape![BQ, 128]);
        let out_ptrs = base.offset_tile(offsets);
        store_ptr_tko(
            out_ptrs,
            packet,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<1>,
        );
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
    /// Internally, scores and maxima are scaled by log2(e) for exp2 with FTZ.
    /// FTZ flushes subnormal exponential weights to zero.
    /// Performance comments estimate attention-kernel latency reduction as the
    /// prefill e2e reduction / 23% (historical B4 attention share), assuming all
    /// other costs stay fixed. These are rough estimates, not isolated measurements.
    ///
    /// Running state for each valid query row, over processed keys:
    /// ```text
    /// maximum     = max(scores)
    /// denominator = sum(exp2(scores - maximum))
    /// acc         = sum(exp2(scores - maximum) * values)
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
        // ~2-3% prefill e2e improvement; ~9-12% attention-kernel estimate.
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

        // ```text
        // Q [packed_tokens, 14 * 64]
        //                  columns
        //           0..63    64..127             832..895
        //         +---------+---------+-------+----------+
        // row 0 a | Q head 0| Q head 1|  ...  | Q head 13|
        // row 1 b | Q head 0| Q head 1|  ...  | Q head 13|
        // row 2 c | Q head 0| Q head 1|  ...  | Q head 13|
        //         +---------+---------+-------+----------+
        //                ^
        //       [query_rows, query_heads * HEAD_DIM + head_columns]
        //
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
        // cuTile 0.4 scalar zero padding fails GPU JIT broadcasting to this 2D load.
        // Omit padding here; the select below explicitly zeros masked query elements.
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
        let kv_head = (pid.1 * GROUP) / (qheads / kvheads);
        let end = ((query_tile + 1) * BQ + GROUP - 1) / GROUP;
        let end = if end < query_count { end } else { query_count };
        let visible_end = absolute_start + end;
        // Full KV blocks ending at/before the earliest query need no causal mask.
        // Blocks are relative to history_start, including cached/chunked histories.
        // ```text
        // history_start=0, BK=64, first_query_position=100
        //                         key_position ->
        //                  block 0          block 1          block 2
        //                  0......63        64.......127     128....191
        // query 100        [ visible ]      [visible|mask]    [ skipped ]
        // query 101        [ visible ]      [visible |mask]   [ skipped ]
        // query 102        [ visible ]      [visible  |mask]   [ skipped ]
        //                      ^                    ^
        //               no causal mask       diagonal: key = query
        //
        // unmasked_blocks = (100 - 0 + 1) / 64 = 1
        // block < 1: fully visible; block >= 1: check the causal boundary.
        // ```
        let first_query_position = absolute_start + query_tile * BQ / GROUP;
        let unmasked_blocks = (first_query_position - history_start + 1) / BK;
        let (acc, _maximum, denominator) = attention_tile::<T, BQ, BK>(
            query,
            query_positions,
            valid_q,
            k,
            v,
            table,
            context_row,
            kv_head,
            kvheads,
            history_start,
            visible_end,
            0,
            (visible_end - history_start + BK - 1) / BK,
            unmasked_blocks,
        );
        let acc = acc.reshape(shape![BQ, HEAD_DIM]);
        let denominator = denominator.reshape(shape![BQ, 1]);
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
    pub(in crate::runner::model) fn add<T: ElementType, const ROWS: i32>(
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
    pub(in crate::runner::model) fn swiglu<T: ElementType>(
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

    /// Two-stage greedy reduction; ties select the smallest valid vocabulary index.
    #[cutile::entry()]
    pub(super) fn argmax_blocks<const BLOCK: i32>(
        maxima: &mut Tensor<f32, { [1, 1] }>,
        indices: &mut Tensor<i32, { [1, 1] }>,
        logits: &Tensor<f32, { [-1, -1] }>,
        len: i32,
    ) {
        let pid = get_tile_block_id();
        let offsets: Tile<i32, { [BLOCK] }> = iota(shape![BLOCK]);
        let ids = offsets + (pid.1 * BLOCK).broadcast(shape![BLOCK]);
        let valid = lt_tile(ids, len.broadcast(shape![BLOCK]));
        let neg: Tile<f32, { [BLOCK] }> = constant(f32::NEG_INFINITY, shape![BLOCK]);
        let values = select(
            valid,
            logits
                .partition(shape![1, BLOCK])
                .load([pid.0, pid.1])
                .reshape(shape![BLOCK]),
            neg,
        );
        let maximum: Tile<f32, { [1] }> = reduce_max(values, 0);
        let big: Tile<i32, { [BLOCK] }> = constant(2147483647i32, shape![BLOCK]);
        let max_value: f32 = tile_to_scalar(maximum.reshape(shape![]));
        // Padding must not win ties when NaNs/-infinity leave the maximum at -infinity.
        let mask = valid & eq_tile(values, max_value.broadcast(shape![BLOCK]));
        let candidates = select(mask, ids, big);
        let index: Tile<i32, { [1] }> = reduce_min(candidates, 0);
        maxima.store(maximum.reshape(shape![1, 1]));
        indices.store(index.reshape(shape![1, 1]));
    }

    #[cutile::entry()]
    pub(super) fn argmax_final<const BLOCK: i32>(
        out: &mut Tensor<i32, { [1] }>,
        maxima: &Tensor<f32, { [-1, -1] }>,
        indices: &Tensor<i32, { [-1, -1] }>,
        count: i32,
    ) {
        let offsets: Tile<i32, { [BLOCK] }> = iota(shape![BLOCK]);
        let valid = lt_tile(offsets, count.broadcast(shape![BLOCK]));
        let neg: Tile<f32, { [BLOCK] }> = constant(f32::NEG_INFINITY, shape![BLOCK]);
        let values = select(
            valid,
            maxima
                .partition(shape![1, BLOCK])
                .load([program_id(0), 0])
                .reshape(shape![BLOCK]),
            neg,
        );
        let maximum: Tile<f32, { [1] }> = reduce_max(values, 0);
        let big: Tile<i32, { [BLOCK] }> = constant(2147483647i32, shape![BLOCK]);
        let max_value: f32 = tile_to_scalar(maximum.reshape(shape![]));
        let mask = valid & eq_tile(values, max_value.broadcast(shape![BLOCK]));
        let candidates = select(
            mask,
            indices
                .partition(shape![1, BLOCK])
                .load([program_id(0), 0])
                .reshape(shape![BLOCK]),
            big,
        );
        let index: Tile<i32, { [1] }> = reduce_min(candidates, 0);
        out.store(index.reshape(shape![1]));
    }
}
