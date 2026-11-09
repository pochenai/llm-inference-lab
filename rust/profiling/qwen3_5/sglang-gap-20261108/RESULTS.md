# Why Qwen3.5 BF16 decode trails SGLang

The approved CUDA Graph iteration is committed as `70e8e87`. This investigation changes no inference code. The main differences are projection/expert kernel efficiency, parallel shared/routed MoE execution in SGLang, and measurable Rust replay/completion/validation overhead.

The native reference is BF16 Qwen3.5-35B-A3B, RTX 6000D, B1, 32 inputs and eight fixed outputs: Rust graph **10.654 ms/decode** versus SGLang graph **7.049 ms/decode**, a 1.512x ratio. The complete cohort is 131.103 versus 119.244 ms (1.099x); Rust's faster prefill offsets part of the decode gap. Native numbers come from the previous three-process comparison, not from instrumented trace latencies.

## Paired graph traces

A fresh trace for committed Rust and a SGLang graph trace each include one prefill and seven decodes after warmup. Both use CPU cores 24–31, BF16 weights/activations and FP32 recurrent state. SGLang uses the same stock MoE configuration and Triton attention as the native comparison. Profiler runs, native controls and compilation run sequentially. GPU clocks and external host load are not controlled.

The SGLang wrapper marks prefill/decode and starts/stops CUDA profiling around one measured sample. It does not hook model modules during graph replay. A **separate eager trace** with module hooks supplies role labels; no eager timing is substituted for graph timing. A warmup-only readback records SGLang's eight generated IDs as 9707, matching Rust's generation history. No additional token readbacks are added to the profiled SGLang sample. Cross-engine logits and selected-expert equality have not been checked.

| Per-decode trace metric | Rust graph | SGLang graph |
| --- | ---: | ---: |
| GPU kernel instances | 927 | 1,100 |
| Dense projection kernel sum, ms | 4.963 | 4.172 |
| Routed expert GEMM kernel sum, ms | 2.376 | 1.772 |
| GQA kernel sum, ms | 0.345 | 0.070 |
| Other kernel sum, ms | 0.983 | 1.210 |
| All kernel durations summed, ms | 8.667 | 7.224 |
| Union of kernel-active intervals, ms | 8.667 | 6.275 |
| Overlap counted in summed durations, ms | 0 | 0.949 |
| First-to-last kernel span, ms | 10.462 | 7.682 |

These are means over seven profiled decodes. Summed durations double-count parallel execution; they cannot be added to independent native timing differences. The kernel span also includes intervening transfers, scheduling and host submission gaps; its non-kernel portion is not automatically GPU idle time.

SGLang executes more kernels overall. Its advantage therefore cannot be explained by the total kernel count alone. See [the last-decode timeline](decode-timeline.svg) and `profile-analysis.json` for all seven steps.

## Projection gap is concentrated in small matrices

Rust uses the common `[BM, BN, BK] = [16,64,64]` MMA projection with masked/zero-padded input rows. B1 supplies one valid row. SGLang selects cuBLAS GEMV paths and packs selected projections at load time. The role breakdown below uses **graph GPU time** in both columns.

| Projection role, summed over model layers | Rust ms/step | SGLang ms/step |
| --- | ---: | ---: |
| Shared expert gate + up | 0.584 | 0.276 |
| DeltaNet a + b | 0.425 | 0.151 |
| Router | 0.289 | 0.163 |
| Full-attention Q + K + V | 0.437 | 0.316 |
| DeltaNet QKV | 0.841 | 0.862 |
| DeltaNet z | 0.461 | 0.460 |
| LM head | 0.764 | 0.761 |

The first three rows account for about 0.707 ms of the 0.791 ms aggregate dense-projection difference. Large QKV, z and vocabulary projections are already competitive. Replacing all linears or starting with the LM head would miss the main projection opportunities.

Rust executes 391 mathematical projections per decode. SGLang executes 331: packed GQA QKV removes 20 separate projections, and packed shared-expert gate/up removes 40. SGLang additionally has 40 split-K reduction kernels, giving 371 dense kernels. Those packing choices coexist with a larger overall kernel count.

Role assignment is checked, not guessed from short kernel names: the eager trace attributes launches to the innermost module via correlation ID, API timestamp and thread. All eleven dense symbol/grid/block keys and all 371 occurrences per step match the graph trace. Within-key projection order is preserved on the main stream; alternate-stream router projections have distinct keys. Graph timings are then grouped using these labels. The single unhooked vocabulary GEMV is identified from the loaded LM-head weight shape and its unique large output grid. Rust roles follow captured computation order, with every output grid checked against the expected projection width. The scripts stop if these assumptions fail.

## Expert GEMM and attention

Rust routed experts use BF16 weights/inputs converted to FP32 for explicit multiplication and reduction. SGLang's unquantized FusedMoE path uses grouped Triton `tl.dot` kernels. The two expert GEMMs cost 2.376 versus 1.772 ms per decode. This is a concrete candidate for grouped-MMA work; instruction throughput, bandwidth and Tensor Core utilization have not been measured with NCU, so the source-level difference is not a hardware-counter diagnosis.

Rust GQA scans causal history one position at a time per query head. SGLang uses grouped stage-1 plus stage-2 kernels. At this short history, the 0.345 versus 0.070 ms difference is measurable but smaller in absolute terms than projections, expert GEMMs or replay preparation. No long-context conclusion is drawn here.

## MoE overlap: native controlled experiment

Installed SGLang's `Qwen2MoeSparseMoeBlock.forward_normal_dual_stream` runs shared experts on the current stream and routed experts on an alternate stream during graph capture, then joins them. The graph trace confirms concurrent work on the two captured streams. Q/K normalization also uses the alternate stream in some attention layers.

A diagnostic process-local constructor wrapper sets **only each MoE block's `alt_stream` to `None` before capture**. It does not modify installed files, model weights, kernels, precision or attention's alternate stream. This is an experimental control, not a stock CLI tuning option. Native measurements contain no profiler, module hooks or extra token readbacks. Three independent processes/variant rotate order, with two comparison warmups and three measured samples/process.

| Native control | Process decode medians, ms | Median, ms |
| --- | --- | ---: |
| Stock parallel MoE | 7.021, 7.064, 7.061 | 7.061 |
| MoE alternate stream disabled | 7.966, 7.722, 7.714 | 7.722 |

Disabling the dual-stream MoE implementation makes SGLang **9.4% slower**, a difference of **0.661 ms** between variant medians. The 0.949 ms overlap in the trace is not itself the net wall-time benefit: contention and the branch's clone/synchronization operations also change. Rust currently serializes the shared and routed branches. Copying this scheduling idea into Rust would still require its own correctness/ownership and native-performance checks.

## Rust CPU and completion costs

CUDA API timestamps within `qwen3_5.decode.graph_replay` show the following:

- After the last input metadata upload returns, the interval before the actual `cuGraphLaunch` call is **0.78–1.16 ms**, median **0.855 ms**. The source in that interval calls `GraphLaunch::execute`, which binds/retains the graph, reacquires recorded storage accesses and drains pending updates before the driver call. The exact internal distribution has not been sampled.
- The driver graph-launch call itself has a median around **0.328 ms** under profiling.
- D2H copy duration is about **0.035 ms/step** for 993,280 logit bytes plus one four-byte token.
- After those GPU copies finish, CPU validation begins **0.50–0.76 ms** later, median **0.536 ms**. This interval includes GPU-completion handling, submission cleanup, commit and wakeup; a finer causal split needs instrumentation or CPU sampling.
- The full-logit finite-value scan costs median **0.312 ms/step**. SGLang's benchmark does not perform this scan or copy the full logits back.

The replay preparation and completion intervals are optimization targets, not measured net benefits relative to SGLang: SGLang has its own batch/input preparation overhead. Similarly, removing checks is not the same optimization as reducing their cost.

Code inspection identifies a useful lead: cuda-async deduplicates captured accesses by `Arc<Storage>` identity. Scratch checkout currently creates a fresh foreign Tensor/storage wrapper per view, even when layers reuse the same physical backing buffer. These identities can keep repeated view accesses distinct at replay. Captured-resource counts and a Rust-side ablation are still needed to quantify that contribution. Any change must preserve actual session/KV leases, graph exclusivity and completion ownership; bypassing resource tracking without equivalent ownership proof would be unsafe.

## Suggested iteration order

1. Measure and reduce replay access registration/retirement overhead while preserving numerical kernels and ownership checks. Reuse existing checked capture/recorder patterns where suitable; retain Gate-based completion tests. This gives a focused first experiment with no intended mathematical change.
2. Target small B1 projections: shared gate/up packing, a/b packing or dedicated GEMV, and router shape handling. Keep large QKV/z/LM head as controls.
3. Evaluate shared/routed MoE overlap and grouped expert MMA separately, retaining the user's review checkpoint for improvements above 10%.
4. Revisit grouped attention with a wider sequence-length matrix later.

No inference optimization has been applied in this investigation. The original graph commit is clean; the scripts, reports and traces live under the ignored profiling directory.

## Artifacts and limits

- `commands.json`, `metadata.json`, `rust-graph`, source snapshots/hashes in `sources/`, `run.log`, three `.nsys-rep`/`.sqlite` pairs and per-trace stats retain the profile evidence.
- `sglang_profile.py` marks one warmed cohort; `sglang_serial_moe.py`, `overlap_control.py`, `overlap-commands.json`, six raw control records and `overlap-run.log` retain the native ablation.
- `analyze.py` validates phase counts, interval unions, API boundaries and role mappings, then writes `profile-analysis.json`. `plot.py` produces standalone SVG/PNG timelines.
- Profiling latencies are not native benchmark latencies. The native reference and the fresh scheduling ablation are separate datasets. This is BF16 B1 short-history analysis; it does not establish serving throughput, long-context scaling or cross-engine logit equivalence. No NCU hardware-counter pass was run.
