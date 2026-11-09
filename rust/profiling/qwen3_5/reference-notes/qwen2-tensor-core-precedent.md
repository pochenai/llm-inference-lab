# Qwen B4 / 4096-history decode optimization

Baseline commit: `7bf4c54`. RTX 5060 Ti 8 GiB, native BF16 weights/activation/KV, optimized Rust Release, CUDA Graph decode and eager prefill. No library, dependency, public Model/KV API or scratch Tensor change. Artifact files are ignored by Git.

## Selected implementation

- Reuse `[M,N,K]=[16,32,64]` for packed BF16 QKV at M=2..15 and input/query widths>=512. Attach selection to the existing private `LinearTile` type, separately from square O projections.
- Extend the existing multi-row FFN bucket to compression as well as expansion: one dimension>=2048 and the larger/smaller ratio in 2..8. This selects the same tile for down; bounded split-K and its existing partial buffer/merge remain shared.
- Retain B1, FP32, vocabulary and square O fallback policies, and benchmark prefill rows>=16. Grid dimensions and JIT generics derive from the selected tile. No new kernels or workspace allocations.

## Native B4/4K comparison

All modes use fresh KV,16640 slots, fixed64 outputs (one prefill+63 decode steps), GPU feedback and no profiler. Two warmups and five measured samples/process. Six independent processes/mode with balanced rotated mode order; final decode cells are median(process median(sample median(step))). Prefill and total cells are median(process mean(sample)). Final Rust D2H/text decoding is outside timing; no serving/overlap throughput claim.

An initial selected-process repetition had simultaneous prefill/decode slowdown and then recovered. A fixed three extra repetitions were added for ALL modes. Every observation is retained; no outlier was excluded. GPU clocks are unlocked.

| Engine | Decode ms/step | Process range | Prefill ms | Total64 ms |
| --- | ---: | --- | ---: | ---: |
| baseline | 4.382 | 4.371-4.434 | 344.152 | 627.715 |
| tuned | 4.159 | 4.134-4.723 | 343.111 | 610.088 |
| sglang | 4.249 | 4.175-4.318 | 374.883 | 648.936 |

Own decode latency reduction: 5.1%. Selected Rust vs SGLang: -2.1% (positive means slower).

## Bounded controls

One process/mode/case, five samples each; these are screens, not repeated performance-equivalence evidence. B1 dispatch is structurally unchanged.

| Input | Batch | Baseline decode ms | Selected decode ms | Baseline prefill ms | Selected prefill ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1 | 3.917 | 3.835 | 5.757 | 5.057 |
| 4096 | 1 | 4.120 | 4.095 | 91.243 | 91.085 |
| 32 | 2 | 3.663 | 3.452 | 6.406 | 6.507 |
| 4096 | 2 | 3.953 | 3.779 | 171.010 | 172.732 |
| 32 | 4 | 3.706 | 3.589 | 8.452 | 8.525 |

## Separate Nsight Systems evidence

Fresh baseline/SGLang node traces were collected before tuning. Independent selected-candidate traces preserve one-change attribution. Values below sum24 layers at decode step33 (visible history4129); profile sums are not native wall-time components.

| Candidate | QKV ms | O ms | Gate/up ms | Down+merge ms | Attention+merge ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.300 | 0.171 | 1.144 | 0.780 | 0.685 |
| qkv-k64 | 0.197 | 0.171 | 1.136 | 0.769 | 0.682 |
| down-k64 | 0.296 | 0.170 | 1.138 | 0.659 | 0.680 |
| o-k64 | 0.296 | 0.124 | 1.133 | 0.770 | 0.676 |

QKV alone reduces its GPU role by about34%; down+merge alone by about15%. O GPU time also fell in its profile, but the one-process native screen did not show a gain; it was not selected this round. This does not establish that O can never benefit. N=16 QKV (moreCTAs) was rejected: native~5.10ms versus~4.40ms for N=32 in that isolated screen. K=128 candidates were tested but had no robust advantage over K=64. All CPU oracle/screen/profile artifacts remain available.

## Nsight Compute and generated SASS

One matching first-layer QKV kernel per binary, after warmup; kernel replay/counter timings are distinct from warmed native latency and the24-layer Systems sums. Basic hardware counters and full SASS are preserved in `compute/`.

| Metric | Baseline | QKV selected |
| --- | ---: | ---: |
| Kernel us | 15.90 | 10.14 |
| Achieved occupancy % | 10.83 | 10.70 |
| DRAM throughput % of peak | 31.21 | 47.91 |
| Tensor pipe activity % | 0.00 | 6.92 |
| Waves per SM | 0.50 | 0.50 |

The baseline SASS has no HMMA; selected SASS contains `HMMA.16816.F32.BF16`. The M=4 generic `mma` path was lowered to SIMT here; the selected M=16 path uses Tensor Cores. Grid remains36CTAs×256threads, registers/thread128, achieved occupancy stays near11%. Thus the gain is not explained by higher occupancy. This is evidence for these specializations on this device, not a rule that every generic mma invocation uses Tensor Cores.

## Validation and numerical limits

- Independent CPU oracles cover all three packed-QKV outputs, partial M/N/K, and selected down split-K2/4/8/16 including empty partitions, with NaN sentinels. Rejected candidates were checked before timing too.
- Full engine suite:101passed,4opt-ins skipped;3compile-fail doctests passed. All-target Release check, formatting and diff checks passed.
- Explicit real BF16 checkpoint eager/Graph test:384batch-step comparisons/896per-request predictions over B1/B2/B4,repeated/ramp4K histories,64steps; logits/tokens exactly equal within selected implementation, KV release restores capacity.
- 140 measured native samples validated: frozen source/binary identities, captures fixed at2,63decode forwards, stable Rust IDs within mode and KV release.
- Old/new free-generation final IDs equal in main case: False. Raw values are in final-matrix/validation.json. Different BF16 reduction/MMA choices can change logits and token trajectories; this is NOT asserted to be only near-tie behavior.
- No held-out text-quality or serving-workload evaluation. Hardware evidence above is limited to the profiled QKV specialization; no universal achieved-bandwidth claim or general stall attribution.

## Reproduction

Use the exact commands in stage-specific JSON files; do not overlap GPU jobs or compile during native timings. `screen_projections.py`,`screen_n16.py`,`screen_o.py` freeze isolated binaries; `profile_baseline.py`,`profile_sglang.py`,`profile_candidates.py`,`collect_compute.py` collect separate traces. `final_compare.py` runs initial native/controls, `extra_compare.py` adds the predeclared three main-case repetitions, and `report.py` validates/summarizes all retained samples. Stage output directories must be fresh; previous artifacts are never overwritten.

## Final combined Graph trace

The final frozen combined binary has63replays and389nodes/replay. Cells below are medians of the GPU role sums over ALL63decode steps; this aggregation differs from the matched single-step isolated table above. Profile times are separate from native wall-time measurements.

| QKV ms | O ms | Gate/up ms | Down+merge ms | LM head ms | Attention+merge ms | Kernel sum ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.200 | 0.171 | 1.145 | 0.658 | 0.662 | 0.689 | 3.782 |

Step33 had attention+merge0.917ms versus the all-step median0.689ms. That observation is retained, together with all63step summaries, in `combined-attribution.json`; no step is excluded. The baseline all-step attention median is0.685ms. The earlier fresh SGLang single-step trace remains in `baseline-attribution.json`; native latency advantages do not imply every GPU role is faster.
