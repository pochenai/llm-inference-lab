# Qwen3.5 BF16 performance iterations

Target: Qwen3.5-35B-A3B on one RTX 6000D, matching SGLang CUDA Graph mode.
Each successful optimization is validated, benchmarked and committed separately.
Full logits validation and host token feedback remain enabled for comparisons.
Only BF16 performance is a target. Prioritize the largest end-to-end costs using
Amdahl's law. Once SGLang is matched or beaten, stop pursuing items below 1% of
end-to-end latency. Before rewriting GEMMs, consult the Qwen2 profiling history
and check JIT launch hints and emitted tensor-core instructions.

## Baseline

Commit `70e8e87`: Rust graph decode 10.654 ms/token; SGLang graph 7.049
ms/token (three independent processes, B1, 32 prompt tokens, 8 output tokens,
2 warmups and 3 measured samples per process, CPU affinity 24-31).
Raw results: `rust/profiling/qwen3_5/cuda-graph-20261108/`.
Profiling investigation: `rust/profiling/qwen3_5/sglang-gap-20261108/`.
Profile timings must not be compared directly with native timings.

## Plan

- [x] Reduce graph replay resource registration and completion cleanup overhead.
- [x] Inspect emitted instructions and optimize routed expert GEMMs, which cost
  2.376 ms/decode step in the graph profile, before narrower projection roles.
- [x] Fuse shared expert gate/up/SwiGLU with consistent BF16 arithmetic.
- [x] Optimize scalar and MoE router projections while preserving GDN a/b MMA.
- [x] Pack GDN a/b weights and reuse one MMA projection, preserving accumulation.
- [ ] Re-profile GQA and host validation after the large GEMM changes; only
  optimize material end-to-end costs while decode parity remains outstanding.
- [ ] Overlap shared and routed expert work with explicit stream dependencies.
- [ ] Re-profile remaining gaps and repeat until native decode matches or beats
  the SGLang reference; check prefill and end-to-end latency too.

## Decisions and measurements

- Weight residency service skipped: startup through weights/pool initialization
  took 47.380 seconds for 69,321,574,912 weight bytes, with a warm filesystem
  cache. Forward JIT excluded. This is below the user's one-minute threshold.
  Raw measurement: `rust/profiling/qwen3_5/weights-load-20261108/result.json`.
- Replay investigation: each scratch view creates separate foreign Tensor
  storage even when backing memory is reused across layers. Verify a reduction
  in registrations without weakening graph checkout, output or submission
  lifetime protection.
- Qwen2 hardware precedent reviewed:
  `rust/profiling/qwen2/decode-b4-4k-20261101-110247/RESULTS.md`.
  Its BM=4 generic MMA compiled to SIMT; BM=16 emitted
  `HMMA.16816.F32.BF16`, confirmed by NCU/SASS, with unchanged grid and
  occupancy. Qwen3.5 dense projections already use BM=16; routed experts
  currently use explicit FP32 multiply/reduce. Check the actual specialization
  rather than assuming an MMA API call guarantees tensor cores.

## Iteration results

### 1. Replay storage reuse (done)

Scratch slots now lend their original unique Tensor owner and recover it at
last CPU use, preserving storage identity across layers. Escaped output views
hold the slot and exclusive checkout; the compute submission/capture explicitly
retains the checkout. Live slot reuse is rejected instead of making another alias.
Kernel arithmetic and full logits validation are unchanged.

- Validation: 14 Qwen3.5 release tests passed, including all ignored tiny-fixture
  tests, graph/eager comparisons, independent reference, real-geometry state
  operators, Runner and gated resource/escaped-output lifetime checks.
- `cargo check --offline -p engine --all-targets` passed.
- Native comparison complete: three independent processes per variant, rotated.
  Frozen baseline: `70e8e87`; candidate source diff and executable hashes are in
  `rust/profiling/qwen3_5/replay-storage-20261108/`.
- Final workspace metadata-restoration fast path: both targeted tests passed.

| Variant | Prefill ms | Decode ms/token | Total 8 outputs ms |
| --- | ---: | ---: | ---: |
| `70e8e87` | 55.550 | 10.857 | 131.595 |
| Replay storage reuse | 54.444 | 10.185 | 126.321 |

Decode -6.19%, total -4.01%; every paired process improved. Generated IDs match,
zero warm scratch allocations/recaptures, resident scratch unchanged. Timing
cells use median(process median(sample)); decode samples average seven steps.
Raw artifacts and report: `rust/profiling/qwen3_5/replay-storage-20261108/`.
Clocks remain unlocked; no new CPU subphase attribution claim is made.
Commit: `58b3fbe`.

### 2. Routed expert tensor cores (done)

Keep the original SIMT fallback for other dtypes. For BF16, try an explicit
16-row MMA tile per picked expert, retaining FP32 accumulation and casting only
at the same output boundary. Verify real gate/up (K2048/N1024) and down
(K512/N2048) geometry against independent exact CPU sums and the original
kernel, then inspect NCU/SASS before running the full-model comparison.

- Exact CPU and old-kernel comparison passed for both real geometries and
  non-monotonic/repeated expert IDs.
- NCU hardware counters are unavailable on this host (`ERR_NVGPUCTRPERM`).
  Instead, a diagnostic-only test build enabled cuTile 0.5's filesystem JIT
  store; checksum-validated cubin payloads were disassembled with `cuobjdump`.
  Both old specializations have zero HMMA instructions; both new ones contain
  `HMMA.16816.F32.BF16` (20/32 static instructions). Registers/thread: 255 -> 128;
  shared memory grows from 1,108 bytes to 33,956/44,244 bytes. These are static
  facts, not achieved utilization measurements. Test instrumentation was removed
  from the working source and does not affect the native benchmark binary.
- SASS, raw cubins, cache headers, instrumentation and NCU rejection logs:
  `rust/profiling/qwen3_5/expert-mma-20261108/compute/`.
- BF16 regression subset: 14 tests passed (the unchanged FP32 pooled control
  was excluded); tiny graph/eager logits match exactly and independent tiny
  reference max error is 2.38e-7. Real 35B BF16 reference passed: max error
  0.66619, RMS 0.12629, matching first ID, and all four chunking variants exactly
  match full prefill. Existing tolerances were not changed.
- Native three-process paired comparison completed; no observations excluded.

| Variant | Prefill ms | Decode ms/token | Total 8 outputs ms |
| --- | ---: | ---: | ---: |
| `58b3fbe` | 54.419 | 10.155 | 125.251 |
| Expert MMA | 31.775 | 9.564 | 98.790 |

Prefill -41.61%, decode -5.82%, e2e -21.13%; every paired process improves.
IDs match; warmed graph/scratch counts and bytes are unchanged. Raw results:
`rust/profiling/qwen3_5/expert-mma-20261108/`. E2e is now lower than the earlier
SGLang reference (119.244 ms), but decode still exceeds 7.049 ms/token. Continue
with the major decode gaps; confirm the final cross-engine result freshly.
Commit: `6ca209d`.

### 3. Shared gate/up fusion (done)

The SGLang B1 graph/eager mapping shows GEMV for shared gate/up; tensor cores
are not a universal requirement for narrow one-token projections. Try four
output channels per CTA over the full reduction, fuse both projections and
SwiGLU, and preserve BF16 rounding at gate, up, activation and final output.
The initial B1-only dispatch was rejected by real-model chunking validation.
Try consistent BF16 arithmetic across token counts before timing.
Validate against the unfused kernels at tiny/real geometry and a masked N tail,
then compare against `6ca209d` with the same native protocol.

- Operator test passed bitwise at K128/N128, K2048/N512 and K2048/N513.
- Fifteen BF16-focused regressions passed, with exact tiny graph/eager logits
  and chunking. Real-model B1-only vector validation failed: chunk=1 max
  difference 0.31143212 exceeds 0.02. No timing or commit for that candidate.
  Artifacts: `rust/profiling/qwen3_5/shared-gate-up-20261109/b1-only-*`.
- Consistent vector arithmetic across all BF16 token counts passed validation.
- Consistent variant passed the real 35B check: max error 0.56865, RMS 0.11685,
  matching first ID, and zero difference for all four chunking variants.
  One-process screen (not confirmation): prefill 32.900 ms, decode 8.986
  ms/token, total 95.720 ms. Prefill is about 1.1 ms slower than the preceding
  baseline, but seven decode steps save about 4 ms. The screen preceded the
  independent confirmation below. Artifacts:
  `rust/profiling/qwen3_5/shared-gate-up-20261109/consistent-vector/`.
- A separate CPU-only prototype for the existing full-logit finite scan found
  `.all` 0.189 ms versus non-short-circuit `.fold` 0.052 ms on 248,320 floats.
  This is a hot-buffer microbenchmark, not native inference evidence, and no
  production validation change has been made. Revisit after larger GPU costs.

Final confirmation: all 15 BF16 regressions and all-target check passed.

| Variant | Prefill ms | Decode ms/token | Total 8 outputs ms |
| --- | ---: | ---: | ---: |
| `6ca209d` | 31.630 | 9.489 | 97.979 |
| Consistent shared fusion | 33.009 | 8.995 | 96.035 |

Decode -5.21%, e2e -1.98%, prefill +4.36% (+1.380 ms). All paired processes
improve decode/e2e; no samples excluded. Scratch allocations 142 -> 126,
resident bytes 10,729,868 -> 10,594,684. IDs and ownership checks pass with
zero warm recaptures/allocations. The prefill tradeoff is accepted for this
workload; larger batch/context performance is unverified.
Raw confirmation: `rust/profiling/qwen3_5/shared-gate-up-20261109/consistent-vector/`.
Commit: `aa1fb74`.

### 4. Scalar and MoE router vector projections (done)

Reuse the shared fusion's full-K vector projection for BF16 outputs with
N <= 256 (GDN a/b, router and scalar shared gate). Keep the same arithmetic
across token counts; leave the competitive larger projections unchanged.
The independent CPU oracle covers N=1/32/33/256, three distinct input rows and
masked output tails. Then run tiny/real BF16 checks and native paired timing.

- Operator CPU oracle and all 16 BF16 regressions passed.
- Real 35B reference rejected the combined vector candidate: logit max 0.54415,
  RMS 0.10929 satisfy the error budget, but first ID changes to 16 (reference
  3636). The reference's top-two margin is 0.05652. Keep the existing first-ID
  gate; no performance sampling or commit for this candidate.
- A diagnostic-only test build loads weights once and independently toggles
  scalar, GDN a/b and router projection families to locate the change. Restore
  this temporary instrumentation before any production commit. If required,
  keep MMA accumulation and optimize through fusion/tile settings instead.

Precision screen isolated GDN a/b: baseline, scalar-only and router-only all
produce identical reference logits/ID 3636; a/b-only and all-vector produce
ID 16. Diagnostic source has been restored. Keep a/b MMA and select vector
only for N=1 and N=256. Revalidate the combined scalar/router path; consider
precision-preserving a/b MMA fusion as a separate iteration.
Screen artifacts: `rust/profiling/qwen3_5/skinny-projections-20261109/precision-screen/`.

The scalar+router candidate passed the full real reference: ID 3636, max
0.56865 / RMS 0.11685 (same as preceding baseline), all four chunks exactly
match full prefill. Diagnostic flags/tests have been removed. All 16 BF16 regressions and all-target check passed before three-process
native confirmation. Final candidate
artifacts: `rust/profiling/qwen3_5/router-vectors-20261109/`.

Final native confirmation:

| Variant | Prefill ms | Decode ms/token | Total 8 outputs ms |
| --- | ---: | ---: | ---: |
| `aa1fb74` | 33.044 | 9.015 | 96.310 |
| Scalar/router vector | 33.117 | 8.685 | 93.981 |

Decode -3.65%, e2e -2.42%; prefill +0.22% (+0.073 ms). Every paired process
improves decode/e2e, IDs match, scratch/graph counts remain fixed with no warm
allocation/recapture. All observations retained. Native confirmation and checks:
`rust/profiling/qwen3_5/router-vectors-20261109/`.
Commit: `d66bf6c`.

### 5. Packed GDN a/b MMA (done)

Concatenate a/b weights on the host during loading, upload one [2*VH,H] tensor,
and reuse the existing [16,64,64] MMA projection. Borrow a/b column views with
the packed row stride; force MMA so other configurations cannot accidentally
select the router vector bucket. Original file shapes are still validated and
total weight bytes remain unchanged. No new device kernel or extra weights.
The precision oracle compares all FP32 accumulator bits with the two separate
BF16-input MMA calls at real/tiny geometries and rows 1/3/32.

All 2316 FP32 accumulator bits match the old separate BF16-input MMA calls.
All 18 Qwen3.5 regressions and all-target checks pass. Real BF16 reference is
unchanged (max0.56865/RMS0.11685/ID3636; chunk1/2/3/4 all max0).

| Variant | Prefill ms | Decode ms/token | Total 8 outputs ms |
| --- | ---: | ---: | ---: |
| `d66bf6c` | 33.282 | 8.671 | 94.220 |
| Packed a/b | 33.018 | 8.406 | 91.975 |

Decode -3.05%, e2e -2.38%, prefill -0.79%; every paired process improves all
metrics. No warm allocations/recaptures, IDs match, KV releases. Weight bytes
unchanged69,321,574,912; scratch allocations126->122 and bytes10,594,684->10,594,680.
Raw confirmation: `rust/profiling/qwen3_5/packed-ab-20261109/`.
Commit: this update accompanies the packed a/b implementation.
Next: refresh profiling before choosing the largest remaining e2e cost.
