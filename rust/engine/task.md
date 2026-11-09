# Qwen3.5 BF16 performance iterations

## Machine handoff checkpoint — 2026-11-09 (Asia/Shanghai)

PAUSED at the user's request: the source machine no longer has a GPU. No GPU
experiment is active. Resume only after the replacement machine is ready.
All production optimizations are committed through `827f7f9`; the rejected
expert tile16 experiment is fully reverted. The remaining changes at this
checkpoint are handoff documentation, ignore rules and archived evidence.

### Result and scope

Native BF16 Qwen3.5-35B-A3B, one RTX6000D (156SM), B1,32 repeated token9707
inputs,8 fixed outputs. Decode uses CUDA Graphs in both Rust/SGLang; prefill is
eager. Each Rust optimization has3 rotated independent processes/variant,
2 warmups+3 measured samples. CPU affinity24-31 was local to the OLD GPU.
Rust retains full-logit D2H/finite checks and host token feedback. The SGLang
adapter uses GPU sampling; these boundaries must remain explicit.

| Historical snapshot | Prefill ms | Decode ms/token | E2E8 outputs ms |
| --- | ---: | ---: | ---: |
| Rust graph starting point `70e8e87` | 55.362 | 10.654 | 131.103 |
| Latest Rust `827f7f9` | 32.499 | 7.705 | 86.433 |
| Earlier SGLang0.5.9 graph reference | 69.120 | 7.049 | 119.244 |

E2e is below the earlier SGLang reference; decode is still about9.3% slower.
The parity objective is NOT complete. These snapshots are from different rounds;
unlocked clocks/host load mean cross-round differences are not isolated gains.
Use each iteration's paired table below for its claimed improvement. A fresh
same-period SGLang comparison has NOT been performed after the last changes.
No large-batch/long-context/serving-throughput or broad text-quality claim.

### Completed and committed

| Commit | Completed optimization | Evidence directory under `rust/profiling/qwen3_5/` |
| --- | --- | --- |
| `58b3fbe` | Reuse scratch Tensor storage identities for graph resource deduplication | `replay-storage-20261108/` |
| `6ca209d` | BF16 routed-expert MMA/Tensor Cores, confirmed HMMA SASS | `expert-mma-20261108/` |
| `aa1fb74` | Shared gate/up/SwiGLU fusion with consistent reduction across token counts | `shared-gate-up-20261109/consistent-vector/` |
| `d66bf6c` | Scalar/MoE router vector projections; keep GDN a/b MMA | `router-vectors-20261109/` |
| `0c5dcc0` | Pack GDN a/b weights and reuse one unchanged MMA projection | `packed-ab-20261109/` |
| `06aa36e` | Factory-only immutable weight ownership for captured graphs | `frozen-weights-20261109/` |
| `a0cd88f` | 32-token blocked GQA with causal/slot/tail masks | `blocked-gqa-20261109/` |
| `827f7f9` | Vectorizable complete CPU finite scan; preserve all validation | `finite-scan-20261109/` |

GPU changes passed the21-test Qwen3.5 suite (including opt-in tiny/ownership/
context checks) and real35B BF16 reference/chunking gates. The final CPU-only
scan change passed its new contract cases, a model reference/chunking smoke
check and all-target check. No numerical threshold was widened. Last real GPU
reference: ID3636, max0.48618/RMS0.10955, chunk1/2/3/4 all max0. FP32 performance
was deliberately excluded; existing FP32 correctness controls were used when
shared layouts changed.

### Rejected or deliberately skipped

- B1-only shared-vector dispatch: real chunking max0.31143 exceeded0.02. Rejected;
  the committed consistent-all-token-count version passes the original gate.
- Vector GDN a/b: changed reference ID3636->16. Isolated in a single-weight-load
  diagnostic; committed scalar/router optimization excludes a/b.
- Expert output tile16: numerical gates pass, but the one-process screen regresses
  to69.405ms prefill/8.331ms decode/127.722ms E2E. Reverted; no optimization commit.
- Resident weight service: skipped by the user's threshold; warm-filesystem startup
  through weights/pool load was47.380s, below1min (forward JIT excluded).
- NCU hardware counters: unavailable on the old host (`ERR_NVGPUCTRPERM`). HMMA
  proof uses checksum-validated JIT cubins and SASS, not achieved utilization.
- Memcheck: round-robin async-free warnings also occur with the old GQA oracle.
  Explicit stream completion in the operator oracle yields0 errors. No production
  fence was added; keep all old/new/fenced logs and separate Gate lifetime evidence.

### Not completed; resume in this order

- [ ] Rebuild and establish fresh Rust/SGLang baselines on the NEW hardware.
- [ ] Integrate shared/routed MoE CUDA Graph branch overlap. Only the small CUDA
  dependency-insertion prototype is implemented/tested; production inference
  remains serial. Probe source/log: `moe-parallel-20261109/probe/`.
- [ ] Preserve module, scratch checkout, session/KV and escaped-output ownership;
  add Gate-based ordering/lifetime tests before accepting parallel execution.
- [ ] Re-profile remaining e2e costs and repeat paired measurements/commits until
  decode matches or beats SGLang. After parity, skip targets below1%e2e.
- [ ] Run final same-period cross-engine comparison and record exact boundaries.

Design notes are proposals, not completed code: `parallel-design.md` and
`frozen-weights-plan.md` (the latter records the design already implemented in
`weights.rs`). The validated CUDA13.2 child-node probe uses
`cuStreamGetCaptureInfo_v3`, `cuGraphAddChildGraphNode` and
`cuStreamUpdateCaptureDependencies_v2`; it does not access SDK private fields.

### Moving to the replacement machine

The Qwen3.5 evidence directory is now included in Git: reports, raw JSON/logs,
commands, source snapshots/diffs, plots, small GPU-code evidence and `.nsys-rep`
traces. Derived SQLite exports/sidecars remain ignored and can be regenerated
without a GPU using the installed Nsight Systems version:

```bash
nsys export --type=sqlite --output=/path/to/report.sqlite /path/to/report.nsys-rep
```

Repeated host executables (~609MiB), temporary `jit-store/`,
model weights, virtual environments and build directories stay ignored.
Rebuild executables from commits; their archived hashes identify old runs.
Push/copy the Git commits separately; this session has not published to a remote.

Small reference logits are archived in
`rust/profiling/qwen3_5/validation-references/` with SHA256 manifest. The full
35B model is NOT in Git. Tiny model weights must be copied or regenerated with
`rust/engine/examples/qwen3_5_reference.py` / the example guide. Do not copy an
old Python venv blindly; recreate SGLang0.5.9 and its recorded dependency setup.
The relevant Qwen2 Tensor Core precedent is copied to
`rust/profiling/qwen3_5/reference-notes/qwen2-tensor-core-precedent.md`.
Historical scripts contain OLD absolute paths; adjust paths for reruns and keep
old results immutable. Avoid copying secret environment files.

From the repository root, after installing a supported CUDA/Rust environment
and the locked cuTile/cuda-core/cuda-async0.5.0 dependencies:

```bash
env -u LD_LIBRARY_PATH CUDA_HOME=/path/to/cuda cargo check --manifest-path rust/Cargo.toml -p engine --all-targets
env -u LD_LIBRARY_PATH CUDA_HOME=/path/to/cuda QWEN35_TEST_MODEL=/absolute/path/to/tiny QWEN35_TEST_REFERENCE=/absolute/path/to/repo/rust/profiling/qwen3_5/validation-references/tiny-reference.f32 cargo test --release --manifest-path rust/Cargo.toml -p engine qwen3_5 -- --include-ignored --test-threads=1
env -u LD_LIBRARY_PATH CUDA_HOME=/path/to/cuda QWEN35_TEST_MODEL=/absolute/path/to/Qwen3.5-35B-A3B QWEN35_TEST_REFERENCE=/absolute/path/to/repo/rust/profiling/qwen3_5/validation-references/real-reference.f32 cargo test --release --manifest-path rust/Cargo.toml -p engine reference_chunking_and_state_validation -- --ignored --nocapture --test-threads=1
env -u LD_LIBRARY_PATH CUDA_HOME=/path/to/cuda cargo bench --manifest-path rust/Cargo.toml -p engine --bench qwen3_5 -- --model-dir /absolute/path/to/Qwen3.5-35B-A3B --tokens 32 --output-tokens 8 --warmup 2 --iterations 3 --cuda-graph
```

Use `--offline` only if dependency caches were transferred. Re-evaluate GPU
architecture support, SM count, tile choices, CUDA paths and NUMA CPU affinity;
24-31 and SM120 hints are old-host settings. Run GPU jobs sequentially, and keep
profiling separate from native timing. SGLang commands/config are preserved in
`cuda-graph-20261108/commands.json` and `sglang-gap-20261108/`.

---

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
- [x] Remove remaining immutable-weight replay bookkeeping with typed ownership.
- [x] Re-profile GQA and host validation after the large GEMM changes; only
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
  occupancy. At the initial profile, Qwen3.5 dense projections used BM=16
  while routed experts still used explicit FP32 multiply/reduce; the committed
  expert optimization has since replaced that BF16 path with MMA. Check the actual specialization
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
Commit: `0c5dcc0`.
Next: refresh profiling before choosing the largest remaining e2e cost.

### 6. Frozen capture-owned weights (done)

Refreshed profile: replay prelaunch CPU median0.290ms, post-readback tail0.226ms,
full finite scan0.312ms, GQA GPU0.341ms. 817kernels/decode; experts1.770ms now
match SGLang's earlier1.772ms sum. Profile durations are not native latency.
Artifacts: `rust/profiling/qwen3_5/post-packed-profile-20261109/`.

A private immutable weight type has one factory: upload fresh data and await
completion. It exposes no tensor/buffer/mutable accessor and cannot inherit a
pre-existing captured writer. Eager input retention delegates to normal cuTile
tracking. Capture validates actual context and retains the tensor for the
whole graph lifetime, eliminating repeated weight read leases; all mutable
state/scratch/KV tracking remains unchanged. No GPU math/ABI changes.

Two focused tests pass (default completion and a forced notification run):
Gate proves frozen weights survive dropping CPU weight+graph owners until GPU
completion, then release; a separate valid CUDA context on ordinal0 is rejected
for eager and capture. A rescue allocation/drain guard makes failed assertions
safe. Full20-test regression, real BF16 check and native comparison follow.

Full20-test regression and real BF16 reference passed unchanged (max0.56865,
RMS0.11685,ID3636, all chunk max0), plus all-target check.

| Variant | Prefill ms | Decode ms/token | Total8 outputs ms |
| --- | ---: | ---: | ---: |
| `0c5dcc0` | 33.249 | 8.504 | 92.600 |
| Frozen weights | 33.368 | 8.111 | 90.146 |

Decode -4.62%, e2e -2.65%, prefill +0.36%. Every paired process improves
 decode/e2e; IDs match and weight/scratch/graph accounting is unchanged, with
zero warm allocation/recapture. No samples excluded. Raw results:
`rust/profiling/qwen3_5/frozen-weights-20261109/`.
Commit: `06aa36e`.

Fresh trace confirms ownership attribution:817 kernels/decode and~7.10ms GPU
sum unchanged, prelaunch CPU0.290->0.035ms, post-readback tail0.226->0.093ms;
finite scan stays0.312ms. These are profile values, not native latency cells.
Raw trace: `rust/profiling/qwen3_5/post-frozen-profile-20261109/`.

### 7. Blocked GQA (done)

Replace scalar-position KV walking with32-token tiles and stable block softmax,
using the same implementation for eager/graph and all token counts. Preserve
causal masks, physical slot indirection, FP32 accumulation and BF16 y/sigmoid
rounding boundaries. CPU oracle covers grouped heads, fragmented slots, pooled
and standalone addressing,31/32/63/64 boundaries, invalid/unmapped NaN tails,
and graph/eager agreement. Existing model thresholds remain unchanged.

Operator oracle passed all pooled/standalone cases and exact pooled graph/eager
comparisons. Compute Sanitizer initially reported potential async-free ordering
warnings with the round-robin global policy; the OLD kernel under the same oracle
also reports them. Adding an explicit stream-completion fence to the oracle
(host-polled SDK completion is not an explicit CUDA timeline fence for the tool)
gives0 errors with the blocked kernel. No production synchronization was added.
All logs preserved under `rust/profiling/qwen3_5/blocked-gqa-20261109/memcheck/`.
Full21-test regression and real-model gate precede native paired timing.

All21 regressions/all-target check pass. Real BF16 reference ID3636, max0.48618,
RMS0.10955; all four chunk variants max0. No threshold changes.

| Variant | Prefill ms | Decode ms/token | Total8 outputs ms |
| --- | ---: | ---: | ---: |
| `06aa36e` | 33.242 | 8.090 | 89.900 |
| Blocked GQA | 31.913 | 7.639 | 84.202 |

Decode -5.57%, e2e -6.34%, prefill -4.00%; every paired process improves.
Candidate decode range7.492-7.888ms is retained in full; no samples excluded.
IDs/ownership/warm allocation/recapture checks pass, scratch counts unchanged.
Raw confirmation: `rust/profiling/qwen3_5/blocked-gqa-20261109/`.
Commit: `a0cd88f`.

### 8. Full-logit finite scan (done)

Replace short-circuit `.all` with a non-short-circuit boolean reduction so LLVM
can vectorize the complete scan. Every logit is still checked; NaN/Inf rejection,
token range validation, full D2H and host feedback remain in the timed path.
CPU contract cases cover signed zeros, subnormals, finite extremes, NaN and both
infinities at first/middle/last positions, SIMD tails, empty logits and bad IDs.
No GPU math or kernel code changes; use the contract test plus a model reference
smoke check, then rotated native timing.

Contract test, tiny reference/chunking smoke and all-target checks passed.
GPU math unchanged; full D2H/finite/token validation remains inside timing.

| Variant | Prefill ms | Decode ms/token | Total8 outputs ms |
| --- | ---: | ---: | ---: |
| `a0cd88f` | 32.541 | 7.892 | 87.783 |
| Vectorizable finite scan | 32.499 | 7.705 | 86.433 |

Decode -2.37%, e2e -1.54%; every paired process improves decode/e2e. All
observations retained; first-pair prefill variance is visible and not entirely
attributed to the scan. IDs/ownership/warm allocation/capture checks pass.
Raw confirmation: `rust/profiling/qwen3_5/finite-scan-20261109/`.
Commit: `827f7f9`.

### 9. Expert MMA output tile screen (rejected, restored)

A minimal CUDA13.2 driver reproducer validates adding a child graph to a
capturing alternate stream, setting dependencies and joining back to main;
results match. Probe: `rust/profiling/qwen3_5/moe-parallel-20261109/probe/`.
Full-model overlap remains pending.

First screen a smaller change: BF16 expert MMA output columns32->16, preserving
K64/M16 and the existing SIMT/f32 tile32. More CTAs may improve this larger GPU;
Qwen2 N16 was slower on the older36-SM GPU, so measure instead of assuming.
Qualify CPU/SIMT oracle and real BF16 first-ID/chunking before native screening.

Tile16 oracle and real BF16 reference passed (max0.47784/RMS0.08379/ID3636,
all chunks max0). On156-SM GPU, one-process screen regresses to69.405ms prefill,
8.331ms/decode,127.722ms/e2e. Rejected before confirmation; production tile32
restored, no optimization commit. Artifacts: `rust/profiling/qwen3_5/expert-tile16-20261109/`.
Next: implement private shared/routed CUDA Graph branch overlap using the
validated child-node dependency insertion; preserve checkout/module ownership.
