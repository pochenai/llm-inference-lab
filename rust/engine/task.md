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
- [ ] Inspect emitted instructions and optimize routed expert GEMMs, which cost
  2.376 ms/decode step in the graph profile, before narrower projection roles.
- [ ] Optimize small projections, starting with shared expert gate/up, GDN a/b
  and router. Preserve large projections that already match SGLang.
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
Commit: this task/result update accompanies the replay storage implementation.
