# BF16 scalar and MoE router vector projections

Baseline `aa1fb74`; three rotated processes/variant, B1/32-input/8-output CUDA
Graph mode, 2 warmups + 3 measured samples, CPU 24-31, full logits validation
and host feedback. Median(process median(sample)); each decode sample averages
seven steps. All observations retained; no tests/compilation/profiler overlap
native timing. Clocks unlocked; CUDA async completion variables are unset.

| Variant | Prefill ms | Decode ms/token | Total 8 outputs ms |
| --- | ---: | ---: | ---: |
| Baseline | 33.044 | 9.015 | 96.310 |
| Scalar/router vector | 33.117 | 8.685 | 93.981 |

Decode -3.65%, total -2.42%, prefill +0.22% (+0.073 ms). Every paired process
improves decode/e2e. Decode process medians baseline 8.987/9.061/9.015 versus
candidate 8.685/8.627/8.832 ms. IDs match; all sessions release KV; warmed samples
keep one graph, 126 allocations, 10,594,684 resident scratch bytes with zero
recapture/allocation. The earlier SGLang 7.049 ms/token reference is still ahead.

The vector kernel reuses the shared expert's full-K projection helper. Only
BF16 N=1 and N=256 projections select it, consistently across token counts.
Larger projections and GDN a/b retain MMA.

## Precision gate

The broader N<=256 candidate passed its CPU oracle and tiny tests but changed
the real reference ID 3636 -> 16; it was rejected before native timing. A
single-load diagnostic independently enabled scalar, a/b and router families:
scalar/router match baseline reference logits exactly; only a/b changes them.
The final scalar+router candidate passes the unchanged first-ID/error/chunking
gates: real reference max 0.56865, RMS 0.11685, ID 3636, all four chunks max 0.
All 16 BF16-focused tests pass, including independent CPU N1/32/33/256 tests,
tiny reference, exact graph/eager logits, Runner and gated lifetime checks.
All-target check passes. Diagnostic switches/tests are absent from production.
No numerical thresholds were widened.

Native samples, metadata/commands, binary hashes, patch.diff, compare.py,
oracle.log, tiny-tests.log, real-reference.log and all-target-check.log accompany
this report. Rejected candidate and precision-screen artifacts are retained in
../skinny-projections-20261109/. No larger batch/context performance or broad
text-quality claim. Next: pack a/b and reuse the original MMA accumulation.
