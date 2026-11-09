# 32-token blocked GQA

Baseline `06aa36e`; native BF16 B1/32-input/8-output CUDA Graph protocol,
three rotated processes/variant,2 warmups+3 measured samples, CPU24-31. Full
logits validation/host feedback unchanged. Median(process median(sample));
decode samples average7 steps. All observations retained, clocks unlocked;
no compilation/tests/profiler overlap native timing.

| Variant | Prefill ms | Decode ms/token | Total8 outputs ms |
| --- | ---: | ---: | ---: |
| Scalar-position GQA | 33.242 | 8.090 | 89.900 |
| Blocked GQA | 31.913 | 7.639 | 84.202 |

Decode -5.57%, e2e -6.34%, prefill -4.00%. Every paired process improves all
metrics. Candidate decode process medians7.888/7.639/7.492 versus baseline
8.062/8.090/8.094. The candidate range is visibly wider; no best-case selection
or exclusions. IDs match, KV releases, warm graph/scratch accounting unchanged.

The shared eager/graph implementation gathers32 visible tokens per block,
uses causal/padding masks and slot indirection, stable FP32 block softmax, then
the original BF16 y and sigmoid boundaries before multiplication. Both entry
points target SM120 occupancy2. Public kernel arguments/engine API unchanged.

All21 Qwen3.5 regressions and all-target check pass. The independent CPU oracle
covers D256/QH16/KVH2, three causal query rows, fragmented79-slot storage,
standalone/pooled addressing, positions0/31/32/63/64, NaN/unmapped masked tails,
and exact pooled graph/eager outputs. Real BF16 first ID3636 matches reference;
max error0.48618/RMS0.10955, all four chunks exactly match full prefill. Existing
model thresholds are unchanged.

## Memory-check qualification

Compute Sanitizer initially reports potential async-free ordering warnings with
the round-robin policy (blocked20, old kernel10 in these separate runs). Both
complete the CPU oracle successfully; neither initial run proves an actual
kernel out-of-bounds defect. The oracle now explicitly synchronizes its submitted
stream before releasing input owners, making completion visible in the CUDA
timeline rather than relying only on the SDK's host-polled flag. The blocked
oracle then reports0 errors, including unmapped/NaN tail cases. Raw old/new/fenced
logs and old oracle binary are preserved in memcheck/. No production sync was
added; asynchronous ownership is covered separately by Gate tests. These logs
are not a claim that arbitrary cross-stream SDK retirement is sanitizer-clean.

Raw samples, metadata/commands/hashes/source diff, compare.py, oracle.log,
tiny-tests.log, real-reference.log and all-target-check.log accompany this report.
Decode still exceeds SGLang's earlier7.049ms reference; next optimize the material
CPU validation cost and assess remaining expert/parallel costs. No serving or
larger batch/context performance claim.
