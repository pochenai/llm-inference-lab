# Vectorizable full-logit finite scan

Baseline `a0cd88f`; BF16 B1/32-input/8-output CUDA Graph comparison,3 rotated
processes/variant,2 warmups+3 measured samples, CPU24-31, clocks unlocked. No
GPU tests/compilation/profiling overlap native timing; all observations retained.
Cells are median(process median(sample)); decode samples average7 steps.

| Variant | Prefill ms | Decode ms/token | Total8 outputs ms |
| --- | ---: | ---: | ---: |
| Short-circuit scan | 32.541 | 7.892 | 87.783 |
| Boolean reduction scan | 32.499 | 7.705 | 86.433 |

Decode -2.37%, e2e -1.54%, prefill -0.13%. Every paired process improves decode
and e2e. Process decode medians candidate7.654/7.754/7.705 versus baseline
7.967/7.892/7.867. First-pair prefill varies substantially; no causal claim is
made for its entire improvement. All samples retain full logits D2H and finite
validation, token checks and host feedback; IDs/KV/graph/scratch checks pass.

One source expression changes `.all` to non-short-circuit boolean `.fold`,
allowing LLVM to vectorize the complete scan. No GPU code/math or benchmark
boundary changes. Contract test covers finite extremes, signed zeros,
subnormals, NaN/positive+negative Inf at first/middle/last positions, SIMD tails,
empty arrays and bad token IDs. Contract and tiny model reference/chunking smoke
checks pass; all-target/diff checks pass. The prior21-test/real BF16 GPU gate
remains applicable to unchanged GPU math. No tolerance was changed.

The earlier hot-buffer CPU microbenchmark (all0.189/fold0.052ms) is retained as
prototype.rs; it is not substituted for native inference measurements. Raw
samples, source diff, commands/hashes, summary.json, compare.py, contract tests,
model-smoke.log and all-target-check.log accompany the report. Rust decode
still exceeds the earlier SGLang7.049ms reference; continue material targets.
