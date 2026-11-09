# Qwen3.5 BF16 workspace reuse — review candidate

This is one performance iteration: reuse scratch buffers and add NVTX attribution.
GPU kernels, numerical reductions, host feedback and full-logit validation are
unchanged. No GPU-feedback or graph implementation is included in the candidate.
The working tree is intentionally uncommitted pending user review.

## Native paired measurements

RTX 6000D, CUDA 13.2, cuTile 0.5.0; official Qwen3.5-35B-A3B BF16 checkpoint.
BF16 inference in every performance case, with the existing FP32 recurrent state
and required accumulations. No separate FP32 inference performance comparison.
CPU affinity is 24–31 on GPU-local NUMA node 0. B1, 32 repeated token-9707 inputs,
eight fixed outputs without EOS stopping, fresh session state per sample.

Baseline, candidate and SGLang graph run sequentially, rotating their order over
three independent processes each. Each process has two warmups and three measured
samples. No profiler, compiler or GPU tests overlap native timing. GPU/CPU clocks
and unrelated host load are not controlled. Values below are medians of process
aggregates: mean prefill/total, median of per-sample median decode steps.

| Engine | Prefill ms | Decode ms/step | Total for eight outputs ms |
| --- | ---: | ---: | ---: |
| Baseline Rust eager | 60.753 | 31.594 | 279.430 |
| Workspace Rust eager | 52.959 | 12.811 | 142.067 |
| SGLang graph reference | 68.651 | 7.059 | 118.076 |

Observed prefill reduction is **12.8%** and decode reduction **59.5% (2.47x)**.
The remaining decode ratio to the graph reference is **1.81x**. All measured Rust
samples have identical complete generated IDs before/after, eight forwards and
released KV. Every candidate measured sample has zero scratch allocations, zero
in-use workspaces and two cached workspaces, totaling **10,729,868 bytes (10.23 MiB)**
of logical scratch storage. This excludes weights, KV/recurrent state, uploads,
host readback buffers and allocator reservation.

Independent decode process medians in milliseconds:

- Baseline: 37.868, 29.006, 31.594.
- Candidate: 12.811, 12.570, 13.418.
- SGLang graph: 7.066, 7.059, 7.021.

Baseline process variation is material. These are observed results without a
confidence-bound claim, and the fresh baseline is used instead of comparing only
against the older measurement window.

## Scope and ownership

The cache holds up to two idle row/context-specific workspaces. Each checkout has
global hidden/residual/head storage and separate reusable DeltaNet/GQA layer
banks, so scratch storage does not grow with decoder depth. Producers overwrite
valid elements; scalar bias slots retain their initial zero. Session recurrent
and KV state remain separate and keep their existing initialization/lifetime.

A lease is retained by the execution context and every derived tensor owner.
Buffers return to the cache only after submission completion and release of
escaped views. Foreign views report their backing tensor's actual CUDA context.
The existing integer upload pool also reuses input destinations. Allocation and
ownership snapshots are recorded by the benchmark outside forward timers.

## Separate Nsight evidence

Each capture contains one prefill and seven decodes after warmup. These profile
numbers are diagnostics; they are not pooled with native timings or added to
GPU kernel duration sums.

| Captured API calls | Baseline | Candidate |
| --- | ---: | ---: |
| Kernel launches | 18,260 | 7,476 |
| Allocation calls | 10,852 | 60 |
| Free calls | 10,852 | 60 |
| Event records | 10,844 | 60 |

All remaining 60 allocation calls fall inside `qwen3_5.state.allocate`: fresh
session convolution/recurrent state. Warm scratch allocation ranges are absent.
Summed GPU decode kernel time is 9.73/8.84 ms per forward before/after; these are
not native wall times. Reducing repeated allocation, zero-fill and submission
work is the measured change; it does not change model math.

NVTX assertions confirm one prefill submission, seven decode submissions, eight
workspace checkouts, and eight visits to every `layer.0` through `layer.39`.
They also confirm 240 DeltaNet and 80 GQA mixer ranges, 320 routed gate/up and
down ranges each, and eight host-validation ranges. All recorded Qwen3.5 ranges
are closed on their originating thread. Scopes cover synchronous CPU submission
only and never cross an await; use CUDA correlation for GPU attribution.

## Validation and comparison limits

- Ten Qwen3.5 tests passed, including reference/chunking, pooled Runner controls
  and two Gate-based workspace tests. The final BF16 reference and both Gate
  tests passed again after instrumentation was added.
- All workspace targets compile; formatting and diff checks passed.
- NVTX counts and GPU grouping pass assert-based trace checks. Native checks
  validate all nine candidate samples and before/after generated IDs.
- The required offline pyright command remains blocked by the absent locked
  CPU PyTorch wheel; it is not recorded as passing.

Rust still includes host feedback and full-logit D2H/finite-value validation;
SGLang uses GPU sampling. SGLang 0.5.9 uses Triton attention, BF16 KV, FP32 Mamba
state, and default MoE tuning because no RTX 6000D-specific file is installed.
Cross-engine generated histories were not verified equal. These are B1/short-input
static forwards, not production serving throughput or a tuned-SGLang ranking.
Gate tests cover local scratch ownership, independent completion and escaped
views; full hybrid-model cancellation stress is not claimed by this iteration.

Raw samples, commands, process aggregates, frozen binaries/sources and hashes,
validation logs, Nsight reports/SQLite and the analysis scripts accompany this
report. The baseline is the frozen pre-workspace 0.5.0 profiling executable from
the earlier baseline; both executables run natively without --profile first.
