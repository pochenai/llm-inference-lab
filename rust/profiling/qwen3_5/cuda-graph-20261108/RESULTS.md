# Qwen3.5 BF16 CUDA Graph iteration

Status: graph iteration committed as `70e8e87` after user approval. Frozen measurement source/binaries precede the later cold-path Vec-capacity cleanup, which passed its graph consistency regression. The approved workspace/NVTX iteration was committed as `6d5ab1f`.

## Native latency

RTX 6000D, CUDA 13.2, native BF16 Qwen3.5-35B-A3B, batch one, 32 repeated input tokens, eight fixed generated outputs. Each engine has three independent processes, two warmups and three measured samples/process. Process order rotates eager/graph/SGLang. CPU affinity is 24–31 on the GPU-local NUMA node. No profiler, compiler or GPU tests overlap native timing. Loading/JIT/capture are excluded; prefill includes fresh recurrent-state initialization. Clocks and external host load are not controlled.

Each process aggregates prefill/total by sample mean and decode by the median of sample step medians. The table is the median of the three process aggregates.

| Engine | Prefill ms | Decode ms/step | Total ms |
| --- | ---: | ---: | ---: |
| Committed workspace Rust eager | 52.882 | 12.404 | 140.965 |
| Rust CUDA Graph | 55.362 | 10.654 | 131.103 |
| SGLang 0.5.9 CUDA Graph | 69.120 | 7.049 | 119.244 |

Decode decreases 14.1% (1.164x), total decreases 7.0%, and prefill regresses 4.7%. This crosses the review threshold on decode, not on total latency. The prefill regression is visible and its root cause has not been isolated. Rust graph decode remains 1.512x SGLang; total remains 1.099x. Candidate process decode medians are 10.648, 10.654 and 10.685 ms; baseline medians are 12.404, 12.169 and 12.846 ms.

All Rust samples generate the same eight IDs. Each graph process captures once during warmup, replays seven times/sample, and performs no warmed scratch allocation or recapture. Session/KV metrics return to their initial values after every sample. One graph keeps one workspace checked out; one prefill workspace is idle. Logical resident scratch remains 10,729,868 bytes, excluding graph runtime, staging, weights, KV and recurrent state.

Rust keeps host token feedback and full-logit CPU finite validation. SGLang uses GPU sampling with its stock/default MoE tuning, Triton attention, BF16 weights and FP32 Mamba/DeltaNet state. This is the same SGLang graph setup used in the previous workspace iteration, not an exhaustive tuning search. Cross-engine generated-ID/logit equivalence is not established.

## Implementation

Only pooled single-token continuations after initial prefill replay a graph. Initial/multi-token prefill and standalone diagnostics remain eager. A cached instance owns fixed token/metadata buffers, captured compute/output/scratch, weights and pool backing storage. Each replay uploads current token, position, mapping-row offset and convolution/recurrent addresses. The pending execution view validates write authority and retains the current session/KV owners through completion. The cache does not retain request row/slot leases or recurrent allocations between samples.

The warmed scratch checkout remains exclusively held through capture. Graph preparation runs before the append submission; capture uses cuda-async's capture execution context without nested synchronization. Private recurrent/KV state is used for warmup. Readback and host validation remain outside the captured compute. Up to four graph instances are cached by table identity; busy entries have independent staging/output/scratch, and overflow entries expire after use. Escaped output owners also exclude reuse.

Dynamic state pointers explicitly preserve the host-checked 16-byte alignment used by eager kernel arguments. Without this information, tileiras selected different memory/reduction code: an operator reproducer differed by up to 1.49e-8 in FP32 recurrent state and the real model later amplified a difference to 0.185 in logits. Restoring alignment fixes both, while keeping the original arithmetic formulas. No FMA rewrite or looser numerical threshold is included. Diagnostic logs and unhinted SASS are retained in `diagnostics/`.

## Validation

- Final release Qwen3.5 suite: 14 passed, including independent BF16 reference/chunking, BF16/FP32 pooled controls, graph Runner, workspace/reset, real-geometry convolution/recurrence and Gate checks.
- Real 35B BF16 common-history comparison: all twelve paired eager/graph logits vectors have max difference zero. It alternates two rows with fragmented slots and repeats after fresh session/state allocation, while reusing one capture.
- The real-geometry operator regression matches BF16 outputs and FP32 recurrent-state bits. The Gate test verifies independent completion and excludes reuse while submissions or escaped outputs remain live.
- All-target compilation, formatting and diff checks pass. Root pyright remains blocked by the missing offline `torch==2.14.0+cpu` wheel; no Python dependency environment was changed.

The first real-checkpoint debug run was stopped during slow weight loading and replaced with release validation. Initial unhinted numerical failures are diagnostic evidence, not accepted validation. Full hybrid cancellation/overlap stress, long-context performance, packed batches and serving throughput are outside this iteration.

## Reproduction and artifacts

`run.py` contains exact commands; `commands.json`, `metadata.json`, frozen `baseline`/`candidate` executables, source snapshots, binary/model-config hashes, raw process outputs, `processes.json` and `summary.json` retain the native evidence. `validation-release-tests.log` and `real-bf16-logits-validation.log` contain the completed numerical checks. Separate Nsight captures use CUDA-profiler API boundaries after two warmups and `--cuda-graph-trace=node`; they are attribution evidence, not the native latency table.

## Separate Nsight attribution

One measured prefill plus seven decodes has the same 7,476 GPU kernel instances in both modes. Ordinary `cuLaunchKernel` calls decrease from 7,476 to 994, with seven `cuGraphLaunch` calls added. The remaining ordinary launches include prefill and per-step pool mapping. Allocation/free calls and event records remain 60 each, corresponding to fresh recurrent/convolution state initialization rather than scratch misses.

The profiled per-decode GPU kernel sum is 8.847 ms eager and 8.660 ms graph; prefill sums are 49.779/49.807 ms. These profiler measurements support submission reduction as the main change, without treating profiler kernel sums as native wall latency or subtracting them from the native table. They do not isolate the native prefill regression.

Candidate NVTX has one prefill, seven `qwen3_5.decode.graph_replay` ranges, no capture range and no workspace miss during measurement. All ranges are closed on the same thread. Each layer is visited once on the host for eager prefill; graph node tracing supplies replayed kernel events. `profile-summary.json` retains validated counts and kernel attribution. Candidate decode kernel time is 57.3% `linear_entry` and 27.4% `expert_entry`; these remain potential later-iteration targets after review.
