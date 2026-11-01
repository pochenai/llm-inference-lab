# LLM Inference Lab

An experiment-driven, learning-first roadmap for understanding LLM inference mechanisms, where they are implemented in real inference engines, performance trade-offs, and validation methods from model files to distributed serving.

## Roadmap

See the detailed [learning roadmap](roadmap.md) for goals, references, experiments, and acceptance criteria.

The Rust/cuTile engine runs Qwen2.5-0.5B-Instruct through Scheduler/Runner with
streaming greedy generation, sessions, reusable scratch and metrics. Packed GPU
batches support ragged prefill/decode, continuous batching, chunked prefill and
one-round submission-first overlap. Native BF16 is implemented.

W8 is Done. It groups the real-model kernel/execution optimizations starting at `52f19b0`:
KV-tile/GQA reuse, shape-based tile-size selection, fused operators, CTA traversal swizzling, split-KV decode
and optional Qwen decode CUDA Graphs. Multi-row BF16 projection tuning uses
Tensor Core-compatible tile shapes; generated SASS confirms the QKV HMMA path.
Measurements distinguish eager and Graph
modes; neither static-model latency nor fixed-cohort overlap establishes serving
throughput. Next is W9 radix prefix reuse, shared KV ownership and cache-aware routing.

| Week / unit | Brief content | Status |
| --- | --- | --- |
| 1 | [Model loader, formats, and storage](d1_model_formats/README.md) | ✅ Implemented |
| 2 | [Scheduler/Runner framework + minimal GEMM/ReLU MLP](rust/engine/README.md), 预备知识 [Nvidia cutile-rs tutorials](https://nvlabs.github.io/cutile-rs/main/index.html) | ✅ Baseline implemented |
| 3 | KVPool/KVContext, token-slot indirection, attention, and fused KV writes | ✅ Baseline implemented |
| 4 | CUDA Graph foundations, pooled workspace/replay, and mapped host output | ✅ SimpleMLP implemented; Qwen extension in W8 |
| 5 | Qwen2.5-0.5B-Instruct: operators, reference checks, scratch reuse, metrics and benchmark | ✅ Done |
| 6 | Continuous batching, chunked prefill and submission-first overlap | ✅ Fixed-cohort overlap A/B and Nsight validated; mixed arrivals pending |
| 7 | Native BF16 Qwen; separate quantization follow-up | ✅ BF16 implemented; quantization planned |
| 8 | Qwen kernel/execution optimization: KV-tile/GQA reuse, M/N/K tile selection, HMMA QKV/down, fusion/swizzling, split-KV and CUDA Graph | ✅ Done; current-source prefill/decode SGLang comparison and Nsight/SASS validation |
| 9 | Radix prefix caching, shared KV lifecycle, and cache-aware routing | ⬜ Next |
| 10 | Modern model architectures | ⬜ Planned |
| 11 | Multi-GPU execution | ⬜ Planned |
| 12 | Distributed KV cache and prefill/decode disaggregation | ⬜ Planned |
| 13 | Routing and serving | ⬜ Planned |
| 14 | End-to-end synthesis | ⬜ Planned |


## Current baseline and next experiment

- Unified `CausalModel::next_tokens`; Runner owns greedy generation, EOS/budgets,
  sessions and streamed text. Qwen executes ragged suffixes in one packed GPU forward.
- Independent reference logits/tokens and full/chunked KV checks; exclusive
  per-forward scratch is reused across decoder layers without zero fills.
- Scheduler observer plugins and `metrics` counters/gauges/histograms; optional
  structured tracing. No exporter or external monitoring service is installed.
- [Qwen benchmark](rust/engine/benches/qwen2.rs) uses fresh KV per sample and
  reports prefill/decode latency, tokens/s, scratch ownership and CUDA-pool peaks.
  See the [engine guide](rust/engine/README.md) for runnable commands.

## W8: Qwen kernel and execution optimization — Done

This unit begins with commit `52f19b0d734c1526a3e5be9ee23df6d00f152829`.
It covers multi-query BF16 prefill attention MMA, K/V tile reuse across queries
and GQA heads, shape-dependent linear/pointwise tiles, bias/SwiGLU fusion, CTA traversal swizzling,
causal K-loop specialization, early V loads, FMA, exp2/FTZ and adaptive split-KV
decode. Shape-based tile-size selection groups input sizes into buckets and selects a tile configuration for each bucket;
CTA traversal swizzling remaps tile indices to improve cache reuse.
Kernel tile reuse is distinct from W9's cross-request prefix reuse.
Long BF16 decode reuses the prefill GQA/MMA and online-softmax computation, with
bounded split-KV packets and the same FP32 merge. Small BF16 forwards pack Q/K/V
into one launch; skinny decode down projections use split-K and reuse attention
scratch after its merge on the same stream.


Qwen's optional decode Graph uses the same Transformer stages as eager. Fixed
scratch/input buffers and dynamic GPU metadata allow request reordering and
history growth without recapture for the same batch size and KV-split count. Cache keys include
KV storage identity; cached graphs retain storage, never session row/slot leases.
Capture/JIT warmup is a cold-path cost. Prefill remains eager and Graph defaults off.

```bash
cargo run --manifest-path rust/Cargo.toml --offline --release -p engine --example qwen2 -- --model-dir MODEL_DIR --cuda-graph
cargo bench --manifest-path rust/Cargo.toml --offline --locked -p engine --bench qwen2 -- --model-dir MODEL_DIR --dtype bf16 --tokens 4096 --batch-size 1 --decode-steps 63 --warmup 2 --iterations 3 --gpu-feedback --cuda-graph
/usr/bin/python3 rust/engine/benches/qwen2_sglang_compare.py --model-dir MODEL_DIR --include-rust-graph --include-sglang-graph
```

The comparison driver builds optimized binaries, warms JIT/capture, runs B1/B2/B4
at 32/4096 initial tokens with 64 fixed outputs, and rotates independent process
order. GPU feedback avoids per-step token readback; final D2H/text decoding and
serving queueing are outside these static-model timers. Eager and Graph results
are reported separately, with raw samples and frozen binaries under the ignored
local `rust/profiling/qwen2/` directory. Numerical/lifecycle checks accompany the
measurements; BF16 reduction/fast-math choices can alter logits and argmax selections.

W8 is complete. The final comparison reruns the current source against SGLang
in both eager and decode Graph modes at B1/B2/B4 and 32/4096 initial tokens.
The complete prefill/decode tables and measurement contracts are in the
[roadmap](roadmap.md#w8-final-current-source-benchmark-results); raw samples,
frozen sources/binary and validation are in the local [full report](rust/profiling/qwen2/w8-final-20261101-114737/matrix/RESULTS.md).

The 4K-input subset is below (Rust / SGLang, milliseconds). Parentheses show
Rust latency reduction versus SGLang in the same execution mode:
`(SGLang - Rust) / SGLang × 100%`.

| Batch | Eager prefill ms | Eager decode ms/step | CUDA Graph decode ms/step |
| ---: | ---: | ---: | ---: |
| 1 | 90.93 / 95.56 (↓4.8%) | 5.665 / 7.266 (↓22.0%) | 4.151 / 4.335 (↓4.2%) |
| 2 | 175.92 / 185.42 (↓5.1%) | 5.528 / 7.934 (↓30.3%) | 3.799 / 4.116 (↓7.7%) |
| 4 | 344.80 / 377.92 (↓8.8%) | 5.563 / 8.253 (↓32.6%) | 4.221 / 4.297 (↓1.8%) |

Rust uses optimized release code; both engines use BF16, two warmups, three measured
samples per process and three independent process repetitions. Modes are compared separately.
Graph replay accelerates decode; prefill remains eager. Clocks are unlocked; these
static-model results do not establish serving throughput or generation quality.

The final small-M tuning uses Tensor Core-compatible tiles for packed QKV and
split-K down, selected through private M/N/K policies. QKV SASS changes from SIMT
to `HMMA.16816.F32.BF16`; independent profiles lower QKV GPU time about 34% and
down+merge about 15%, with no new kernel or scratch Tensor. The earlier isolated
[B4/4K experiment](rust/profiling/qwen2/decode-b4-4k-20261101-110247/RESULTS.md)
records its own 5.1% decode wall-time improvement, hardware counters and rejected
candidates. That targeted percentage is separate from the final matrix above.
BF16 reduction/tile changes may alter logits and generated IDs; CPU/operator,
lifecycle and exact real-model eager/Graph checks accompany the implementation.

Next: W9 radix prefix matching, shared computed-KV ownership/reclamation and
cache-aware routing. True quantization remains a separate W7 follow-up.


Graph cache counters report captures, replays, cached instances and retained
workspace bytes. This permanent storage is included in workspace in-use bytes;
KV row/slot release and in-flight scratch ownership are checked separately.
The [engine guide](rust/engine/README.md) documents the execution contracts.
Radix caching/routing is W9; true quantization remains a separate W7 follow-up.

## Future plans

- [Speculative decoding](roadmap.md#speculative-decoding-deferred): deferred; no fixed week or current acceptance requirement.
- RDMA
- CoW for forked Conversation
- Local LLM like [Strata](https://github.com/Niko1221/Strata)

## References
- https://github.com/patchy631/time-to-first-token
