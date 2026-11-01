# LLM Inference Lab

An experiment-driven, learning-first roadmap for understanding LLM inference mechanisms, where they are implemented in real inference engines, performance trade-offs, and validation methods from model files to distributed serving.

## Roadmap

See the detailed [learning roadmap](roadmap.md) for goals, references, experiments, and acceptance criteria.

The Rust/cuTile engine runs Qwen2.5-0.5B-Instruct through Scheduler/Runner with
streaming greedy generation, sessions, reusable scratch and metrics. Packed GPU
batches support ragged prefill/decode, continuous batching, chunked prefill and
one-round submission-first overlap. Native BF16 is implemented.

W8 groups the real-model kernel/execution optimizations starting at `52f19b0`:
KV-tile/GQA reuse, shape-based tile-size selection, fused operators, CTA traversal swizzling, split-KV decode
and optional Qwen decode CUDA Graphs. Measurements distinguish eager and Graph
modes; neither static-model latency nor fixed-cohort overlap establishes serving
throughput. Radix prefix reuse/routing now starts at W9; later units shift by one.

| Week / unit | Brief content | Status |
| --- | --- | --- |
| 1 | [Model loader, formats, and storage](d1_model_formats/README.md) | ✅ Implemented |
| 2 | [Scheduler/Runner framework + minimal GEMM/ReLU MLP](rust/engine/README.md), 预备知识 [Nvidia cutile-rs tutorials](https://nvlabs.github.io/cutile-rs/main/index.html) | ✅ Baseline implemented |
| 3 | KVPool/KVContext, token-slot indirection, attention, and fused KV writes | ✅ Baseline implemented |
| 4 | CUDA Graph foundations, pooled workspace/replay, and mapped host output | ✅ SimpleMLP implemented; Qwen extension in W8 |
| 5 | Qwen2.5-0.5B-Instruct: operators, reference checks, scratch reuse, metrics and benchmark | ✅ Done |
| 6 | Continuous batching, chunked prefill and submission-first overlap | ✅ Fixed-cohort overlap A/B and Nsight validated; mixed arrivals pending |
| 7 | Native BF16 Qwen; separate quantization follow-up | ✅ BF16 implemented; quantization planned |
| 8 | Qwen kernel/execution optimization: KV-tile/GQA reuse, shape-based tile-size selection, kernel fusion, CTA traversal swizzling, split-KV decode and CUDA Graph | 🚧 Kernels/Graph validated; SGLang Graph performance gap remains |
| 9 | Radix prefix caching, shared KV lifecycle, and cache-aware routing | ⬜ Planned |
| 10 | Speculative decoding | ⬜ Planned |
| 11 | Modern model architectures | ⬜ Planned |
| 12 | Multi-GPU execution | ⬜ Planned |
| 13 | Distributed KV cache and prefill/decode disaggregation | ⬜ Planned |
| 14 | Routing and serving | ⬜ Planned |
| 15 | End-to-end synthesis | ⬜ Planned |


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

## W8: Qwen kernel and execution optimization

This unit begins with commit `52f19b0d734c1526a3e5be9ee23df6d00f152829`.
It covers multi-query BF16 prefill attention MMA, K/V tile reuse across queries
and GQA heads, shape-dependent linear/pointwise tiles, bias/SwiGLU fusion, CTA traversal swizzling,
causal K-loop specialization, early V loads, FMA, exp2/FTZ and adaptive split-KV
decode. Shape-based tile-size selection groups input sizes into buckets and selects a tile configuration for each bucket;
CTA traversal swizzling remaps tile indices to improve cache reuse.
Kernel tile reuse is distinct from W9's cross-request prefix reuse.

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
measurements; BF16 reduction/fast-math changes can alter borderline argmaxes.

The matched three-process matrix shows decode Graph lowers our own latency by
about 16–25%. At 4K, Rust/SGLang Graph decode is 4.76/4.22 ms (B1), 5.40/3.95 ms
(B2) and 6.68/4.22 ms (B4). Short B1 is close (about 1.7% slower); Graph is not generally
faster than SGLang. The remaining GPU work motivates GQA decode packing/MMA and
small-M linear/GEMV experiments. These are static measurements, not serving results.
See the local [Graph comparison report](rust/profiling/qwen2/cuda-graph-20261101/RESULTS.md).

Graph cache counters report captures, replays, cached instances and retained
workspace bytes. This permanent storage is included in workspace in-use bytes;
KV row/slot release and in-flight scratch ownership are checked separately.
The [engine guide](rust/engine/README.md) documents the execution contracts.
Radix caching/routing is W9; true quantization remains a separate W7 follow-up.

## Future plans

- RDMA
- CoW for forked Conversation
- Local LLM like [Strata](https://github.com/Niko1221/Strata)

## References
- https://github.com/patchy631/time-to-first-token
