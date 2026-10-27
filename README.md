# LLM Inference Lab

An experiment-driven, learning-first roadmap for understanding LLM inference mechanisms, where they are implemented in real inference engines, performance trade-offs, and validation methods from model files to distributed serving.

## Roadmap

See the detailed [learning roadmap](roadmap.md) for goals, references, experiments, and acceptance criteria.

The Rust/cuTile engine now runs Qwen2.5-0.5B-Instruct through Scheduler/Runner,
with streaming greedy generation, sessions, reusable scratch and metrics.
Packed GPU batches support ragged prefill and decode. Release profiling led to
KV-block attention with a measured occupancy target and larger prefill linear tiles.
Runner supports continuous batching, token-budgeted chunked prefill and one-round
submission-first overlap. Runner serial/overlap release measurements and Nsight
validation are complete; gains depend on workload. Next: align Qwen CUDA Graph
and native BF16 execution for the SGLang comparison. See the roadmap's
current checkpoint for acceptance gaps; prefix reuse and true quantization follow.
Profiling and numerical/lifecycle checks accompany every phase. Week labels are
learning units; speculative decoding and the later plan remain unchanged.

| Week / unit | Brief content | Status |
| --- | --- | --- |
| 1 | [Model loader, formats, and storage](d1_model_formats/README.md) | ✅ Implemented |
| 2 | [Scheduler/Runner framework + minimal GEMM/ReLU MLP](rust/engine/README.md), 预备知识 [Nvidia cutile-rs tutorials](https://nvlabs.github.io/cutile-rs/main/index.html) | ✅ Baseline implemented |
| 3 | KVPool/KVContext, token-slot indirection, attention, and fused KV writes | ✅ Baseline implemented |
| 4 | CUDA Graph, pooled workspace/replay, and mapped host output | ✅ SimpleMLP implemented; Qwen pending |
| 5 | Qwen2.5-0.5B-Instruct: operators, reference checks, scratch reuse, metrics and benchmark | ✅ Done |
| 6 | Continuous batching, chunked prefill and submission-first overlap | ✅ Fixed-cohort overlap A/B and Nsight validated; mixed arrivals pending |
| 7 | Native BF16 Qwen; separate quantization follow-up | ⬜ Planned; BF16 is not quantization completion |
| 8 | Radix prefix caching, shared KV lifecycle, and cache-aware routing | ⬜ Planned |
| 9 | Speculative decoding | ⬜ Planned |
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

The local [packed Qwen baseline](rust/profiling/qwen2/packed-nsys/RESULTS.md) and
[attention optimization report](rust/profiling/qwen2/attention-tiling/RESULTS.md)
record short/long prefill and decode at batch sizes 1 and 4. The subsequent
[batch tuning report](rust/profiling/qwen2/batch-tuning/RESULTS.md) compares batch
sizes 1, 2 and 4 with larger prefill linear tiles and attention occupancy targets.
Split-KV decode remains optional; Qwen CUDA Graph and native BF16 are comparison-alignment follow-ups.
Continuous batching admits new requests between packed forwards and releases
finished requests independently. Chunked prefill shares a bounded token budget
with decode; intermediate chunks advance KV without emitting predictions. Mixed
long/short arrival profiling remains open. The local
[Runner overlap report](rust/profiling/qwen2/runner-overlap/RESULTS.md) compares fixed
cohorts at batch 1/2/4 and input 32/4096; CUDA Graph and native BF16 remain next.
Radix caching and quantization follow.

## Future plans

- RDMA
- CoW for forked Conversation
- Local LLM like [Strata](https://github.com/Niko1221/Strata)

## References
- https://github.com/patchy631/time-to-first-token
