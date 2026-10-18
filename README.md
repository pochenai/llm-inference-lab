# LLM Inference Lab

An experiment-driven, learning-first roadmap for understanding LLM inference mechanisms, where they are implemented in real inference engines, performance trade-offs, and validation methods from model files to distributed serving.

## Roadmap

See the detailed [learning roadmap](roadmap.md) for goals, references, experiments, and acceptance criteria.

Extend one small Rust/cuTile engine through execution, KV management, CUDA Graph,
scheduling, prefix reuse, and quantization before adding Qwen2.5-0.5B-Instruct.
Profiling and numerical/lifecycle checks accompany every phase. Week labels are
learning units; speculative decoding and the later plan remain unchanged.

| Week / unit | Brief content | Status |
| --- | --- | --- |
| 1 | [Model loader, formats, and storage](d1_model_formats/README.md) | ✅ Implemented |
| 2 | [Scheduler/Runner framework + minimal GEMM/ReLU MLP](rust/engine/README.md), 预备知识 [Nvidia cutile-rs tutorials](https://nvlabs.github.io/cutile-rs/main/index.html) | ✅ Baseline implemented |
| 3 | KVPool/KVContext, token-slot indirection, attention, and fused KV writes | ✅ Baseline implemented |
| 4 | CUDA Graph, reusable workspace, and stable-address replay | ⏭ Next |
| 5 | Step-level generation, continuous batching, and chunked prefill | ⬜ Planned |
| 6 | Radix prefix caching, shared KV lifecycle, and cache-aware routing | ⬜ Planned |
| 7 | Quantization on the toy engine | ⬜ Planned |
| 8 | Qwen2.5-0.5B-Instruct: real architecture and additional operators | ⬜ Planned |
| 9 | Speculative decoding | ⬜ Planned |
| 10 | Modern model architectures | ⬜ Planned |
| 11 | Multi-GPU execution | ⬜ Planned |
| 12 | Distributed KV cache and prefill/decode disaggregation | ⬜ Planned |
| 13 | Routing and serving | ⬜ Planned |
| 14 | End-to-end synthesis | ⬜ Planned |


## Future plans
- RDMA
- CoW for forked Conversation
- Local LLM like [Strata](https://github.com/Niko1221/Strata)

## References
- https://github.com/patchy631/time-to-first-token