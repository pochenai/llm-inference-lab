# LLM Inference Lab

An experiment-driven, learning-first roadmap for understanding LLM inference mechanisms, where they are implemented in real inference engines, performance trade-offs, and validation methods from model files to distributed serving.

## Roadmap

See the detailed [learning roadmap](roadmap.md) for goals, references, experiments, and acceptance criteria.

| Day | Brief content | Status |
| --- | --- | --- |
| 1 | [Model formats and loading](w1_model_formats/README.md) | ✅ Done |
| 2 | Core operators and kernel fusion | ⬜ Planned |
| 3 | Benchmarking and profiling | ⬜ Planned |
| 4 | Attention, FlashAttention, and paged-attention kernel concepts | ⬜ Planned |
| 5 | Single-node engine, scheduling, and chunked prefill | ⬜ Planned |
| 6 | Paged KV cache management; optional direct paged-attention / paged-FlashAttention integration | ⬜ Planned |
| 7 | Compilation and execution | ⬜ Planned |
| 8 | Quantization | ⬜ Planned |
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