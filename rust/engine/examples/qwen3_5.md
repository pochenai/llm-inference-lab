# Qwen3.5-35B-A3B BF16 text engine

This experiment implements the language backbone of the official
`Qwen/Qwen3.5-35B-A3B` checkpoint in the existing Rust/cuTile engine.
It has 40 decoder layers: three Gated DeltaNet layers followed by one full
attention layer, repeated ten times. Every layer has 256 routed experts
(top eight per token) and a sigmoid-gated shared expert. Hidden width is 2048.
The vision encoder and MTP weights are not loaded.

Source lives in `rust/engine/src/runner/model/qwen3_5/`, alongside the existing
Qwen2 engine. `config.rs` validates supported checkpoint features and computes
state budgets; `constants.rs` names architecture dimensions and launch tiles,
sharing fixed GPU tile constants with `kernels.rs`; `engine.rs`
loads weights and owns execution. Embedding, BF16/FP32 projections, residual
addition and SwiGLU reuse Qwen2's operators without changing their math.

## Execution and state

`Qwen3_5Engine::load` loads the sharded checkpoint and tokenizer. Allocate one
exclusive `Qwen3_5Context` per session with `new_context(max_seq_len)`.
`step_with_logits(history, &mut context)` consumes the uncached suffix of a
complete token history and returns a greedy token ID plus final-position FP32
logits. `max_seq_len` limits the session's total token history, including prompt
and generated tokens. Histories must extend the committed prefix and remain
within this limit.
Separate contexts may execute independently; a context cannot be used with a
different engine. The engine uses its loading policy for all submissions.

Full-attention layers store rotated BF16 K/V, with per-head zero-centered
RMSNorm, 64 rotary channels of each 256-channel head, and a sigmoid output gate.
Text position IDs coincide across all three mRoPE axes. DeltaNet layers retain
three raw convolution projections and a FP32 `[value_heads, key_dim, value_dim]`
recurrent state. Q/K L2 normalization precedes the decay/delta update; gated
RMSNorm follows it. Expert routing/selection and selected-expert projections
stay on the GPU; tied router scores choose the lowest expert index.

Successful GPU completion commits the history. Errors or cancellation during
submission invalidate the context, which must then be discarded. Input errors
before submission leave it usable. Recurrent states are exclusive and cannot
be split into arbitrary cached prefixes. Runner/CausalModel integration,
recurrent snapshots, prefix caching, CUDA Graphs, packed batches and sampling
remain subsequent work; this engine does not declare W10 complete.

This is a correctness baseline: prefill uses sequential DeltaNet recurrence,
attention scans causal history, and selected-expert projections use direct FP32
reductions. Scratch is allocated per forward/layer. Submission can block on JIT/module
loading, allocations or pageable H2D; submission-first overlap and a fully warmed
gated cancellation check require workspace/input reuse and remain unvalidated. These choices need tuning before
serving performance comparisons. The vocabulary head runs only for the last
input position. Greedy selection currently reads diagnostic logits to the CPU.

## Memory

The downloaded checkpoint index reports 71,903,655,008 bytes including vision
and MTP. The text-only loading plan contains 693 tensors and 69,321,574,912
resident weight bytes (64.56 GiB), including FP32 norm/gating metadata.
`engine.weight_bytes()` reports the tensors actually loaded. BF16 full-attention K/V costs 20,480 bytes/token
across ten layers. Each of the thirty DeltaNet layers has 2,097,152 bytes of
recurrent state and 49,152 bytes of convolution history. At a maximum sequence length of 4096 tokens,
`config.text_config.state_bytes(4096)` is 148,275,200 bytes/request.
Weights, context storage, live scratch, the CUDA allocator and JIT overhead all
need to fit; the checkpoint's active-parameter count does not describe residency.

## Run

From the repository root, with CUDA 13.2 and `tileiras` available:

```bash
cargo run --manifest-path rust/Cargo.toml --offline --release -p engine --example qwen3_5 -- \
  --model-dir ../models/Qwen3.5-35B-A3B --prompt 'What is a language model?' \
  --chunk-size 32 --max-new-tokens 16
```

The example formats one user turn with thinking disabled, supports chunked
prefill and EOS, and optionally writes first-step logits with `--logits PATH`.
The FP8 checkpoint is rejected; native FP8 execution belongs to W13.

## Independent checks

`qwen3_5_reference.py` is an eager PyTorch implementation with no Rust operators,
Transformers dependency or custom CUDA kernels. It loads one decoder layer at
a time, keeping reference peak GPU residency below the full model size.
It needs an existing PyTorch environment. BF16 matmul partial reductions stay
in FP32 (`allow_bf16_reduced_precision_reduction=False`); tied router logits use
a stable sort by expert index. Stock backend precision modes may select different
experts or greedy tokens. Optional `--trace-dir PATH` saves per-layer diagnostic
tensors for independent comparisons. All generated files go under ignored
`artifacts/`; checkpoint files are never modified.

```bash
python rust/engine/examples/qwen3_5_reference.py \
  --tiny-from ../models/Qwen3.5-35B-A3B \
  --model-dir rust/engine/artifacts/qwen3_5/tiny \
  --output rust/engine/artifacts/qwen3_5/tiny/reference.f32
cargo test --manifest-path rust/Cargo.toml --offline -p engine qwen3_5::config --lib
QWEN35_TEST_MODEL="$PWD/rust/engine/artifacts/qwen3_5/tiny" \
  cargo test --manifest-path rust/Cargo.toml --offline -p engine qwen3_5::tests --lib -- --ignored --nocapture
mkdir -p rust/engine/artifacts/qwen3_5/real
python rust/engine/examples/qwen3_5_reference.py --model-dir ../models/Qwen3.5-35B-A3B \
  --device cuda --output rust/engine/artifacts/qwen3_5/real/reference.f32
QWEN35_TEST_MODEL="$PWD/../models/Qwen3.5-35B-A3B" \
QWEN35_TEST_REFERENCE="$PWD/rust/engine/artifacts/qwen3_5/real/reference.f32" \
  cargo test --manifest-path rust/Cargo.toml --offline --release -p engine qwen3_5::tests --lib -- --ignored --nocapture
```

The reference fixture uses IDs `[1,7,9,3,12]`. GPU tests compare independent
logits/argmax, full prefill against chunks of 1/2/3/4 tokens (including decode),
and input/capacity validation.
A separate GPU test checks tied router indices and selected-score normalization. Config tests reject unsupported models and FP8.

See [validation results](qwen3_5_validation.md) for measured errors, exact chunking results,
smoke output and validation limits. The engine and reference read configuration directly from the supplied model directory.

## References

- [Official model](https://huggingface.co/Qwen/Qwen3.5-35B-A3B); configuration is read from the supplied model directory.
- [Transformers Qwen3.5 MoE](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py).
- [NVIDIA TileGym Qwen3.5](https://github.com/NVIDIA/TileGym/blob/main/src/tilegym/transformers/qwen3_5/modeling_qwen3_5.py).
- Local `../sglang/python/sglang/srt/models/qwen3_5.py` for hybrid layer structure, packed projection loading and norm semantics.
