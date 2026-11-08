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
loads weights and owns execution. `state.rs` owns hybrid storage, typed execution
views and standalone state transitions; `pool.rs` integrates those resources
with Runner's incremental KV allocation and commit lifecycle. Embedding, BF16/FP32 projections, residual
addition and SwiGLU reuse Qwen2's operators without changing their math.

The engine, context, state pool and execution view are generic over floating-point
storage `T`, defaulting to `bf16`, like the existing Qwen2 implementation. `T`
selects weight, activation, GQA KV and convolution-history storage; norm metadata,
reductions, DeltaNet recurrent matrices and diagnostic logits stay FP32. The
example explicitly selects BF16. A reduced fixture also exercises `f32` through
the same standalone and Runner paths. Checkpoint `dtype` describes source weights,
not the requested execution type; WeightLoader performs the existing conversion.
This change does not add quantized checkpoint support.

## Runner execution and state

The `qwen3_5` example uses Scheduler -> ModelRunner -> CausalModel, with chunked
prefill, greedy decoding, EOS and normal session cleanup. `--max-batch-tokens`
controls Runner chunking (`--chunk-size` remains an alias). `--repeat 2` checks
independent requests and state release.

`Qwen3_5StatePool<T>` wraps the existing `NaiveKVPool<T>`: shared backing buffers
are allocated at startup, while each prefill chunk reserves only its new token
slots and each decode appends one slot. Mapping rows, token slots and maximum
sequence length remain separate capacities. Only full-attention layers occupy
KV storage; their model layer IDs map to compact pool layer IDs. DeltaNet
convolution/recurrent buffers have fixed size per session, are initialized on
first forward and released with the context. MoE weights remain engine-owned.
The pool delegates admission, slot mapping, K/V views and in-flight slot leases
to NaiveKVPool. Prefix caching is rejected during Runner construction because
recurrent state snapshots are not implemented.

Models do not receive mutable session metadata. The optional `ExecutionStatePool`
extension supplies a backend-specific `ExecutionView` for an already pending
append through `KVContext::execution_view(tokens, exec)`. Qwen3.5's view provides
per-layer DeltaNet update pointers and GQA cache access, retaining GPU buffers and
slot leases before GPU access. It cannot replace session storage or modify history,
validity or logical progress. Context commit calls the backend only after GPU
success; abandoning a submission without commit leaves the update uncommitted
and unusable. Qwen2
keeps its existing KV views and the default backend commit operation is a no-op.

The current CausalModel forward processes sessions serially and returns completed
host predictions from Qwen2's shared GPU argmax kernels. The example disables overlap; packed execution and
submission-first GPU feedback remain future optimizations.

`Qwen3_5Engine::load` loads the sharded checkpoint and tokenizer. The standalone
logits API remains available for diagnostics. Allocate one
exclusive `Qwen3_5Context` per session with `new_context(max_seq_len)`.
`step_with_logits(history, &mut context)` consumes the uncached suffix of a
complete token history and returns a greedy token ID plus final-position FP32
logits. `max_seq_len` limits the session's total token history, including prompt
and generated tokens. Histories must extend the committed prefix and remain
within this limit.
Separate contexts may execute independently; a context cannot be used with a
different engine. Standalone execution uses the loading policy; CausalModel
forwards use the policy supplied by Runner.

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
be split into arbitrary cached prefixes. Recurrent snapshots, prefix caching,
CUDA Graphs, packed batches and GPU sampling remain subsequent work; this engine does not declare W10 complete.

This is a correctness baseline: prefill uses sequential DeltaNet recurrence,
attention scans causal history, and selected-expert projections use direct FP32
reductions. Scratch is allocated per forward/layer. Submission can block on JIT/module
loading, allocations or pageable H2D; submission-first overlap and a fully warmed
gated cancellation check require workspace/input reuse and remain unvalidated. These choices need tuning before
serving performance comparisons. The vocabulary head runs only for the last
input position. Greedy selection reuses Qwen2's two-stage GPU argmax. Logits are still copied
to the CPU for finite-value validation; token feedback waits for completion.

## Memory

The downloaded checkpoint index reports 71,903,655,008 bytes including vision
and MTP. The text-only loading plan contains 693 tensors and 69,321,574,912
resident weight bytes (64.56 GiB), including FP32 norm/gating metadata.
`engine.weight_bytes()` reports the tensors actually loaded. BF16 full-attention K/V costs 20,480 bytes/token
across ten layers. Each of the thirty DeltaNet layers has 2,097,152 bytes of
recurrent state and 49,152 bytes of convolution history. At a maximum sequence length of 4096 tokens,
`config.text_config.state_bytes::<bf16>(4096)` is 148,275,200 bytes for a full
4096-token history plus recurrent state. In the Runner pool this is an occupied
state budget, not an upfront per-session KV reservation.
Weights, context storage, live scratch, the CUDA allocator and JIT overhead all
need to fit; the checkpoint's active-parameter count does not describe residency.

## Run

From the repository root, with CUDA 13.2 and `tileiras` available:

```bash
cargo run --manifest-path rust/Cargo.toml --offline --release -p engine --example qwen3_5 -- \
  --model-dir ../models/Qwen3.5-35B-A3B --prompt 'What is a language model?' \
  --max-batch-tokens 32 --max-new-tokens 16 --repeat 2
```

The example formats one user turn with thinking disabled. `--logits PATH`
optionally runs a separate standalone diagnostic prefill before Runner generation.
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
and input/capacity validation. The `pooled_state_and_runner_match_standalone`
test checks interleaved slot mappings, chunk/decode token parity, invalid peers,
capacity recovery, persistent-session continuation, repeated stateless requests
and rejection of prefix caching.
A separate GPU test checks tied router indices and selected-score normalization. Config tests reject unsupported models and FP8.

See [validation results](qwen3_5_validation.md) for measured errors, exact chunking results,
smoke output and validation limits. The engine and reference read configuration directly from the supplied model directory.

## References

- [Official model](https://huggingface.co/Qwen/Qwen3.5-35B-A3B); configuration is read from the supplied model directory.
- [Transformers Qwen3.5 MoE](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py).
- [NVIDIA TileGym Qwen3.5](https://github.com/NVIDIA/TileGym/blob/main/src/tilegym/transformers/qwen3_5/modeling_qwen3_5.py).
- Local `../sglang/python/sglang/srt/models/qwen3_5.py` for hybrid layer structure, packed projection loading and norm semantics.

## Warmed B1 latency baseline

The pooled forward benchmark keeps weights and JIT state warm, creates fresh
session state per sample, and generates a fixed number of tokens without stopping
at EOS. It times `CausalModel::forward` through completed host feedback, including
full-logit finite-value validation. Prefill includes lazy recurrent-state
initialization; model loading, pool construction, tokenization, Scheduler, and
text decoding are excluded. This is the current serial B1 path, not packed
batching or a GPU-feedback benchmark.

```bash
cargo bench --manifest-path rust/Cargo.toml --offline --locked -p engine --bench qwen3_5 --no-run
GPU_LOCAL_CPUS=24-31 # GPU-local physical cores on this RTX 6000D host
taskset -c "$GPU_LOCAL_CPUS" cargo bench --manifest-path rust/Cargo.toml --offline --locked -p engine --bench qwen3_5 -- \
  --model-dir ../models/Qwen3.5-35B-A3B --tokens 32 --output-tokens 8 --warmup 2 --iterations 3
```

JSONL records retain individual decode steps, generated IDs, forward counts and
resource-release checks. Every sample must reproduce the same generated IDs and
return pool metrics to their initial state. Run GPU jobs sequentially and freeze
the release executable before comparing independent process repetitions.
SGLang's existing static adapter in `benches/sglang_qwen2.py` also accepts this
checkpoint using text inputs and FP32 Mamba state. In SGLang 0.5.9,
`--language-only` configures encoder disaggregation and is not a switch for this
standalone text benchmark; leave it unset. Hybrid GDN on this Blackwell GPU
requires an explicit supported attention backend; the baseline uses
`--attention-backend triton`. Its GPU-sampling boundary
differs from Rust's current host/full-logit validation, so report that difference
alongside latency rather than claiming kernel parity.

For the installed SGLang 0.5.9 environment on this host, run the eager reference
separately from Rust (all commands start at the repository root):

```bash
SGLANG_PYTHON=rust/engine/artifacts/sglang-0.5.9-venv/bin/python
mkdir -p rust/profiling/qwen3_5/manual
taskset -c "$GPU_LOCAL_CPUS" env -u LD_LIBRARY_PATH \
  HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=2 \
  PATH="$PWD/rust/engine/artifacts/sglang-0.5.9-venv/bin:$PATH" \
  "$SGLANG_PYTHON" rust/engine/benches/sglang_qwen2.py \
  --model-path ../models/Qwen3.5-35B-A3B --dtype bfloat16 \
  --attention-backend triton --mamba-ssm-dtype float32 \
  --mem-fraction-static 0.9 --max-total-tokens 128 --max-running-requests 1 \
  --max-mamba-cache-size 4 --context-length 512 --disable-radix-cache \
  --batch-size 1 --input-len 32 --output-len 8 \
  --comparison-warmup 2 --comparison-iterations 3 --disable-cuda-graph \
  --result-filename rust/profiling/qwen3_5/manual/sglang-eager.jsonl
```

For a separate graph reference, remove `--disable-cuda-graph`, add
`--cuda-graph-max-bs 1`, and choose a fresh output filename. Run three independent
processes per engine and preserve each process's samples before aggregating.
The initial results, frozen executables/sources, exact sequential commands and
separate traces are in the ignored
`rust/profiling/qwen3_5/cutile-0.5.0-20261109/` directory.

The benchmark accepts `--profile` for a separate Nsight run. Use
`nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --output=PROFILE_PATH BENCH_PATH ... --warmup 2 --iterations 1 --profile`
to exclude loading and warmup. Never pool profiler timings with native samples.
