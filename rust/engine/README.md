# Minimal inference engine

A scheduler collects candidate requests; an independent model runner encodes inputs,
executes a single-head causal-attention model on the GPU, and decodes a scalar
result into text. Each token has a 16-dimensional embedding. Q/K/V projections,
attention over cached K/V, a residual, and the existing bias-free `16 -> 16 -> 1`
MLP run in f32. Sequence length is variable; the output is the last token's scalar.
There is no positional encoding, LayerNorm, sampling, or autoregressive generation.

Run from the repository root:

```bash
cargo run --manifest-path rust/Cargo.toml --offline -p engine --example export_simple_mlp
cargo run --manifest-path rust/Cargo.toml --offline -p engine
cargo test --manifest-path rust/Cargo.toml --offline -p engine
```

Re-export old MLP fixtures: the new `simple_mlp_attn` config requires embedding
and Q/K/V projection weights. Existing incompatible or corrupt directories fail
instead of silently falling back. A missing model directory still initializes
random GPU weights without saving files. Export uses fixed seeds 42 through 47.
Pass a destination directory as the export example's first argument.

The exported Hugging Face style directory contains `config.json`,
`model.safetensors`, `generation_config.json`, and `tokenizer.json`.
Generated files are ignored by Git. Tensor names and shapes are:

| Name | Shape |
| --- | --- |
| `layers.0.weight` | `[16, 16]` |
| `layers.1.weight` | `[16]` |
| `embedding.weight` | `[256, 16]` |
| `q_proj.weight`, `k_proj.weight`, `v_proj.weight` | `[16, 16]` each |

Numeric words `0..255` map to input token IDs. `input_divisor` scales embeddings
(default 1); unknown words and empty inputs are rejected. The scalar result is
spelled as output tokens for decoding. These output characters are not appended
to the input history: this demo validates attention/KV rather than text generation.

`KVPool` abstracts storage through `ContextState` and `LayerView`. `NaiveKVPool`
preallocates per-layer K/V buffers and a GPU `[context_row, token_position]` slot
table. `KVContext` tracks completed and pending token ranges. The model receives
full history plus a mutable context, computes only the uncached suffix, and uses
the Runner's policy for one mapping/write/attention/MLP submission. It commits
`ready_len` after that submission completes. Runner admits nonempty prompt deltas,
checks the full sequence length, and calls Model::validate_input on new tokens.
Model trusts idle contexts with an uncached suffix. This version does not cache
logits or replay the final token.

The Runner's private batch execution owns temporary contexts. The Runner's
private `open_session` / `close_session` methods manage history lifetimes.
Applications use `Scheduler::open_session().await` to obtain a SessionId,
`Scheduler::execute` to append prompt deltas, and `Scheduler::close_session(id).await`
to release the session. Creation returns only after Runner acknowledges success.
Session commands share the generation queue; pending batches flush before open/close,
so close cannot overtake earlier requests. None remains a temporary, stateless request.
Request execution is internal and enters through `Scheduler::execute`. Construct requests with `SessionRequest::new(session_id, input)`; the
private request ID starts as None. Scheduler assigns a fresh ID per admission
using a counter shared across clones, then forwards the assigned envelope to
Runner. Responses expose the assigned ID; internal execution trusts Scheduler
to assign unique IDs, without repeating admission checks. Execution uses
`&self` and per-session locks. Runner awaits one batch at a time, taking locks
in input order; future concurrent batch execution would require consistent lock
ordering. Scheduler guarantees unique session IDs per batch, carrying a repeated session
into the next candidate batch in FIFO order. Runner filters individual invalid
requests and splits candidates by local KV token/context capacity. Submission failures invalidate affected
sessions; ordinary validation errors preserve them. Runner uses local states and
existing mutex guards with explicit success/error cleanup. Cancellation after
taking session state releases it and leaves the entry invalid until close. `release_state` drops context
ownership; retained submission leases prevent recycling rows/slots still in use.
Packed suffixes share embedding, QKV projection, and MLP computation. One attention
launch uses per-query history metadata; K/V stores use borrowed packed slices.
Pool metrics report assigned/free bytes separately from allocated backing bytes.
Fork, truncate, prefix reuse/publication, eviction, and paged storage are deferred.

Tests compare dense CPU attention against full prefill, chunked prefill, and
single-token appends; check direct GPU KV readback, concurrent context isolation,
capacity failures, temporary/session cleanup, pending submission leases, and
single-file/indexed-shard weight round trips. Tests generate their own fixtures.

`runner/loader.rs` accepts `model.safetensors` or an authoritative
`model.safetensors.index.json` mapping names to shards. Missing or malformed
indexed shards fail. Config is deserialized into the model schema; generation
settings are optional. This custom model is not a Transformers AutoModel.

Weight loading accepts F16/BF16/F32/F64 sources and independent f32/f16/bf16
host/device destinations through `load_host::<T>` / `load_device::<T>`. Native
same-dtype bits are preserved; narrowing may round or overflow. Loader dtype
support does not establish operator support; this model executes only in f32.


Public interfaces cover application ingress, Runner construction/lifecycle,
model/pool extension traits, and loader/export operations. Runner dispatch
(`execute_batch`), decoding, and its assembly
constructor are private. BatchReply bookkeeping and NaiveLayerView layout fields
are `pub(crate)`; their types remain public for channel wiring and trait signatures.
Kernel launchers are `pub(super)` inside private kernel modules. Tests use module
privacy rules rather than widening production APIs.


ModelOutput contains request_id and a per-request result. Model::generate returns
a Vec of these outputs, including an error on every affected request if shared GPU
execution fails. BatchReply also contains only per-request results; Scheduler routes
each success/error independently. Invalid input never fails otherwise valid peers.
Admission uses KVPool::can_append as a conservative capacity snapshot; reservation
remains the pool's responsibility. Workspace-based batching is deferred.


Naive K/V writes use `[8, head_dim]` tiles, with masked tails. Layer views take a
backend state and logical token range rather than the owning KVContext; the pool
maps that range to its physical layout. The Naive attention backend supports
reading completed-cache windows as well as newly appended suffixes. The grouping
change has correctness coverage; no throughput improvement is claimed.
