# Minimal inference engine

A scheduler collects candidate requests; an independent model runner encodes inputs,
drives batched next-token steps on the GPU, and decodes generated tokens into text. Each token has a 16-dimensional embedding. Q/K/V projections,
attention over cached K/V, a residual, and the existing bias-free `16 -> 16 -> 1`
MLP run in f32. The toy backend rounds/clamps its scalar output into token IDs
0..255; it exercises autoregressive history/KV management without a vocabulary head.
Qwen2.5 provides the real language-model baseline. Only greedy generation is supported.

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
rounded/clamped to a numeric token ID. Runner appends every generated ID to history.
Default generation is one token; requests may set max_new_tokens and EOS overrides.

`KVPool` abstracts storage through `ContextState`, `ReadView`, and `WriteView`. `NaiveKVPool`
preallocates per-layer K/V buffers and a GPU `[context_row, token_position]` slot
table. `KVContext` tracks completed and pending token ranges. The model receives
full history plus a mutable context, computes only the uncached suffix, and uses
the Runner's policy for one mapping/write/attention/MLP submission. It commits
`ready_len` after that submission completes. Runner admits nonempty prompt deltas,
checks the full sequence length, and calls CausalModel::validate_input on new tokens.
CausalModel trusts idle contexts with an uncached suffix. This version does not cache
logits or replay the final token.

The Runner's private batch execution owns temporary contexts. The Runner's
private `open_session` / `close_session` methods manage history lifetimes.
Applications use `Scheduler::open_session().await` to obtain a SessionId,
`Scheduler::execute` to append prompt deltas, and `Scheduler::close_session(id).await`
to release the session. Creation returns only after Runner acknowledges success.
Scheduler forwards immediately. Clients serialize turns of one session and await
completion before its next request or close. Runner rejects accidental overlap
with a busy-session error; it does not queue turns or wait for session locks.
None uses temporary, stateless state. Construct requests with `SessionRequest::new(session_id, input)`;
Scheduler assigns fresh request IDs across clones and responses expose those IDs.

Runner performs one packed forward at a time. At each step boundary it admits
waiting requests, lets new prefills join ongoing decodes, and removes finished
requests. It holds one private ActiveRequest per turn with the existing session
state/guard and library text decoder. Try-locks reject busy sessions without waiting.
Admission accounts for active requests' future decode slots, as well as the new
prompt/output budget and context rows. Capacity-limited work waits for active
requests to finish; requests that still cannot fit fail individually. This is
capacity waiting across requests, independently of client-side session serialization.

Generated history is restored and temporary KV released before per-request EOF.
Validation errors preserve session state; changed/pending KV on a model failure is
invalidated. Cancellation drops active local states, and GPU submission leases
keep storage alive until in-flight accesses finish. Dropping an output stream
still stops delivery rather than cancelling generation.

The serial control flow follows SGLang's normal loop (local source revision `0263fdacb2`):
`event_loop_normal` receives work, `get_next_batch_to_run` selects/merges requests,
then one forward runs and results update the running batch. Our existing packed
backend permits prefills and decodes in the same forward. Token-budgeted chunked
prefill is implemented; preemption and Qwen CUDA Graph remain deferred. The
implemented one-round submission-first overlap is described below. Fixed-cohort
performance results do not establish mixed-arrival scheduling benefits.

`Qwen2Engine::kv_pool_config` defaults to one context row. Applications enabling
concurrency must size their pool accordingly, for example:

```rust
let mut config = model.kv_pool_config(max_seq_len)?;
config.max_rows = 2;
config.num_slots *= config.max_rows;
```

Rows also include retained idle sessions. The Runner uses available rows/slots
for admission; there is no separate fixed GPU batch-size limit in Scheduler.

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
(`drive`), decoding, and its assembly
constructor are private. Tests collect the same RunnerReply events as Scheduler. Read/write view
fields are private; their types stay public for trait signatures. Read views expose
const pointers through safe accessors; writable handles require a distinct WriteView.
KVContext selects only pending positions for write views and holds an exclusive
borrow until they are consumed or dropped. GPU storage and leases remain retained
in exec until submission completion, independently of these CPU borrows.
Kernel launchers are `pub(super)` inside private kernel modules. Tests use module
privacy rules rather than widening production APIs.


CausalModel::next_tokens returns one ordered Result<u32, ModelError> per input, including
an error on every affected request if shared GPU execution fails. Request IDs stay
in Runner rather than entering the model interface. RunnerReply separates text deltas
from per-request completion/error statuses; Scheduler routes both independently. Invalid input never fails otherwise valid peers.
Admission uses KVPool::can_append as a conservative capacity snapshot; reservation
remains the pool's responsibility. Workspace-based batching is deferred.


The standalone Naive store uses `[8, head_dim]` tiles with masked tails. SimpleMLP
uses a fused QKV projection instead: Q goes to its temporary tensor, while K/V
accumulators scatter directly into cache via WriteView metadata. Padding lanes
never read slot-table entries or write cache; no temporary K/V tensors are allocated. Read/write views take a
backend state and logical token range rather than the owning KVContext; the pool
maps that range to its physical layout. The Naive attention backend supports
reading completed-cache windows as well as newly appended suffixes. The grouping
change has correctness coverage; no throughput improvement is claimed.

## Uniform attention-MLP profiling

`examples/simple_mlp_attn.rs` runs uniform stateless requests through Scheduler and
Runner using fixed-seed random weights, so it does not depend on exported model files.
It checks identical finite outputs and verifies KV release after each batch. The
example-only model wrapper measures CausalModel::next_tokens separately from end-to-end time.

```bash
cargo build --manifest-path rust/Cargo.toml --offline --release -p engine --example simple_mlp_attn
rust/target/release/examples/simple_mlp_attn --warmup 10 --iterations 50
mkdir -p rust/profiling/simple_mlp_attn
nsys profile --trace=cuda,nvtx,osrt --gpu-metrics-devices=0 --capture-range=cudaProfilerApi --capture-range-end=stop --sample=none --cpuctxsw=none --event-sample=none --backtrace=none --force-overwrite=true --output=rust/profiling/simple_mlp_attn/uniform-b5-t16 rust/target/release/examples/simple_mlp_attn --warmup 10 --iterations 50 --profile
```

The defaults are five requests, sixteen tokens per request, and one execution
stream. `--batch-size` accepts 1..5 and `--tokens` accepts 1..512. Smaller batches
are forwarded immediately; Runner forms batches from ready work at each step boundary.
Profiler start/stop excludes weight/pool initialization, JIT, and warmup. CUDA
tracing selects hardware trace where supported; GPU metrics sample device 0.
Profiling can perturb timings, so compare wall times without the profiler.
NVTX ranges label the synchronous `prepare_append_batch`, `submit_batch`, and
`commit_append` calls. Prepare/submit occur per batch; commit occurs per request.
The Naive pool reserves slots on CPU and returns its concrete `NaiveMappingPlan`.
This deferred-upload API is specific to Naive; the public `KVPool` trait only
exposes the ordinary preparation contract.
The model packs that plan's payload with token IDs and shared query metadata,
then performs one upload before bind and model kernels. The single-context
prepare API and automatic batch prepare API remain available.
These ranges measure CPU work and submission, not GPU completion or the outer await.
On the packed path, prepare covers CPU reservation only; submit includes upload
and bind, so compare combined phases or full model latency across versions.

The attention MLP reuses six floating-point scratch buffers per workspace.
A submission exclusively checks out an exact-shape workspace for its device;
in-flight ownership and tensor handles prevent reuse until completion.
The idle cache retains at most four workspaces. GEMM/matvec start accumulators
at zero, so reused output buffers need no per-batch clearing. Input/metadata
uploads use a separate integer buffer pool (at most four idle buffers per pool).
KV mapping updates, token IDs, and shared query metadata occupy separate views
of one reused device upload allocation, with no device-side copy between views.
QKV and attention share `[context row, absolute position, history start]`
metadata; attention derives its causal end as position + 1. The packed batch
upload is submitted before bind and embedding. Eager execution retains its
pageable host readback.

CUDA Graph replay is optional: enable it with
`SimpleMLPEngine::random(0).await?.with_cuda_graphs()` (or after `load`/`new`).
The profiling example accepts `--cuda-graph`; experiment commands and measurements
belong in the local results file linked below. Eager execution remains the default.
Each graph owns fixed GPU input, five GPU scratch allocations, and a mapped
page-locked host output. It captures bind plus the seven model kernels through
`CudaGraph::scope`, borrowing ordinary tensor handles for each operation. Eager
execution and capture share the same kernel definitions, with capture restricted
to `GraphNode` operations. The final matvec writes its output through the host
allocation's CUDA device alias, eliminating a separate DtoH command for these
small scalar outputs. This still transfers data over the interconnect; it does
not move intermediate computation into host memory.

Each replay uploads the packed payload using the original pageable HtoD path,
then launches once. It polls for completion inline for at most 2 ms before
falling back to the existing DeviceFuture completion mechanism. This favors
latency for short graphs at the cost of occupying the calling CPU thread during
polling; no global runtime setting is changed. After outer completion, the CPU
copies the mapped result into the returned Vec and commits KV progress. A result
lease keeps the mapped storage and instance alive until consumption. CPU
reservation and upload remain outside capture. Eager execution is unchanged.

Graphs are captured lazily for a packed shape, KV backing store, and weight
identity. A new instance allocates its private buffers and warms kernels on a
dedicated stream before capture, so its first use has an extra initialization
cost. Warmup repeats the current append's K/V writes without committing progress.
Changes in token IDs, slot IDs, context rows, and history positions only change
the uploaded payload and reuse the same graph when its geometry is compatible.
An in-flight batch exclusively owns its graph until completion; concurrent batches
use separate instances. The cache keeps at most four idle instances and retains
GPU storage, never per-request KV state leases. Pool/weight replacement or shape
changes select a separate instance. This implementation uses exact shapes rather
than padding requests into a fixed set of graph buckets.

The initial [results and commands](../profiling/simple_mlp_attn/RESULTS.md) include
three unprofiled runs, CUDA kernel/API summaries, and a directly openable
[Nsight Systems report](../profiling/simple_mlp_attn/uniform-b5-t16.nsys-rep).
These local profiling artifacts are ignored by Git. The example uses libloading
only as a dev dependency to resolve CUDA profiler APIs without a WSL linker path.

## Qwen2.5-0.5B-Instruct: single-request correctness baseline

`Qwen2Engine` loads a local Hugging Face checkpoint using the existing WeightLoader.
`Qwen2Engine<T>` and `NaiveKVPool<T>` use matching storage types. The Qwen CLI
and benchmarks default to `--dtype bf16`; `--dtype f32` retains the reference path.
BF16 checkpoint weights load directly as BF16, preserving their bits without
expanding the model to FP32. All numerical operators use cuTile: embedding, checkpoint-layout linear projections,
RMSNorm, Q/K bias and split-half RoPE, causal GQA over mapped KV slots, residual,
SwiGLU, the tied/untied LM head, and two-stage GPU greedy argmax. It does not invoke
PyTorch, cuBLAS, or a Python model for inference. The optional Python script is an
independent Transformers reference (CPU/FP32 by default, or BF16/CUDA with SDPA).

| Storage / calculation | Precision in BF16 mode |
| --- | --- |
| Weights, activation workspace, K/V and read/write views | BF16 |
| Linear MMA operands / accumulator | BF16 / FP32 |
| RMSNorm reduction, SiLU, attention scores and online softmax | FP32 calculation |
| RoPE frequencies and trigonometry | FP32; products use activation dtype |
| LM-head logits and argmax maxima | FP32 |
| Token IDs, mapping table, Scheduler/Runner APIs | Unchanged integer types |

`KVPool::Element` ties `store` and `AttentionBackend` tensor types to the pool at
compile time. FP32 scalar constants (including negative infinity) stay inside
FP32 reductions; no generic infinity trait or block scaling is needed. Activations
are rounded back to T at operator boundaries; norms and SiLU preserve Qwen's
intermediate activation rounding. Attention currently widens gathered K/V tiles
inside the kernel for FP32 scores/weighted reductions; this is not a full-cache
conversion or an optimized BF16 attention MMA implementation. Scratch admission
and ownership metrics count T storage plus the separate FP32 logits/maxima and
integer buffers. BF16 does not mean INT8/INT4/FP8 quantization.

Use matching reference precision and backend for numerical checks:

```bash
HF_HUB_OFFLINE=1 ~/.venvs/py3.12/bin/python rust/engine/examples/qwen2_reference.py "$QWEN2_MODEL_DIR" --dtype bf16 --device cuda --attention sdpa --max-new-tokens 8 --output-dir rust/engine/artifacts/qwen2-bf16
cargo run --manifest-path rust/Cargo.toml --offline --locked --release -p engine --example qwen2 -- --model-dir "$QWEN2_MODEL_DIR" --dtype bf16 --max-new-tokens 8 --reference-dir rust/engine/artifacts/qwen2-bf16
```

The example checks finite logits, a dtype-dependent logit tolerance and exact
first-token/text agreement for the supplied fixture. Arbitrary BF16 greedy output
need not match FP32 token-for-token. It logs max and RMS logit error before asserting;
these are correctness checks, not a generation-quality evaluation.

Release validation passed 84 engine tests and 3 compile-fail doc tests; the opt-in
checkpoint test stayed ignored, with real weights checked separately via the CLI.
English/Chinese BF16 SDPA fixtures, cached/ragged/chunked steps and Runner serial/
overlap checks passed. At batch 4 with 4096 input tokens per request, per-forward
scratch peaked at 638 MiB (BF16) vs 1274 MiB (FP32). See the local
[BF16 report](../profiling/qwen2/bf16/RESULTS.md) for raw timings, pool accounting
and reference errors. BF16 eager has different rounding boundaries and fails the
same 0.5 max-error fixture threshold; it is not claimed numerically equivalent.


A single `CausalModel<P>` trait exposes `next_tokens(&mut [ModelInput], policy)`.
Models own one numerical step and EOS recognition; all generation loops live in Runner.
Single-input callers use a stack array; results remain an ordered Vec of per-input results.
Models expose their matching tokenizer through `CausalModel::tokenizer()`.
RunnerBuilder's model factory returns only the model; Runner borrows its tokenizer
for encoding and decoding.

`Scheduler::execute_stream` admits a request and returns a
`Stream<Item = Result<LLMResponse, EngineError>>`. Each response contains a text
delta and the same request ID. Runner uses `tokenizer.decode_stream(true)` directly;
the library owns buffering, UTF-8 assembly, and context-sensitive decoding. Runner
forwards `Some(text)` and keeps generating on `None`. There is no application-level
flush or replacement-character synthesis. If generation stops with an undecodable
fragment, the response contains only chunks already emitted by the library; it
may differ from batch decoding's replacement characters. Token history is preserved.
Finished carries completion/error status and the actual generated-token count for one request.
Finished is emitted after that request's state cleanup, before any peer's next step.
The execution methods return unit and send all results through the reply channel.
An error terminates that request's stream; earlier deltas remain delivered. `execute` collects the same stream into
one response. Output channels are buffered; dropping the stream stops delivery
without cancelling generation.

Scheduler observers can be attached with
`Scheduler::new_with_observers(cancel, commands, replies, Vec<Arc<dyn SchedulerObserver>>)`.
The existing `new` constructor attaches none. Clones share observers. Synchronous,
nonblocking hooks observe admission, text deltas and completion outside pending-map
locks; they do not return scheduling decisions. `[T]` implements collection notification and `Arc<T>` forwards to its inner value.
Scheduler stores `Arc<[Arc<dyn SchedulerObserver>]>`; one method call notifies the
collection. Vec method calls can dereference to the slice implementation. Completion-only closures implement the same trait directly. Admitted requests finish once
on success, validation/execution error, shutdown or transport failure, even if the
consumer dropped its stream. Fatal Runner transport failure closes admission;
already-routed terminal replies remain readable. Shutdown still requires `wait_stopped`.

Completion supplies a named `RequestStats` struct: admission-to-finish `elapsed`,
optional first **nonempty** text latency, and Runner's actual generated-token count
(including EOS and tokens not yet decoded into visible text). Interrupted requests
without a terminal Runner report have an unknown token count (`None`), not zero.
Request TPS is generated tokens divided by total request time; it includes queueing,
prefill, generation and text decoding, and differs from model-only decode TPS.
Time ends at server-side completion, without waiting for a slow consumer to poll.

`MetricsObserver::default()` emits `engine_requests_total`, failures,
generated-token/text-byte counters, an in-flight gauge, request-duration histograms labelled
by success/error, first-text latency, and successful-request output TPS. Empty text
does not record a first-text observation. Failed/unknown-count requests do not
record successful TPS. These metrics aggregate across attached schedulers, without
request/session labels. Construct the observer after installing the application's
recorder; no recorder/exporter is installed here. The Qwen example attaches this
plugin and a completion-only closure for its JSON summary, replacing local timers.
Exact queue time is not reported because Runner does not yet emit a start event.

```rust,ignore
let scheduler = Scheduler::new_with_observers(
    cancel, commands, replies,
    vec![Arc::new(MetricsObserver::default())],
);
```

```rust,ignore
use futures::TryStreamExt;

let chunks = scheduler.execute_stream(request)?;
futures::pin_mut!(chunks);
while let Some(chunk) = chunks.try_next().await? {
    print!("{}", chunk.output);
}
```


Runner drives batch steps, appends generated IDs to each session's history, removes
finished requests from later steps, and stops at EOS or max_new_tokens (default 1).
`RunnerBuilder::with_max_batch_tokens(NonZeroUsize)` limits uncached input tokens
per forward (default 512). Runner's internal `BatchPolicy::plan` returns borrowed
`ScheduledRequest { request, compute_end }` entries. `MixedChunkPolicy` reserves
one token per existing decode, fills the rest with prefill chunks in active order,
and executes them together. Without prefill work/budget, it executes decode alone.
A turn enters decode after its first generated token; a one-token prompt remains
prefill. Admission and result/KV settlement stay in Runner. Requests that do not fit wait
until a later forward. Each prefill chunk submits all model layers on the ordered compute stream; only the final chunk appends/emits a generated token and consumes max_new_tokens.
Intermediate chunks compute and discard the LM-head prediction without starting
Qwen's deferred D2H readback.
Accepted CPU history stays intact; Runner shares an Arc history prefix ending at
the chunk boundary, avoiding a history copy for owned submission. Session continuation also computes any uncached last output token first.
The token budget bounds packed scratch rows, not cached-history attention latency;
it does not replace full-turn KV admission or implement preemption.
The Qwen example exposes the same setting as `--max-batch-tokens`, for example
`--max-batch-tokens 7 --max-new-tokens 8 --reference-dir rust/engine/artifacts/qwen2`
to force several chunks while checking the reference response.
Admission budgets include all possible generated KV positions, excluding the final
sampled token whose KV will be computed in a subsequent turn. Only pure greedy
settings are supported. Mid-turn cancellation releases contexts; partial failed
turns are invalidated if KV has advanced. Completed history survives text-decode errors.
The Qwen example submits through Scheduler -> Runner; optional reference mode still
reads diagnostic logits directly. Exact generated IDs and multi-turn Runner continuation
are checked by the real-checkpoint regression. The CLI response contains text and request_id.

Qwen packs uncached token suffixes into one GPU forward, supporting ragged prefill,
different cached-prefix lengths and batched decode. Linear/FFN and GQA process all
packed rows; RoPE/attention share per-token context/position metadata uploaded once
per forward. KV writes retain each context's pending-only store contract. Final
RMSNorm gathers each request's last packed row, then LM-head/argmax produce one
token per request in a single D2H readback. Invalid inputs are filtered before
submission; valid peers stay in one batch and results retain the original order.
Append batches require one KV pool. SimpleMLP retains CUDA Graph replay; Qwen is
eager. ModelInput accepts BorrowedHost(history_prefix), SharedHost { tokens: Arc<Vec<u32>>,
end } or Device(token). BorrowedHost retains the zero-copy slice API for direct
callers; Runner uses SharedHost so owned tasks share history without copying it.
Input storage does not determine output storage: Qwen's CPU inputs still produce
NextToken::Device predictions. `CausalModel::forward` commits GPU work/KV and returns
NextToken::Device for Qwen; its shared readback future is independent of the GPU
feedback handle. `next_tokens` still returns CPU u32 values for direct callers.

Mixed/reordered inputs use one embedding_gather launch for the packed batch. Its
cached u64 source table stores [GPU token address, inline host token ID] per row;
address=0 selects the inline ID. Host suffixes need no separate ID upload on this
path, and device IDs are read directly without an intermediate token gather.
ExecutionContext retains every addressed source allocation through completion.
This trades per-request launches for one 16-byte-per-row metadata upload. Host-only
and contiguous device-only batches keep their existing single-launch fast paths
without that table. The local [launch-count comparison](../profiling/qwen2/embedding-batch/RESULTS.md)
records the measured scope and limits.

Runner uses submission-first overlap with one round of lookahead: enqueue F(N),
then complete/commit/read/process F(N-1). It does not wait for F(N) before planning
the next iteration. All forwards use one ordered compute stream; D2H uses a separate
stream after producer completion. GPU feedback never requires a CPU ID/H2D round trip.

CausalModel::enqueue_forward_and_return_completion is the Runner entry. Qwen temporarily moves selected
KVContext owners into an owned DeviceOp, builds borrowed ModelInput views inside
execute, and returns contexts/predictions through a handoff after the normal
DeviceFuture's first poll. That poll registers completion before the next submit;
the library continues to retain/release GPU resources. No cuda-async modifications,
manual completion waiter, or dependency upgrade are required. The default method
wraps an already completed forward, preserving SimpleMLP and other implementations.
The generic SubmittedGpuWorkWithCompletion<T> holds both helper handoff output and
model predictions, replacing separate Submission/SubmittedForward containers. Runner
combines the former BatchMember/PendingBatch data into per-request records and calls
wait_commit_kv_and_forward for GPU wait, KV commit and token processing.
Direct forward still awaits GPU completion and commits KV; next_tokens then reads
CPU values. Models never implement forward by calling next_tokens.

KV allocated_len is the reservation cursor; ready_len advances only when the matching
completion succeeds. A FIFO permits at most two pending appends, and GpuTokenPredictionHandle
carries its absolute position. Runner retains a Vec<RequestForwardAwaitingGpuCompletion>;
each record owns identity, KV append end, prediction and shared GPU completion rather
than request borrows. Only the newest append is writable, and FIFO commits cannot
publish a later append first. Admission budgets already reserved positions once.

EOS may cause one extra submitted step: it drains and its prediction is discarded.
EOF follows cleanup. The final max_new_tokens output is settled without another
append, preserving the original KV-capacity budget. No dummy token enters CPU history.
RunnerBuilder::with_overlap(false), or Qwen --disable-overlap, provides a serial
baseline: next_tokens returns CPU IDs, accepted in the same iteration without
pending completion records. Only the overlap path retains pending results; the
setting selects the path without inspecting prediction variants. Host-only
SimpleMLP examples disable overlap explicitly. Cancellation uses cuda-async's existing drain-on-drop safety, which can
block the host while in-flight GPU work finishes.

The shared model/readback module provides GpuTokenReadback: an independent D2H
stream, reusable pinned cache and deferred GPU token handles. The common submission
method CausalModel::enqueue_append_batch validates feedback stream order, prepares
KV and restores context ownership; models supply their own kernels and instrumentation.
SubmittedGpuWorkWithCompletion::submit constructs the underlying submission handle.
The matching execute_append_batch method instead awaits completion and commits KV.

Token/logit readbacks enqueue asynchronous D2H into reusable pinned buffers and become
readable only after outer completion. Dropping an in-flight model step may require
context release; retained GPU leases prevent premature storage reuse.
Qwen uses an exclusive workspace keyed by packed token count, batch size and
device for each eager forward. Two backing allocations hold disjoint float/integer regions; all decoder
layers reuse them in stream order. Every output is fully overwritten, so scratch
needs no zero-fill kernel. The lease survives submission failure, cancellation,
and escaped tensor handles; concurrent forwards receive different workspaces.
The cache retains at most two idle workspaces (normally prefill/decode) and drops
idle shapes under memory pressure. On a miss, scratch admission checks its exact
size against device free memory plus unused execution-pool reservations after
weights/KV allocation. This snapshot cannot reserve memory against other callers;
allocation errors still propagate as per-request failures.
For Qwen2.5-0.5B, scratch bytes are
`4 * (packed_suffix_tokens * 20352 + batch_size * 154021)`: for batch size 1, about 3.07 MiB for 32 tokens and
318.59 MiB for 4096 tokens, independent of the decoder layer count. These figures
exclude weights, KV, input uploads, pinned readbacks, and other idle/in-flight
workspaces. Qwen CUDA Graph capture remains subsequent work.
This is a numerical/lifecycle baseline, not an optimized throughput result. Initial
support requires full causal attention, unscaled RoPE, and 64-dimensional heads.
Qwen emits standard `metrics` instruments without extending `CausalModel`:

| Metric | Type |
|---|---|
| `qwen2_steps_total`, `qwen2_input_tokens_total` | Counter, labelled `phase=prefill/decode` |
| `qwen2_failed_steps_total` | Counter |
| `qwen2_step_wall_duration_seconds`, `qwen2_host_submit_duration_seconds` | Histogram, labelled by phase |
| `qwen2_batch_duration_seconds`, `qwen2_batch_size` | Histogram |
| `qwen2_readback_wall_duration_seconds` | Histogram; independent token D2H submission through completion |
| `qwen2_workspace_forward_bytes` | Histogram |
| `qwen2_workspace_{cache_hits,cache_misses,allocations,checkout_failures,evictions}_total` | Counter |
| `qwen2_workspace_*_bytes`, `qwen2_workspace_*_workspaces` | Gauge for current/peak ownership |

Every instrument carries a `model_instance` label identifying one loaded model
lifetime, so concurrent model instances cannot overwrite each other's gauges.
Request/session IDs are never labels. Handles register once at model/pool creation
and are reused through execution and release. Workspace gauge publication follows
the pool's locked transition order; destruction sets current resource gauges to
zero while retaining historical peaks/counters. Scheduler request/first-text times
are histograms in its MetricsObserver; benchmark CUDA pool peaks are gauges.

No exporter, HTTP endpoint, external service or global recorder is installed.
Applications may install their recorder **before loading the model**. Without one,
`metrics` handles are no-ops; cached handles do not switch recorders afterwards.
Tests use `metrics-util::debugging::DebuggingRecorder` as a dev-only in-memory
recorder to check actual counters, gauge transitions and histogram observations.

Qwen execution totals are maintained only by Counter/Histogram handles. There is
no `Qwen2Metrics` accumulator or per-engine last-step timing: simultaneous batches
share those instruments, and their overlapping durations must not be interpreted
as elapsed wall time for an individual batch. Each completed packed forward adds
one step and one duration observation, with input-token counts summed across its
requests. Decode means every input adds one token to existing KV; a mixed batch
is labelled prefill if any input has empty KV or a multi-token suffix.
For Qwen's forward path, `step_wall_seconds` spans CPU preparation/submission,
GPU compute completion and KV commit; D2H is measured separately by
`qwen2_readback_wall_duration_seconds` and the `readback_completed` tracing event.
The direct diagnostic logits path still includes its same-submission D2H.
Scheduler request timing includes CPU observation/decoding and lookahead draining;
overlapping phase durations must not be added to infer request wall time.
`host_submit_seconds` is returned in the private
`StepOutput` for that invocation: host wall time inside submit, including scratch,
input uploads, JIT and CUDA calls, without awaiting GPU completion. It excludes
KV mapping preparation and later host output access. GPU execution can overlap
host submission, so subtracting them does not yield pure GPU kernel time. Failed
submitted forwards increment the failure counter and have no successful duration
sample; rejected inputs do not produce a GPU-step observation. Failed-step
tracing leaves that duration unavailable. Cold JIT,
warmup and diagnostic observations are included; cancelled futures do not emit
completed-step observations.

`Qwen2Engine::workspace_metrics()` returns only current/peak pool ownership,
independent of a recorder: in-use/idle/resident bytes, current workspace counts and
maximum single-forward allocation. These are shared **model-wide** resource states,
not one batch's result. Sizes exclude allocator padding/reservations. GPU leases
and escaped handles delay recycling. Cache hits/misses, allocations, failures and
evictions increment Counter handles directly, without duplicate snapshot counters.

The benchmark times each of its own awaited calls. It does not subtract shared
engine counters before/after a batch. Its `workspace_pool` JSON field is explicitly
a model-wide resource snapshot and can include other concurrent work; it is not a
per-batch allocation or lifetime delta. Internal submission timing is available as
a histogram (and structured tracing), not reconstructed from engine totals.

`qwen2 --trace-steps` enables structured `tracing` JSON events on stderr for steps,
batches and scratch checkout/return. The library does not install a subscriber;
logs are disabled unless the caller enables one. Keep detailed logging off when
sampling latency. The main Qwen example also reports `time_to_first_text_seconds`
from Scheduler observers (null if no visible text), end-to-end
elapsed time, actual generated tokens and request output TPS. This includes queueing/tokenization/text decoding and differs from
model-only prefill latency. The Qwen-only benchmark bypasses Scheduler/text decoding to
isolate model execution and creates fresh KV for every warmup/measured sample:

```bash
cargo bench --manifest-path rust/Cargo.toml --offline -p engine --bench qwen2 --no-run
mkdir -p rust/profiling/qwen2
cargo bench --manifest-path rust/Cargo.toml --offline --locked -p engine --bench qwen2 -- --model-dir "$QWEN2_MODEL_DIR" --tokens 4096 --decode-steps 8 --warmup 1 --iterations 3 --validate-chunked > rust/profiling/qwen2/t4096.jsonl
```

It uses deterministic repeated token IDs, reports prefill/decode tokens/s,
per-step decode p50/p95, CUDA default-pool used/reserved high-water marks and
workspace metrics. Warmup samples are labelled and excluded from the summary.
Full/chunked validation runs afterwards and compares every final-position logit,
next-token IDs and KV release. CUDA pool metrics include weights/KV/workspace but
exclude other processes and non-pool CUDA allocations; they are not device-wide
peak VRAM. The default three measured iterations are a diagnostic sample, not a
statistically robust throughput study.

For Nsight Systems, add `--profile` to start capture after warmup and stop it before
optional validation. Qwen's synchronous NVTX submit ranges label prefill/decode;
they do not span the GPU-completion await. Use the optimized executable path
printed by the benchmark build:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --event-sample=none --backtrace=none --capture-range=cudaProfilerApi --capture-range-end=stop --output=rust/profiling/qwen2/packed-nsys/trace-b1-t32 BENCH_PATH --model-dir "$QWEN2_MODEL_DIR" --tokens 32 --batch-size 1 --decode-steps 8 --warmup 3 --iterations 3 --profile
```

The local [packed Qwen profiling report](../profiling/qwen2/packed-nsys/RESULTS.md)
keeps unprofiled wall times separate from correlated kernel/API measurements.
The first attention optimization gathers 32 logical KV slots per iteration and
uses occupancy target 2, preserving the existing KV layout and lifecycle. This is
a compute tile, not a cache allocation page. The local
[attention optimization report](../profiling/qwen2/attention-tiling/RESULTS.md)
compares release runs and hardware counters. Subsequent
[batch tuning](../profiling/qwen2/batch-tuning/RESULTS.md) selects 16x64 linear
output tiles for at least 16 packed rows, retaining 4x32 tiles for smaller inputs
and the LM head at batch sizes 1/2/4. On SM120, linear targets occupancy 2 and
attention targets 4; both hints live on their kernel definitions. No additional GPU scratch is required.
Split-KV decode remains a candidate; Qwen CUDA Graph is a later submission-cost experiment.

The example supplies the explicit two-message Qwen2 chat template and uses pure
greedy decoding (no repetition penalty or stochastic sampling).

```bash
# Point to an existing complete HF snapshot; no model download is performed.
QWEN2_MODEL_DIR=/home/po/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
cargo run --manifest-path rust/Cargo.toml --offline --release -p engine --example qwen2 -- --model-dir "$QWEN2_MODEL_DIR" --max-new-tokens 32

# Independent reference: use an interpreter with installed CPU-capable torch/Transformers.
env -u LD_LIBRARY_PATH PYENV_VERSION=3.14.3 python rust/engine/examples/qwen2_reference.py "$QWEN2_MODEL_DIR" --max-new-tokens 32 --output-dir rust/engine/artifacts/qwen2
cargo run --manifest-path rust/Cargo.toml --offline --release -p engine --example qwen2 -- --dtype f32 --model-dir "$QWEN2_MODEL_DIR" --max-new-tokens 32 --reference-dir rust/engine/artifacts/qwen2

# Optional real-checkpoint regression: full/chunked prefill logits, decode and release.
QWEN2_MODEL_DIR="$QWEN2_MODEL_DIR" QWEN2_REFERENCE_DIR="$PWD/rust/engine/artifacts/qwen2" cargo test --manifest-path rust/Cargo.toml --offline --release -p engine real_checkpoint_matches_reference_and_chunked_kv -- --ignored --nocapture
```

On revision `7ae5576`, CPU torch 2.9.1 / Transformers 5.3.0 FP32 eager references
and cuTile greedy IDs matched for English and Chinese prompts. First-position
vocabulary-wide max absolute logit errors were approximately `4.53e-5` and `3.19e-5`.
The English run stopped at EOS after ten generated IDs and produced
`Hello! How can I assist you today?`. The example asserts token agreement,
logit error below `1e-3`, and KV allocation release. Local logs and reference data
are under `rust/engine/artifacts/qwen2*` (ignored by Git). Timings include eager
allocation and possible JIT and must not be interpreted as steady-state serving latency.

### Runner serial/overlap benchmark

[benches/qwen2_runner.rs](benches/qwen2_runner.rs) measures real Scheduler -> Runner
requests, including tokenization, incremental text decoding and response delivery.
Its bench-only model wrapper disables EOS stopping to produce a fixed output budget;
it delegates all numerical computation to Qwen. Repeated ID 9707 is decoded and
round-trip checked against the real tokenizer before timing. Uniform requests are
admitted before yielding; assertions check full GPU batch shapes, exactly one full
prefill plus output_tokens-1 decodes, stable output text/counts and KV/scratch release.
The token budget fits the entire prefill, so this experiment excludes chunking and
continuous arrivals. Each warmup and measured cohort starts with fresh KV.

```bash
cargo bench --manifest-path rust/Cargo.toml --offline --locked -p engine --bench qwen2_runner -- --model-dir "$QWEN2_MODEL_DIR" --dtype f32 --mode serial --tokens 32 --batch-size 4 --output-tokens 64 --warmup 2 --iterations 3
cargo bench --manifest-path rust/Cargo.toml --offline --locked -p engine --bench qwen2_runner -- --model-dir "$QWEN2_MODEL_DIR" --dtype f32 --mode overlap --tokens 32 --batch-size 4 --output-tokens 64 --warmup 2 --iterations 3
/usr/bin/python3 rust/engine/benches/qwen2_runner_compare.py --model-dir "$QWEN2_MODEL_DIR" --profile
```

The comparison script freezes the release/bench binary and records commands/source
provenance. Defaults alternate serial/overlap order across three independent process
runs, batch 1/2/4 and input 32/4096. Every process has two warmups and three measured
cohorts; Nsight captures are separate, with one measured cohort after warmup. A new
results directory under rust/profiling/qwen2/runner-overlap preserves raw JSONL,
comparison.json, nsys-rep/SQLite traces and traces.json. See the benchmark header for
a direct Nsight command or pass --results-dir to choose a new experiment directory.

Cohort wall time/output TPS includes ingress and text delivery; observer first-text
time is distinct from pure prefill and first-token timing. CUDA-pool high-water marks reset per cohort and
exclude other processes/non-pool allocations. Workspace peaks are cumulative model
resource snapshots. Do not derive pure decode time by subtracting first-text time.
Profiler output validates submission/readback ordering and measures CPU accept/D2H
intersection with GPU kernel intervals; the GPU activity union is application-trace
coverage, not SM occupancy. Only unprofiled independent runs determine speedup.
This is a Runner A/B test, separate from SGLang's GPU-token static model benchmark.

The local [Runner overlap report](../profiling/qwen2/runner-overlap/RESULTS.md) records
36 unprofiled process runs and 12 separate Nsight captures. Full next-submission
completion precedes prior D2H in every overlap adjacent pair (63/63 per capture),
versus 0/63 in serial. Short batch 2 shows the clearest native gain (about 8%);
other configurations are small/noisy or flat. This is not a SGLang performance claim.


## BF16 static-model comparison with SGLang

Run the sequential release matrix from the repository root:

```bash
/usr/bin/python3 rust/engine/benches/qwen2_sglang_compare.py --model-dir "$QWEN2_MODEL_DIR" --include-sglang-graph
```

The primary comparison is eager vs eager, with SGLang decode CUDA Graph reported
separately. Both use BF16 weights/activations/KV, token ID 9707, fresh KV, batch
1/2/4, 32/4096 input tokens, and 64 outputs regardless of EOS. The default matrix
reserves 16640 KV token slots in both engines. Two warmups and three measured
samples are repeated in three independent processes; engine order rotates.

`benches/qwen2.rs --gpu-feedback` reuses forward and GPU prediction handles without
per-step D2H. A separate untimed host-feedback control checks final IDs. Only the
current predictions survive a step, avoiding retention of all previous scratch
buffers. Final IDs are read after timing. Without this flag the original benchmark
still includes token readback on every step. Both static timers include CPU submission
and GPU completion; Rust input-view construction and final readback are outside timers.

The official SGLang loop uses FlashInfer and synchronizes each step. It does not
exercise serving overlap, HTTP or streamed text. This experiment cannot establish
serving parity, cross-engine output equivalence or pure-kernel speed. Local
[results](../profiling/qwen2/sglang-comparison/RESULTS.md) retain raw samples, backend
and graph capture logs, exact commands, version/hardware metadata, process aggregates
and frozen source/binary. CUDA/torch allocator peaks have different accounting and
are not whole-device memory measurements. No profiler is attached to native runs.


In the initial three-process BF16 matrix, Rust eager short-context decode has
26-31% lower latency than SGLang eager; 4K prefill is about 12x slower. SGLang
Graph further reduces decode latency. A separate Nsight capture verifies 64 Rust
forwards and only one final 4-byte D2H after the last kernel. Profile long prefill
to separate attention/linear cost before tuning; these static results do not
establish serving throughput or cross-engine generated-ID equivalence.


### Prefill attention routing and operator profiling

BF16 suffixes with at least 64 queries per context use multi-query MMA attention;
small/mixed suffixes, FP32 and decode retain scalar attention. The tiled path shares
K/V across queries, preserves token-slot indirection and cached/ragged causal offsets,
and keeps online softmax/accumulation in FP32. No extra global score buffer is used.
The initial 4K comparison improves prefill about 5.9x at B1/2/4 while leaving linear
tiles unchanged; see the local [eager tuning report](../profiling/qwen2/bf16-eager-tuning/RESULTS.md).

NVTX ranges name embedding, metadata, projection roles (`qwen2.linear.q/k/v/o/gate/up/down/lm_head`),
RMSNorm, RoPE, KV store, scalar/tiled attention, residual add, SwiGLU and argmax.
They are thread-local synchronous submission scopes, never GPU timers or ranges
across await. Use launch correlation / projected GPU intervals for attribution.
The SGLang adapter offers --comparison-nvtx module hooks for profiling only; leave
it off for native timing. It requires --disable-cuda-graph. An eager prefill capture
uses --profile --profile-activities CUDA_PROFILER --profile-stage prefill under
nsys --capture-range=cudaProfilerApi --capture-range-end=stop. Prewarm the same shape.


Large BF16 linear matrices use a finite `LinearTile` bucket (at least 128 packed
rows: 32x128, K-reduction 64). Smaller shapes and FP32 retain prior tiles, including
decode/LM-head matrices. Linear rows are independent and may cross request boundaries;
attention routing still checks each context separately. Masked K tails use ceiling
iteration. This isolated blocking change reduces 4K prefill by about another 32-33%
after attention MMA; the [tuning report](../profiling/qwen2/bf16-eager-tuning/RESULTS.md)
retains the full short/long B1/2/4 matrix, candidate screens and precision checks.

For BF16 M >= 2048 and N >= 512, a third bucket uses 128x128/K64; skinny K/V
projections and smaller matrices keep the earlier tiles. Three-process B1/2/4
4K prefills are 114/227/452 ms, another 11–13% throughput improvement. Independent
CPU output-coverage tests and a >2048-token Transformers BF16 SDPA reference pass.
Benchmark `--input-pattern ramp` complements the default repeated-token baseline;
comparison runs assert matching input-pattern checksums for both engines.
