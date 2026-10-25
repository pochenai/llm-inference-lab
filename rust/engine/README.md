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
include Scheduler's 100ms collection window; the default fills its dispatch limit.
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
The initial backend converts checkpoint weights to FP32 once and implements all
numerical operators in cuTile: embedding, checkpoint-layout linear projections,
RMSNorm, Q/K bias and split-half RoPE, causal GQA over mapped KV slots, residual,
SwiGLU, the tied/untied LM head, and two-stage GPU greedy argmax. It does not invoke
PyTorch, cuBLAS, or a Python model for inference. The optional Python script is an
independent CPU/FP32 reference only.

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
Admission budgets include all possible generated KV positions, excluding the final
sampled token whose KV will be computed in a subsequent turn. Only pure greedy
settings are supported. Mid-turn cancellation releases contexts; partial failed
turns are invalidated if KV has advanced. Completed history survives text-decode errors.
The Qwen example submits through Scheduler -> Runner; optional reference mode still
reads diagnostic logits directly. Exact generated IDs and multi-turn Runner continuation
are checked by the real-checkpoint regression. The CLI response contains text and request_id.

Qwen currently serializes batch items through its existing one-context attention kernel;
SimpleMLP retains packed GPU batches and CUDA Graph replay. Shared KV interfaces are unchanged.
Token/logit readbacks enqueue asynchronous D2H into reusable pinned buffers and become
readable only after outer completion. Dropping an in-flight model step may require
context release; retained GPU leases prevent premature storage reuse.
Qwen uses eager execution with an exclusive, exact-suffix-shape workspace per
forward. Two backing allocations hold disjoint float/integer regions; all decoder
layers reuse them in stream order. Every output is fully overwritten, so scratch
needs no zero-fill kernel. The lease survives submission failure, cancellation,
and escaped tensor handles; concurrent forwards receive different workspaces.
The cache retains at most two idle workspaces (normally prefill/decode) and drops
idle shapes under memory pressure. On a miss, scratch admission checks its exact
size against device free memory plus unused execution-pool reservations after
weights/KV allocation. This snapshot cannot reserve memory against other callers;
allocation errors still propagate as per-request failures.
For Qwen2.5-0.5B, scratch bytes are
`4 * (suffix_tokens * 20352 + 154021)`: about 3.07 MiB for 32 tokens and
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
as elapsed wall time for an individual batch. Prefill means empty KV or a multi-token
suffix; decode means one token with existing KV. Durations cover CPU preparation,
submission, GPU/D2H completion and host result handling, excluding Scheduler work.
`step_wall_seconds` spans the entire model step through GPU/D2H completion,
KV commit and CPU output access. `host_submit_seconds` is returned in the private
`StepOutput` for that invocation: host wall time inside submit, including scratch,
input uploads, JIT and CUDA calls, without awaiting GPU completion. It excludes
KV mapping preparation and later host output access. GPU execution can overlap
host submission, so subtracting them does not yield pure GPU kernel time. Failed
steps increment the failure counter and have no successful submission-duration
sample; tracing leaves that duration unavailable. Cold JIT,
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

The example supplies the explicit two-message Qwen2 chat template and uses pure
greedy decoding (no repetition penalty or stochastic sampling).

```bash
# Point to an existing complete HF snapshot; no model download is performed.
QWEN2_MODEL_DIR=/home/po/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
cargo run --manifest-path rust/Cargo.toml --offline --release -p engine --example qwen2 -- --model-dir "$QWEN2_MODEL_DIR" --max-new-tokens 32

# Independent reference: use an interpreter with installed CPU-capable torch/Transformers.
env -u LD_LIBRARY_PATH PYENV_VERSION=3.14.3 python rust/engine/examples/qwen2_reference.py "$QWEN2_MODEL_DIR" --max-new-tokens 32 --output-dir rust/engine/artifacts/qwen2
cargo run --manifest-path rust/Cargo.toml --offline --release -p engine --example qwen2 -- --model-dir "$QWEN2_MODEL_DIR" --max-new-tokens 32 --reference-dir rust/engine/artifacts/qwen2

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
