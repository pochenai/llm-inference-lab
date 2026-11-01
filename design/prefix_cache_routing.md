# Prefix caching and cache-aware routing




## 1. Our approach: ownership first, routing second

### 目标

对，**session/fork 是调用方显式表达共享关系，prefix caching 是引擎自动发现共享关系**。

这样下一步的目标就具体了：两个独立 session，带相同的 system prompt、不同的问题；第二个请求自动复用 system prompt 的 KV，只计算后面的内容。

先在单 Runner 跑通这个场景，再让 cache-aware routing 把类似请求优先发到已有缓存的 Runner。

因此最小改动范围很明确：Scheduler 增加多 Engine 分发、session owner、前缀提示索引和负载计数。 Engine 内的精确缓存继续独立实现；在它完成前，可以验证路由选择，但还不能获得跨 session 的 KV 复用收益。


Status: local multi-destination routing and opt-in Engine-local exact token-prefix
KV reuse are implemented. Cache inventory synchronization and automatic eviction
remain deferred. [The engine guide](../rust/engine/README.md#multi-destination-routing)
shows the current public API and validation command. This section records the
contracts; [spec.md](spec.md) remains the execution/ownership reference.
Rust snippets summarize state and actions rather than literal implementation.
New types/methods stay private or `pub(crate)` unless required for embedding.

### Deployment and ownership boundary

```text
Caller
  -> Scheduler: admit, select destination, forward requests and replies
       |-- RoutingPolicy: read-only destination selection
       |-- RoutingState: placement and bounded prefix hints
       |-- RoutingRuntime.requests_by_destination: reverse request lookup
       |-- RequestForwarder: request/reply transport
       |-- Named destinations and session routes
       |
       +-> Engine A: ModelRunner + exact PrefixCache + KVPool + GPU group
       +-> Engine B: ModelRunner + exact PrefixCache + KVPool + GPU group
```

Start with one Scheduler process and multiple Engine endpoints. No separate gateway,
HA service, directory sharding or KV transfer is required. An Engine endpoint can
be one independently executable model replica, including a TP GPU group. A routing destination itself is just a named request ingress;
it can resolve to an Engine or another Scheduler without a target-kind enum.
Each ModelRunner retains full-forward batching, chunking and overlap control.
The Scheduler never directs per-layer KV access or individual GPU forward steps.

The routing directory predicts locality. Only an Engine's exact token-prefix
cache can authorize KV reuse. Recording a dispatch neither publishes KV nor
proves GPU completion. Routing can be tested before local prefix reuse exists,
but cross-session prefill savings require that Engine feature.

### State: configuration, placement and bounded prefix hints

```rust
const DEFAULT_MAX_PREFIX_BYTES: usize = 4 * 1024;

/// Opaque registered name; scoped to one endpoint registration lifetime.
struct DestinationId(u64);

/// Interned compatibility identity resolved from model/input configuration.
struct CacheNamespace(u64);

struct AccessState {
    /// Logical recency counter; used by pruning, not a wall-clock timestamp.
    last_access_tick: u64,
}

struct RoutingConfig {
    /// Cap both lookup and retained routing-prefix bytes; default is 4096.
    max_prefix_bytes: NonZeroUsize,
    /// Total accounted index allocation budget; default 64 MiB, configurable.
    max_index_bytes: NonZeroUsize,
}

struct RoutingState {
    /// Placement metadata; endpoints and lifecycle counters have separate owners.
    destinations: HashMap<DestinationId, DestinationState>,
    session_routes: HashMap<SessionId, DestinationId>,

    /// Bounded, advisory prefix index; owns no GPU storage.
    config: RoutingConfig,
    /// One namespace is configured per Scheduler.
    prefixes: PrefixTree,
    /// Increasing logical recency counter assigned to recorded dispatches.
    access_tick: u64,
}

struct DestinationState {
    /// Immutable model/input compatibility identity, shared with directory keys.
    namespace: CacheNamespace,
    available: bool,
}

/// Routing fields only; existing request/reply and cancellation fields are omitted.
struct RoutingRuntime<F, P> {
    state: RoutingState,
    forwarding: Forwarding<F>,
    policy: P,
    /// Reverse index of unresolved forwards for targeted cleanup, not load prediction.
    /// This actor owns updates, including future async completion reports.
    /// Client disconnect alone does not prove Engine execution finished.
    requests_by_destination: HashMap<DestinationId, HashSet<RequestId>>,
}

struct PrefixTree {
    /// Shared generic radix map; routing owns access/budget policy, never KV.
    destinations: HashMap<DestinationId, DestinationPrefixes>,
    charged_bytes: usize,
}

struct DestinationPrefixes {
    entries: RadixMap<u8, AccessState>,
    charged_bytes: usize,
}

```

PrefixTree wraps the reusable `radix_tree::RadixMap<u8, AccessState>`, sharing its
safe arena/compressed-path core with the engine's token radix cache. The module
adapts patricia_tree's compression/splitting algorithm; its MIT notice is retained
in `rust/engine/src/radix_tree/LICENSE`. The crate dependency and recursive bucket
adapter are removed. Map lookup is read-only and includes implicit path overlap:

```text
A's map: "SYS:aaa"
B's map: "SYS:bbb"
query "SYS:ccc" -> {A: 4, B: 4}
query "SYS:aax" -> {A: 6, B: 4}
```

Lookup visits each populated destination's byte path, not all stored prompts or
three neighbor buckets. Ordinary map removal prunes empty leaves, merges valueless
unary nodes and reuses arena slots. The engine's resource tree never merges nodes
behind active path pins. There is no engine KV or full-input backing allocation
in gateway maps. Paths are compressed within, not across, destinations.

Identity and associated state use named types; counts, byte lengths, capacity
limits and simple flags retain primitive types. Raw byte slices describe routing keys rather than another identity. Reuse existing RequestId and SessionId contracts.
The newtype sketches omit routine derives needed for map keys and copying IDs.

AccessState initially contains only recency. Add a monotonic expiry field there
when TTL is implemented, rather than changing every association's value type.
Pruning owns the eviction rule: remove expired associations first, then rank
remaining candidates by recency when over budget. Do not give AccessState a fixed
Ord implementation that conflates expiry eligibility with eviction priority.

A match is advisory and may be shorter than the query. Each destination appears
once, with its own longest matched byte length. Removing one recorded key leaves
matches supported by its remaining keys; removing a destination drops its own map.
Eviction removes complete recorded hints rather than custom internal node values.
An empty query has no locality preference.

Use bytes internally, so the 4096-byte boundary may fall inside a UTF-8 code point;
these labels are never decoded as strings. Query and insertion use the same cap.
The original request string is forwarded intact to the Engine. The cap limits only
routing discrimination, not context length or exact local KV reuse. Copy bounded
segments into the index; retaining a tiny slice of a huge Arc-backed prompt would
otherwise defeat the memory limit.

CacheNamespace partitions incompatible model revisions, adapters and input
construction/position semantics; it is resolved from endpoint/application
configuration. Text matching is still advisory because tokenization boundaries
can differ. A one-model deployment can have one namespace; no per-request model
registry is needed for the first version.

**Memory invariant:** Requests/PendingRequest retains reply/lifecycle metadata only.
Never add full prompts, token histories, or shared buffers/slices/closures that keep
full-input-sized allocations alive. The original input travels in the forwarding
command; only capped prefix segments may persist in the routing index. The index
budget and per-prefix cap do not bound transport queues, response buffering or
Engine history, which have separate owners and admission/backpressure concerns.

Reuse existing Scheduler PendingRequest/reply bookkeeping: add the selected
DestinationId there rather than duplicating the whole pending-request map. Clones share
one Scheduler's ID generators. Independently deployed Schedulers will require
unique IDs and explicit session-owner/response routing; that is not implemented
by merely cloning this state definition.

### Actions: directory mutation and policy selection

```rust
impl RoutingState {
    /// Read the fixed available next hop; missing placement is SessionLost.
    fn session_destination(&self, session: SessionId) -> Result<DestinationId, EngineError>;

    /// Query capped prefix hints only; no counters or policy dependency.
    fn prefix_matches(&self, input: &str) -> HashMap<DestinationId, usize>;

    /// Available compatible endpoint identities, including those without prefix hints.
    fn available_destinations(&self) -> impl Iterator<Item = DestinationId> + '_;

    /// Borrow input and stage capped locality hints before forwarding consumes it.
    /// Send failure invalidates this destination before another request is routed.
    fn record_dispatch(
        &mut self, namespace: &CacheNamespace, input: &str, destination: DestinationId,
    );

    /// Under pressure, remove a batch of old capped hints to create headroom.
    fn prune(&mut self);

    /// Invalidate this endpoint's hints without touching any GPU allocation.
    fn remove_destination(&mut self, destination: DestinationId);
}

trait RoutingPolicy {
    /// Select among available compatible candidates using borrowed load/hint data.
    /// No channel sends, session mutation, index writes or GPU waits occur here.
    fn select(&self, candidates: &[DestinationCandidate]) -> Option<DestinationCandidate>;
}

struct DestinationCandidate {
    destination: DestinationId,
    matched_prefix_bytes: usize,
}

/// Default policy: prefer the longest matching prefix, choose uniformly among ties.
struct RandomPrefixPolicy;

trait RequestForwarder {
    /// Accept a command for forwarding; Ok means local acceptance, not execution.
    /// Later replies/errors use the gateway's existing reply/acknowledgement path.
    fn forward(
        &self, destination: DestinationId, command: RunnerCommand,
    ) -> Result<(), EngineError>;
}

impl<F, P> RoutingRuntime<F, P> {
    /// None opens an empty session; Some borrows an execution request.
    /// Existing sessions keep their next hop; other requests use prefix/random selection.
    fn route(&self, request: Option<&SessionRequest>) -> Result<DestinationCandidate, EngineError>;

    /// Remove the indexed ID after terminal cleanup removed the request exactly once.
    fn remove_pending(&mut self, destination: DestinationId, request_id: RequestId);
}
```

RoutingPolicy returns the selected DestinationCandidate directly; no separate
decision struct is needed. These are snapshots, not mutable destination state.
RoutingRuntime::route builds DestinationCandidate records from compatible available
endpoints and directory results, including zero-match destinations. Candidates
contain only destination identity and matched prefix bytes. The default
RandomPrefixPolicy uses the following rule:

```text
existing session -> fixed next hop; bypass prefix/random selection
new request:
  find the greatest matched_prefix_bytes among eligible destinations
  -> keep every destination with that match length
  -> choose one uniformly at random
no prefix hit -> all eligible destinations have match length 0; choose randomly
no eligible destination -> return no route
```

For a system-prompt-only match shared by A/B/C, each is selected with probability
1/3. With different match lengths, random selection applies only to the longest
matches. Do not weight by association count or accidentally select only the first
HashMap entry. Load-based selection and its candidate fields are deferred; no
load-skew configuration is needed yet. The framework validates the returned endpoint before
forwarding; selection alone does not mutate placement or counters.

Expose a policy-construction hook when wiring Scheduler configuration; a generic
policy parameter or an existing-style Arc policy is sufficient. The policy is the
IoC extension point. Do not expose routing-state mutation to policy implementations
and do not put transport details into RoutingState or RoutingPolicy.
RequestForwarder is the separate transport extension point. Start with the existing
local channel implementation, which resolves DestinationId through its own registry.
A later HTTP implementation can accept work into a transport task and translate
upstream streams/errors into the same reply lifecycle. It must not block the routing
loop for generation. Reuse RunnerCommand/RunnerReply and session acknowledgements;
no new response envelope, per-token callback framework or RPC stack is required now.
A generic forwarder parameter is sufficient; dynamic dispatch is not required.

### Control: Scheduler owns transitions, policy only chooses

#### Explicit assumption: the gateway observes the request lifecycle

The initial gateway is a streaming proxy. It forwards the request and retains
the client response stream until a terminal outcome; responses pass back through
it rather than returning directly from the Engine to the client. This is the
assumption behind local dispatched/finished accounting, not a property of routing
itself. RoutingState and RoutingPolicy must remain independent of this assumption.

```text
Current: client -> gateway -> destination
         client <- gateway <- response stream
         gateway observes terminal outcome -> RoutingRuntime.remove_pending

Future direct return: client <- destination
         gateway needs a separate completion notification or load report
         route selection / request forwarding alone cannot imply completion
```

RequestForwarder defines how replies and terminal outcomes reach the gateway.
A future redirect, direct-return or connection-handoff implementation must state
whether per-request completion remains observable. If it does not, gateway-local
pending accounting cannot be treated as exact execution load; use separate
reported observations instead. Client disconnect and successful enqueue are
neither proof that GPU work finished nor permission to reclaim Engine KV.

Extend the existing dispatch task into a routing event loop. It exclusively owns
RoutingState and its own requests_by_destination map, and handles request commands plus small
completion/endpoint events. The configured RequestForwarder owns endpoint transport.
Existing reply processing continues to stream deltas independently. It notifies
the routing loop of terminal outcomes; routing does not run for every text delta.
Do not hold a routing lock or await a model forward in this loop.
Asynchronous completion delivery does not require moving counters out of the
runtime: producers send events to this actor, which serializes and deduplicates
updates to its existing requests_by_destination map. No shared mutable counter is needed.

```text
New request, no session owner:
  admit/assign RequestId using existing Scheduler path
  -> bound lookup prefix; collect compatible available destinations and their loads
  -> RoutingPolicy.select(candidates)
  -> write selected DestinationId into existing PendingRequest
  -> borrow the original input to stage a capped prefix hint; no temporary Vec
  -> RequestForwarder.forward(destination, original command)
  -> on Ok: insert RequestId into the destination set
  -> on immediate Err: existing terminal cleanup and invalidate destination hints
  -> any later forwarding failure enters the normal terminal reply path

RunnerReply::Delta:
  forward through existing Scheduler stream/observer path
  -> no destination selection and no pending-count change

RunnerReply::Finished / equivalent terminal transport failure:
  remove PendingRequest exactly once through existing terminal cleanup
  -> notify routing loop with selected DestinationId
  -> RoutingRuntime.remove_pending exactly once
  -> retain approximate prefix hints until pruning or endpoint invalidation
```

A dispatch registers accounting before the loop processes completion events, so a
fast Runner cannot decrement an unregistered count. Prefix hints are staged synchronously
before forwarding consumes the input, using a borrow rather than an intermediate
prefix allocation. The same actor completes acceptance or invalidation before
routing another request. A failed send can still cause best-effort index pruning;
no transactional restoration of unrelated evicted hints is promised. No available destination is a
request error, not an invitation to route to an incompatible destination. Index budget
exhaustion skips hint recording after pruning; it must not fail a sent request.
Local send-failure cleanup emits no pending-count decrement event because no
request was counted as dispatched. Only guaranteed enqueue failure is safe to
retry elsewhere. Once enqueued or
partially streamed, execution failure does not trigger transparent replay.

The initial deployment is a streaming proxy: client -> gateway -> destination,
with response deltas returning through the gateway to the original client. Keep
that response connection/stream in existing pending/reply handling, never in the
prefix tree. Forwarding chunks does not run token decoding or model computation
in the gateway. RequestForwarder changes the transport, not the RoutingPolicy.

requests_by_destination indexes this gateway's unresolved forwarded requests, not the
Engine's exact GPU workload. It provides reverse lookup, not load estimates. Successful
forwarding inserts the ID; terminal cleanup removes it once. On disconnect, take
that destination's set and finish only those IDs: O(K) request cleanup for its K
outstanding requests, rather than scanning all N gateway requests. The index holds
IDs only, never copies of request contents. Prefix-hint invalidation has its own
separate traversal cost. Dropping the client stream alone does not prove the
Engine stopped. The current local Runner path retains accounting until Finished;
a transport must retain terminal tracking or obtain cancellation confirmation if
execution continues after delivery stops. A terminal transport failure releases
this gateway's tracking but is not proof of GPU completion or free KV capacity.
Remote worker load reports, if added, are separate advisory observations. Future
redirect/direct-return modes need their own completion/reporting contract; they
must not fabricate Finished events from a routing decision.

Session actions remain on the existing Scheduler API:

| Action | Routing control and transition |
| --- | --- |
| open_session | Choose an available compatible destination uniformly at random; create through it; install the next-hop session route after successful acknowledgement, before returning SessionId. Route acknowledgements back as events so other requests can proceed. |
| execute with session_id | Resolve owner and validate namespace/availability; forward prompt delta there. Unknown/unavailable owner fails explicitly. Record no delta-only prefix as a complete prompt hint. |
| close_session | Send to owner; remove placement after successful close or SessionLost acknowledgement. A busy-session rejection leaves ownership intact. |
| Destination disconnect | Mark unavailable, invalidate hints and resolve affected pending requests once; sessions routed through it become unavailable. Keep other destinations serving. |
| Destination replacement | Use a fresh DestinationId for this registration lifetime; old hints and owner mappings must not identify the replacement as the old process. |

Missing session placement or missing/invalidated Engine history returns the typed
EngineError::SessionLost(SessionId). Execution and close SessionLost replies remove
the lost placement only if its current owner matches the reply's destination. It never implicitly creates an empty session.
A harness that retains full accepted history may explicitly open a new session
and replay that history plus the pending input before generation starts. This
recovery is not implemented in the gateway; a disconnected destination still
returns a transport error, which does not prove the stored history was lost.

First-version clients still serialize a session's turns. Authoritative owner
placement and advisory prefix hints have different lifecycles; index pruning
never removes or reassigns a live session owner.

### Budget, observability and later routing tiers

The default cap is 4 KiB and the default accounted index budget is 64 MiB; both
are configurable at construction; prefix caps above 64 KiB are rejected to bound
routing key work independently of full prompt size. With only one available compatible destination,
skip both prefix lookup and recording: locality cannot change the selected next
hop. Session routing and request lifecycle tracking remain active. Clear obsolete
prefix hints if endpoint loss leaves only one compatible destination. One million distinct
nonsharing capped prefixes contain at most about 3.81 GiB of label payload;
this is not total index memory or a user-count capacity guarantee. Shared prefixes
reduce payload; branches, destination associations and container capacities add costs.

Charge each distinct capped key as `length + 128` bytes, plus conservative map,
arena, label-vector, free-list and child-map capacity charges. Shared paths are
charged repeatedly, so hints may be evicted before true allocation requires it.
This is not process RSS or a bound on fragmentation/transient eviction scratch.
Arena slots recycle; batch pressure can discard a destination's remaining map to
release retained arena capacity. No full-arena compaction occurs per removed key.

After an insertion exceeds budget, reclaim to seven eighths of the budget. Each
pass scans entries once and keeps only the oldest eighth of keys in a temporary
heap, then removes those keys. There is no per-entry node-arena shrink_to_fit or
full-tree scan for every newly admitted hint. Batch scans still reconstruct capped
keys via the map iterator and can cause periodic actor latency spikes; this
is not a claim of constant-time eviction or million-prefix serving throughput. Avoid allocating the full oversized input
before applying the cap. Session routes and pending requests are separate state
and require their own lifecycle/admission bounds.

Record conservative index charges, retained hint count, cached destinations and
dropped hints; routing latency, selected match bytes, no-hit fallback and endpoint errors.
Engine-confirmed reused KV tokens are a separate metric from directory matches.

Keep destination identity, endpoint resolution and selection separate:

```text
prefix lookup + RoutingPolicy -> DestinationId (opaque registered name)
                           -> RequestForwarder resolves DestinationId to its endpoint
                           -> local sender now, remote transport when needed
                           -> the next hop accepts the request
```

No Engine/Router enum or type check participates in routing. The receiver decides
whether to execute locally or select another destination. The forwarder
owns the endpoint registry, initially local RunnerCommand senders. RoutingState
contains only destination metadata and never stores channels or network addresses. A downstream Scheduler can later expose the
same command-ingress contract; remote networking adapts that contract only when
needed. This is logical indirection, not an unsafe function-pointer call.

Each tier keeps RoutingRuntime.requests_by_destination for its own next hops, not individual GPU load
behind another Scheduler. Future downstream capacity/load reports may improve
that estimate without exposing GPU addresses. A successful channel send confirms
only enqueueing; downstream acknowledgement/completion contracts still apply.
Keep the configured routing topology acyclic; do not register a route back to an
ancestor. Correlation IDs and session routing must remain unambiguous across hops.

session_routes maps a session to its fixed next hop. In a hierarchy, each tier
retains its own part of the path; the final Engine owns the actual session/KV.
A short prefix can map to several next hops to avoid a system-prompt hotspot.
Endpoint replacement uses a fresh registration identity; stable display names
alone must not silently reconnect old sessions to empty replacement processes.
No recursive-routing transport, HA, mesh or KV migration is implemented now.

### R1/R2/R3 validation

- Lost session placement is retired on execution and close SessionLost replies;
  busy errors preserve placement. A 30,000-cycle public-API replay no longer grows
  the old session route table or forwards close to those lost placements.
- Singleton hint maintenance is allocation-free. Multi-destination prefix reuse
  decisions retain the independent byte-prefix oracle and UTF-8 cap tests.
- Historical PatriciaMap CPU probes are archived in the local
  [report](../rust/profiling/routing/patricia-20261101/RESULTS.md). Their timings do
  not describe the replacement safe arena map; gateway performance has not been
  re-benchmarked. Current routing tests cover split/merge, prefix oracles, budget
  headroom, singleton bypass and session/endpoint cleanup.

### First implementation checks

- Default 4096-byte cap, configurable smaller/larger caps, multibyte input, and
  full prompt forwarding; lookup and recorded labels obey the same byte limit.
- Shared/partial prefixes, mid-label splits, multiple destinations per prefix, and
  namespace isolation; selection stays within the longest-match candidate set.
- Default random selection handles a single candidate, multiple equal matches,
  no-hit fallback and no eligible destinations. Inject a seeded RNG through
  select_with for reproducible tie-selection tests. Load fields are not part of
  the current candidate API.
- A singleton compatible destination keeps the prefix index empty. Multi-destination
  pressure reclaims batch headroom without changing GPU state or failing execution.
- SessionLost from execution/close retires placement; repeated recovery leaves no
  old routes, while busy errors and another destination's placement remain intact.
- Local/remote forwarders preserve the reply contract. Send failure, fast completion,
  duplicate terminal paths and disconnect do not leak or double-decrement counters;
  client stream drop alone is not an execution-completion event.
- Routing policy/index tests need no live forwarding transport, and transport
  lifecycle tests need no real prefix tree. One destination failure preserves others.
- Session open/execute/close acknowledgements preserve owner placement; prompt
  deltas bypass new-request prefix selection and restarted endpoints are distinct.
- Two Runner endpoints can validate routing on one GPU; real prefix-reuse savings
  are measured only after Engine-local shared KV ownership is implemented.

## 2. How SGLang implements these boundaries

Reviewed local checkout: `0263fdacb223e92035ebc947f098ba724ee5634f`.
This is the current source tree, not necessarily the installed SGLang 0.5.9 used
for W8 benchmarks. Links below pin the reviewed revision. No runtime experiment
or full proof of every overlap/hierarchical-cache configuration was performed.

### Local scheduling and replica topology

SGLang's local Scheduler initializes a TP worker, request-to-token pool, KV
allocator and tree cache; its normal/overlap loops choose complete local forwards.
This maps mostly to our ModelRunner, rather than to our ingress Scheduler.
The data-parallel controller dispatches to model replicas, each potentially a TP
group. The gateway does not schedule each GPU forward.

Sources: [local scheduler ownership](https://github.com/sgl-project/sglang/blob/0263fdacb223e92035ebc947f098ba724ee5634f/python/sglang/srt/managers/scheduler.py#L559-L602),
[normal/overlap loops](https://github.com/sgl-project/sglang/blob/0263fdacb223e92035ebc947f098ba724ee5634f/python/sglang/srt/managers/scheduler.py#L1895-L1975),
[DP/TP launch](https://github.com/sgl-project/sglang/blob/0263fdacb223e92035ebc947f098ba724ee5634f/python/sglang/srt/managers/data_parallel_controller.py#L373-L410).

### Exact local cache: match, protect, checkpoint, release

The default full-attention path now selects **UnifiedRadixCache**, with a FULL
component and optional SWA/MAMBA components. The older standalone RadixCache is
useful to understand the simpler tree and eviction algorithm, but is not the
default implementation in this checkout.

```text
request input preparation
  -> exact token-prefix match, bounded by input_len - 1
  -> acquire cache protection during local scheduling/admission
  -> map cached indices into the request row; compute uncached suffix
  -> checkpoint(req, up_to=...) updates retained prefix and mappings
  -> release_kv_cache drops request-owned tail, unpins and releases row
  -> later eviction reclaims eligible cached storage
```

Current matching accepts MatchPrefixParams and returns device KV indices plus
node/component information. Unified matching delegates to tree_core, applies
actions and finalizes component-specific results. Its request-coupled interface
serves more model types than our first FULL-only implementation needs.

Pinning is eviction protection, not a GPU mutex held while one token waits for
another. The standalone tree's lock_ref walks ancestor nodes and moves capacity
between protected and evictable states. Request release and cache eviction are
separate operations. Current publication uses **checkpoint + release_kv_cache**;
do not describe this checkout with the older cache_finished_req/cache_unfinished_req names.

| Mechanism | Pinned source |
| --- | --- |
| Default backend selection | [registry.py](https://github.com/sgl-project/sglang/blob/0263fdacb223e92035ebc947f098ba724ee5634f/python/sglang/srt/mem_cache/registry.py#L80-L177) |
| Match input/result contracts | [base_prefix_cache.py](https://github.com/sgl-project/sglang/blob/0263fdacb223e92035ebc947f098ba724ee5634f/python/sglang/srt/mem_cache/base_prefix_cache.py#L64-L72), [MatchResult](https://github.com/sgl-project/sglang/blob/0263fdacb223e92035ebc947f098ba724ee5634f/python/sglang/srt/mem_cache/base_prefix_cache.py#L260-L289) |
| Unified exact matching | [unified_radix_cache.py](https://github.com/sgl-project/sglang/blob/0263fdacb223e92035ebc947f098ba724ee5634f/python/sglang/srt/mem_cache/unified_radix_cache.py#L553-L573) |
| Leave one input token for logits | [schedule_batch.py](https://github.com/sgl-project/sglang/blob/0263fdacb223e92035ebc947f098ba724ee5634f/python/sglang/srt/managers/schedule_batch.py#L1715-L1725) |
| Admission pin and release receipts | [schedule_policy.py](https://github.com/sgl-project/sglang/blob/0263fdacb223e92035ebc947f098ba724ee5634f/python/sglang/srt/managers/schedule_policy.py#L1608-L1616), [temporary protection](https://github.com/sgl-project/sglang/blob/0263fdacb223e92035ebc947f098ba724ee5634f/python/sglang/srt/managers/schedule_policy.py#L1200-L1216) |
| Current publication API | [checkpoint](https://github.com/sgl-project/sglang/blob/0263fdacb223e92035ebc947f098ba724ee5634f/python/sglang/srt/mem_cache/unified_radix_cache.py#L1039-L1077) |
| Request cleanup | [common.py](https://github.com/sgl-project/sglang/blob/0263fdacb223e92035ebc947f098ba724ee5634f/python/sglang/srt/mem_cache/common.py#L297-L340) |
| Simpler eviction/protection mechanics | [radix_cache.py](https://github.com/sgl-project/sglang/blob/0263fdacb223e92035ebc947f098ba724ee5634f/python/sglang/srt/mem_cache/radix_cache.py#L551-L615) |

RadixKey includes token IDs, namespace/salt and a logical limit to avoid an initial
full-array slice. The standalone comparator uses array comparisons rather than
per-token Python loops. This does not eliminate the need to inspect long prefixes.
See [key representation](https://github.com/sgl-project/sglang/blob/0263fdacb223e92035ebc947f098ba724ee5634f/python/sglang/srt/mem_cache/radix_cache.py#L59-L82)
and [comparison](https://github.com/sgl-project/sglang/blob/0263fdacb223e92035ebc947f098ba724ee5634f/python/sglang/srt/mem_cache/radix_cache.py#L178-L218).
Our completion/accepted-history publication rule above is a proposed engine
contract, not a claim that every SGLang cache backend uses identical event handling.

### Approximate gateway cache-aware routing

The gateway maintains a separate **text-prefix routing index**, inferred from
request history. It uses characters rather than exact model token IDs. Trees are
isolated by pool/model; tenants identify workers by URL. Selection records text
and the chosen worker before successful model execution has confirmed any GPU KV.
Thus it predicts locality; it is not a GPU cache directory with authoritative holds.

The implemented decision is:

```text
restrict to healthy workers
  -> severe absolute AND relative load imbalance? choose minimum-load worker
  -> otherwise, prefix ratio above threshold? try matching healthy worker
  -> otherwise choose minimum-load worker, with randomized ties
  -> update approximate routing index
```

A stale/unhealthy tenant has a fallback path. Evicting a routing-tree entry only
bounds router memory; it does not free GPU KV. The file header's low-match
smallest-tree description is stale: the actual low-match branch selects minimum
load. Follow implementation rather than that comment.

Sources: [index purpose](https://github.com/sgl-project/sglang/blob/0263fdacb223e92035ebc947f098ba724ee5634f/sgl-model-gateway/src/policies/cache_aware.rs#L19-L30),
[imbalance routing](https://github.com/sgl-project/sglang/blob/0263fdacb223e92035ebc947f098ba724ee5634f/sgl-model-gateway/src/policies/cache_aware.rs#L329-L365),
[selection and index update](https://github.com/sgl-project/sglang/blob/0263fdacb223e92035ebc947f098ba724ee5634f/sgl-model-gateway/src/policies/cache_aware.rs#L392-L512).

### Multiple gateway instances: verified limits

Each gateway can operate with independent approximate hints. Optional mesh hooks
send tree operations, and application/restoration hooks exist. However, this
checkout explicitly states that the remote receive path for cache-aware tree
operations is not wired and its registry forwarder has no in-process caller.
Outbound integration is not evidence of a working runtime convergence loop.
Do not infer CRDT consistency, authoritative session ownership or shared exact
GPU cache state from the presence of mesh types.

Sources: [explicit receive-path limitation](https://github.com/sgl-project/sglang/blob/0263fdacb223e92035ebc947f098ba724ee5634f/sgl-model-gateway/src/policies/cache_aware.rs#L263-L274),
[registry forwarder](https://github.com/sgl-project/sglang/blob/0263fdacb223e92035ebc947f098ba724ee5634f/sgl-model-gateway/src/policies/registry.rs#L417-L457).

## 3. Proposed implementation order and validation

| Stage | Deliverable | Required evidence |
| --- | --- | --- |
| 1. Multiple destinations (routing implemented) | Scheduler selects between Runner endpoints using bounded hints and RandomPrefixPolicy | Longest-prefix ties are selected uniformly; no-hit fallback, stale hints, unavailable endpoints and session routing work. Load-aware policy is a later extension. |
| 2. Local prefix ownership (next) | Independent rows, shared immutable committed KV, exact radix lookup, publication and eviction on one Runner | Repeated/diverging prefixes match uncached generation; full hit, partial hit, namespace isolation and release/eviction restore capacity. |
| 3. Existing execution integration | Chunked prefill, one-round overlap and optional CUDA Graph work with sharing | No pending/unaccepted publication; Gate-controlled tests prove no early slot reuse; changed mappings are honored on replay. |
| 4. Multiple Schedulers | Unique IDs, owner/incarnation resolution, reply routing and bounded admission | Concurrent ingress has no ID collision or session owner split; restart invalidates old hints/owners; no global per-forward lock. |

Use two independent same-device pools/Runners to validate routing logic before
multi-GPU hardware is available; this does not measure cross-GPU performance or
validate TP collectives. Keep local correctness independent of directory quality.
For performance, compare identical prefixes/arrivals with caching disabled/enabled,
report true reused tokens, TTFT, queue time and physical KV, and include cold misses
and load imbalance rather than only warm-hit throughput.

Defer remote KV transfer, session migration, distributed exact prefix locks,
hierarchical storage and multi-component cache frameworks until a measured need.
The first implementation should reuse existing KVContext progress, local Runner
control flow, metrics and GPU lease retention; it should not create a second
execution scheduler or require a fleet-wide service to reuse one local prefix.
