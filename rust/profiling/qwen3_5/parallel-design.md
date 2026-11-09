# Deferred engineering notes: graph-only shared/routed overlap

Do not implement before finishing the larger measured improvements/re-profiling.
No code exists yet. This is a possible safe route if parallel work remains useful.

cuda-async 0.5 has no public capture-context constructor for another stream;
Scope records only on its own stream. Do not transmute private fields or skip
module/storage lifetime ownership. An ordinary alt ExecutionContext held forever
would retain write accesses and conflict with later readers. async_on releases
owners immediately, including modules; relying on the global JIT cache is unsound
because a quiesced cache eviction may legally precede a later graph replay.

Possible existing-API design:

1. During private full-model warmup, collect snapshots of each layer's shared
   branch buffers (normalized input, activation, shared output, scalar gate and
   any bias slots) over the same exclusive Scratch lease. A foreign snapshot
   owner can hold pointer/extent/actual context plus the lease, without another
   canonical user-owner token. This is unsafe/private: aliased GPU accesses must
   be ordered and none of these buffers can escape the DecodeGraph checkout.
2. After warmup sync, outside the DeviceOp execution lock, capture one child
   CudaGraph per shared layer. Child capture contexts own functions/weights/
   snapshots for their entire replay lifetime. Check take_output before reuse.
3. Main capture forks an alt stream after post-norm, launches the corresponding
   child graph there, records routed work on main, and joins before merge and
   before either bank can be reused. Main compute must acquire the same scratch
   slots for shared output placeholders but skip their ordinary kernel launches.
4. Alt ordinary ExecutionContexts are kept in a vector OUTSIDE the parent capture
   closure and dropped only AFTER CudaGraph::capture returns (including errors),
   so cleanup does not query a capturing stream. Always attempt join even when
   the child launch fails; let parent capture RAII end/clean capture on all paths.
5. Parent capture context explicitly retains the child graph collection. Parent
   replay therefore owns all child functions/buffers through GPU completion,
   including model/cache drop and cancellation. DecodeGraph checkout exclusivity
   protects private scratch; immutable private weights need only read access.
   Parent output/readback and real session/KV owners keep their existing leases.
6. Verify that cuGraphLaunch is supported during capture on this CUDA driver,
   creating child nodes (consult official NVIDIA docs / a minimal reproducer).
   No assumption or private library hack is acceptable.
7. Add Gate-based lifetime/order tests for parent-held child graphs; retain all
   module owners and never rely on the JIT cache staying populated. Then tiny and
   real BF16 graph/eager/chunking checks, followed by rotated native comparison.

Alternative small remaining costs after parity analysis: fuse residual add +
RMSNorm using an explicitly BF16-rounded sum before FP32 variance; preserve one
reduction tree across token counts. Carry the final merged scratch guard into
next layer, fuse x=residual+merged with next input norm, drop guard before bank
reuse; fuse mixer residual add with post-norm too. Last layer still adds before
head. Existing CPU finite-scan fold prototype is in /tmp/qwen35-finite.rs.

## Better route found in official CUDA 13.2 documentation

https://docs.nvidia.com/cuda/archive/13.2.0/cuda-runtime-api/group__CUDART__STREAM.html
(cudaStreamGetCaptureInfo and cudaStreamUpdateCaptureDependencies)
https://docs.nvidia.com/cuda/archive/13.2.0/cuda-driver-api/group__CUDA__GRAPH.html
(cuGraphAddChildGraphNode)

The capture-info graph handle is borrowed; graph operations other than destroy /
node removal are allowed while capturing. The dependency array remains valid
until another stream operation or capture termination, and handles can be copied.
Capture dependency flags 0 means ADD, not SET; use the named SET constant.

Prefer this over capturing GraphLaunch (which may be unsupported):
- During each SDK CudaGraph child capture, query/save its raw CUgraph handle via
  cuStreamGetCaptureInfo. Keep the SDK child graph Arc alive; do not destroy the
  borrowed raw handle. Its capture context owns all module/resource descriptors.
- Use ONE precreated alternate stream per DecodeGraph, retained through parent
  capture/graph lifetime. alt.join(main) forks after post-norm. Query alt's current
  capture graph and copy its dependency handles. Add a child graph node using the
  borrowed child root, then SET alt's capture dependencies to that new child node.
- Record routed work on main, main.join(alt), then merge. Always attempt joins on
  errors and let SDK capture RAII end/reset capture before releasing stream owners.
- Parent capture context retains the child-graph collection, guaranteeing modules
  and snapshots survive parent execution/cache/model drops. No ordinary alt
  ExecutionContext or unsafe async_on is needed. Private shared snapshots are
  exclusively owned by this DecodeGraph; child replay access registration is
  unnecessary only under that explicit ownership/order proof. Never expose those
  intermediate aliases or let cache checkout protection be bypassed.
- This needs a minimal on-host reproducer and Gate lifetime/order test before
  inference integration. APIs are public CUDA APIs, not SDK private-field access.

## Other measured-cost followups

cuda-async 0.5 DeviceFuture supports CUDA_ASYNC_SPIN_BUDGET_US (default20 us,
0 disables) and CUDA_ASYNC_HOST_SYNC=spin/block. Both are currently UNSET in
our benchmark environment. Default completion uses a flag-write reactor; it
spins10000 passes then yield_now while work remains (not a 1ms sleep). A bounded
inline budget around 15000 us would cover this B1 decode and avoid a waker hop,
but blocks that executor thread during the wait. This can be an explicit B1
latency configuration (SGLang bench synchronizes on its caller), not a silent
production default or a serving-throughput claim. Screen after profiling CPU
tail; preserve full-logit validation. If useful, add a bench-only CLI flag that
sets the existing SDK env BEFORE the first device future; no engine API/global
library default change. Measure same-binary default vs spin, record CPU tradeoff
and rerun Gate ownership checks. Do not claim extra-performance merely by moving
validation outside timing. Explicit host-sync=spin may help or hurt; measure.

Residual add+norm sum in the OLD profile is only0.207ms/step combined; fusion
might save0.08ms (<1%e2e). Defer until larger changes are exhausted and parity
still fails, per user rule. Other small kernels/argmax are already negligible.
Possible expert MMA tuning if still large: change ONLY BF16 output tile32->16
(in separate const, leave SIMT/f32 fallback32), retaining BK64; inspect static
code and qualify numerical equality. More CTAs may help the larger6000D, but
Qwen2 N16 was rejected on the older36-SM5060Ti, so never assume a gain. Reprofile.
