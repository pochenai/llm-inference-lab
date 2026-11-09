# Potential replay optimization after updated profiling (not implemented)

Current model weights are private immutable Arc<Tensor>, ~660 separate captured
resources. Scratch identity dedup fixed >1000 redundant resources, but each replay
still registers/releases hundreds of read accesses. Re-profile after packed AB.

Possible safe existing-public-API solution: private FrozenWeight<T> owns
Arc<Tensor<T>> and exposes ONLY shape/context/consuming reshape (NO DerefMut,
NO tensor/raw-buffer getter; ideally no Deref either). It can never be a kernel
output. Constructor takes a completed uploaded owned Tensor; inference never
mutates weights. Implement local IntoDeviceOp<Self> -> Value<Self>, KernelInput<T>
Stored/Returned=Self, and unsafe KernelInputStored by delegating ABI/shape/stride/
spec/dtype methods to its inner Arc<Tensor>.

retain(ctx): in eager mode delegate ordinary Arc<Tensor>::retain(ctx), preserving
all usual tracking. In capture mode validate ACTUAL context, then ctx.retain the
inner Arc<Tensor> WITHOUT recording a ReplayResource. Parent GraphExec capture
context owns frozen tensors for every later replay; SDK launch validates actual
context. Exclusive writer access is impossible through the private read-only type,
so no per-replay shared access leases are needed. Mutable scratch/session/KV
resources keep their existing tracking. Do not bypass ownership or contexts.
This is a nontrivial unsafe trait contract: prove immutability and no safe mutable
alias/escape, not just "we don't currently write". Arc owner retention also blocks
safe Arc::get_mut access. Raw/mutable unsafe access remains forbidden by invariant.

cuda_async IntoDeviceOp is not blanket for KernelInput: implement explicitly for
the local struct (Arc<local> orphan rules would be awkward). KernelInputStored ABI
can delegate to inner Arc<Tensor> with UFCS. Need AsyncKernelLaunchBuilder and
SpecializationBits imports from the tensor.rs upstream header.

Use alias Weight<T>=FrozenWeight<T>, Norm=Weight<f32>. Model load helper returns
Weight<W> via new(Tensor); conv, packed AB, frequencies likewise. KV GraphData
Arc<Tensor> remains mutable/unmodified. Existing gu/down weights load 3D then
reshape into 2D: consuming reshape can Arc::try_unwrap the sole private fresh
Tensor, reshape it and wrap again; reject if shared. Engine accesses only
shape/context (verify other methods before implementation). No GPU kernel/ABI
math changes; all input metadata must remain byte-identical.

Required tests: warm compilation/allocation/completion BEFORE Gate. Capture a
simple last_row child graph reading frozen [1,128] weight into normal output;
drop external weight, launch graph behind gate, drop CPU graph owner and assert
weak tensor owner survives pending work, then release/await and ensure it can
retire. Test eager retention too if ordinary delegation unchanged not necessary.
Reject another ACTUAL context on same ordinal (both eager and capture). Re-run
all model/lifetime tests and real BF16 logits; native rotated comparison.

Potential gain ~0.2-0.4ms/step if profile confirms residual read-access overhead.
Prefer this to more intrusive parallel scratch aliasing / CPU busy wait, if sound.

IMPORTANT constructor refinement: owning an arbitrary Tensor is not enough:
a safe previously captured graph could still hold a write node for it, despite
Arc user-owner counts. Do NOT expose a safe from_tensor constructor. Best
invariant-by-construction API is a safe async from_host(Vec<T>, shape, policy)
factory INSIDE weights.rs that creates a fresh tensor, awaits the upload, then
wraps it; no tensor/pointer can escape before freezing. Private struct fields,
no Deref/tensor/buffer getter. Engine load helper keeps all file shape checks
then calls from_host. Conv transpose/AB pack/frequencies use the same factory.
This avoids unsafe constructor call sites while proving no prior captured or
in-flight writers. A private unsafe from_completed_tensor would also need an
explicit no-writer-alias contract, but is less preferable.
