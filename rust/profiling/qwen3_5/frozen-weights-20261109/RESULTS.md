# Capture-owned immutable model weights

Baseline `0c5dcc0`; BF16 B1/32-input/8-output graph mode, three rotated processes
per variant,2 warmups+3 measured samples, CPU24-31. Full-logit validation and host
feedback stay enabled; completion settings unchanged. Median(process median
(sample)); decode samples average7 steps. All observations retained, clocks
unlocked. No GPU tests/compilation/profiling overlap native timing.

| Variant | Prefill ms | Decode ms/token | Total8 outputs ms |
| --- | ---: | ---: | ---: |
| Baseline | 33.249 | 8.504 | 92.600 |
| Frozen weights | 33.368 | 8.111 | 90.146 |

Decode -4.62%, e2e -2.65%; prefill +0.36% (+0.119ms). Every paired process
improves decode/e2e. Decode process medians baseline8.504/8.394/8.535 versus
candidate8.164/8.111/8.097. IDs match and sessions release KV. Warm capture and
scratch counts stay fixed, with no new allocation/recapture. Loaded weight
bytes,122 backing allocations and10,594,680 resident scratch bytes are unchanged.

Private FrozenWeight has only a fresh-upload factory that awaits completion;
there is no tensor/buffer/mutable accessor. This prevents inheriting an existing
captured writer, which an arbitrary from_tensor constructor would not prevent.
Eager retention delegates to normal cuTile tracking. Capture validates the
actual context and registers an owning tensor Arc for the graph's full lifetime;
permanent read-only storage then needs no per-replay shared-access leases.
Mutable scratch/state/KV retain their existing ownership/access checks. All
shape/stride/spec/dtype/ABI methods delegate to the original tensor. No GPU
math, kernel code or benchmark-boundary changes.

All20 Qwen3.5 regressions and all-target checks pass. Focused Gate test drops
CPU weight/graph owners during pending replay, observes retained storage, then
completion and release. Rescue allocation+drain guards avoid use-after-free on
assertion failures. A separate real user CUDA context on ordinal0 is rejected
for eager and capture. Default completion tests pass; forced notification was
also exercised. Real BF16 reference remains max0.56865/RMS0.11685/ID3636, all
four chunk variants max0. No threshold changes or FP32 performance claims.

Raw samples, hashes, commands, source diff, copied weights.rs (new source was
untracked during measurement), native summary, lifetime/model logs accompany
this report. Earlier post-packed profile motivates the change; a fresh trace
will check CPU attribution. Native decode still exceeds SGLang's earlier7.049ms
reference. Larger batch/context and serving throughput remain unmeasured.
