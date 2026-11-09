# Qwen3.5 BF16 graph replay storage reuse

Baseline: `70e8e87`; candidate source diff and executable hashes accompany this
report. Native unprofiled B1, 32 prompt tokens, 8 outputs, full logits validation
and host token feedback. Three independent processes per variant in rotated
order, CPU affinity 24-31, two warmups and three measured samples per process.
Cells are medians of process medians; decode samples average the seven steps.
No observations were excluded. Clocks are unlocked.

| Variant | Prefill ms | Decode ms/token | Total 8 outputs ms |
| --- | ---: | ---: | ---: |
| Baseline | 55.550 | 10.857 | 131.595 |
| Reused Tensor storage | 54.444 | 10.185 | 126.321 |

Decode latency falls 6.19%; total falls 4.01%; prefill falls 1.99%.
Process decode medians: baseline 10.923 / 10.827 / 10.857; candidate
10.481 / 10.185 / 10.033 ms. Candidate is faster in every paired repetition.
All generated IDs match (eight 9707s), every sample restores KV ownership, and
warmed samples allocate no workspace buffers and capture no new graphs.
Resident scratch remains 10,729,868 bytes and 142 backing buffers.

Scratch now lends its original unique Tensor owner from a slot, recovering it
at last CPU use instead of creating a fresh foreign storage per layer. Graph
resource deduplication therefore sees the reused storage. Explicit submission/
capture checkout retention and escaped output owners preserve GPU lifetimes.
A live slot cannot be checked out twice. No kernel arithmetic changes.

Validation: all 14 Qwen3.5 release tests, including opt-in tiny fixture tests,
passed. All-target cargo check passed. Final workspace fast-path recheck is
recorded separately before commit. No new per-resource CPU trace was collected;
this report does not assign the entire native gain to a specific CPU subphase.
SGLang's earlier 7.049 ms/token graph reference is still faster; this iteration
does not claim parity. See engine/task.md for subsequent iterations.
