# Shared expert gate/up/SwiGLU fusion

Baseline `6ca209d`; native BF16 B1/32-input/8-output graph comparison, three
processes/variant, rotated order, two warmups and three measured samples each,
CPU affinity 24-31. Full logits validation and host feedback remain enabled.
Medians of process medians; decode samples average seven steps. No observations
excluded, no GPU tests/compilation/profiling overlap timing, clocks unlocked.

| Variant | Prefill ms | Decode ms/token | Total 8 outputs ms |
| --- | ---: | ---: | ---: |
| Baseline | 31.630 | 9.489 | 97.979 |
| Consistent fused vector | 33.009 | 8.995 | 96.035 |

Decode -5.21%, e2e -1.98%; prefill regresses 4.36% (+1.380 ms).
Every paired process improves decode and e2e. Candidate decode process medians
9.022/8.995/8.980 ms versus baseline 9.512/9.489/9.404. The predeclared one-process
screen (95.720 ms/e2e) is retained separately and excluded from confirmation.
All IDs match, KV is released, and warm samples have no new graph capture or
scratch allocation. Total backing allocation count drops 142 -> 126; resident
scratch drops 10,729,868 -> 10,594,684 bytes (includes prefill and graph workspace).

The kernel calculates four output channels over full K per CTA and fuses both
projections with SwiGLU. Gate, up, activation and final output retain their BF16
rounding boundaries. BF16 uses the SAME reduction tree at all token counts to
preserve chunking consistency. Other dtypes retain the prior operators. The
measured prefill cost is accepted for the larger total/decode gain in this
workload; larger batches/longer contexts are not performance validated.

## Numerical gate and rejected attempt

The earlier B1-only vector path passed operator/tiny tests but failed real
chunking: max 0.31143212 > existing 0.02. It was rejected before performance
sampling and never committed. Its source/binary/logs remain in the parent
b1-only-* artifacts. Tolerances were NOT widened.

The consistent variant passes all 15 BF16-focused regressions, including tiny
reference, exact graph/eager logits, Runner, state operators and Gate lifetime
checks. Real 35B reference: max 0.56865, RMS 0.11685, matching first ID; all four
chunk variants exactly match full prefill. The operator oracle matches unfused
kernels bitwise at K128/N128, K2048/N512 and K2048/N513 (masked tail). All-target
cargo check and diff checks pass. No broad text-quality/serving claim is made.

Artifacts: all JSONL/stderr, summary.json, metadata/commands/binary hashes,
patch.diff, compare.py, screen-*, tiny-tests.log, real-reference.log and
all-target-check.log. Native e2e is below the earlier SGLang reference, while
Rust decode still exceeds SGLang's earlier 7.049 ms/token. Continue major gaps.
