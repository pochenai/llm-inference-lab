# Packed GDN a/b MMA projection

Baseline `d66bf6c`; native BF16 B1/32-input/8-output graph protocol, 3 rotated
processes per variant, 2 warmups + 3 measured samples, CPU 24-31, full logits
checks and host feedback. Cells are median(process median(sample)); decode
samples average seven steps. All observations retained, clocks unlocked. No
GPU tests/compilation/profiling overlap timing; completion settings unchanged.

| Variant | Prefill ms | Decode ms/token | Total 8 outputs ms |
| --- | ---: | ---: | ---: |
| Baseline | 33.282 | 8.671 | 94.220 |
| Packed a/b MMA | 33.018 | 8.406 | 91.975 |

Decode -3.05%, total -2.38%, prefill -0.79%. All three paired processes improve
all metrics. Decode process medians 8.365/8.406/8.486 versus 8.741/8.671/8.653.
IDs match, KV releases, warm capture count1 and allocation count are stable.
Cumulative scratch allocations 126 -> 122; resident scratch10,594,684 ->
10,594,680 bytes. Loaded weights remain69,321,574,912 bytes.

Weights A/B are concatenated once on CPU, preserving both source shape checks,
and uploaded as one [2*VH,H] tensor. The existing [16,64,64] MMA projects both;
borrowed column views retain the packed row stride. A forced-MMA helper prevents
other shapes from selecting the router vector bucket. No new GPU kernel or
extra resident weight copy; original model files and formats are unchanged.
The change removes30 projection launches per decode step.

Precision oracle compares2316 individual FP32 accumulator bits against separate
BF16-input MMA calls at real K2048/VH32 with rows1/3/32 and tiny K128/VH2. All
match exactly. All18 Qwen3.5 regressions pass (including the existing FP32
correctness control because layout changed for generic T). Real BF16 reference
is unchanged: max0.56865, RMS0.11685, first ID3636; all four chunks max0. All-target
and diff checks pass. No FP32 performance comparison or tolerance change.

Raw samples, hashes, source diff, commands, summary.json, compare.py, oracle.log,
tiny-tests.log, real-reference.log and all-target-check.log accompany the report.
No larger batch/context performance or broad text-quality claim. Earlier SGLang
graph decode7.049 ms/token remains faster; re-profile the remaining gap next.
