# BF16 routed-expert tensor-core iteration

Baseline: `58b3fbe`. Same native B1/32-input/8-output graph protocol and median
aggregation as replay-storage-20261108: three processes/variant, rotated order,
two warmups, three measured samples, CPU 24-31, full logits checks and host
feedback. No concurrent GPU jobs, compilation or profiling during measurements.
All observations retained; clocks unlocked.

| Variant | Prefill ms | Decode ms/token | Total 8 outputs ms |
| --- | ---: | ---: | ---: |
| Replay baseline | 54.419 | 10.155 | 125.251 |
| BF16 expert MMA | 31.775 | 9.564 | 98.790 |

Prefill -41.61%, decode -5.82%, e2e -21.13%. Decode process medians:
baseline 10.172/10.079/10.155, candidate 9.564/9.452/9.829 ms.
Every paired process improves all three metrics. All generated IDs match and
all warmed samples keep one graph, 142 backing buffers and 10,729,868 scratch
bytes, with zero recapture/allocation and fully released request KV.
The earlier SGLang graph reference is 7.049 ms/decode and 119.244 ms/e2e:
e2e is now lower, but decode parity has not been reached. A fresh cross-engine
comparison will follow the remaining major decode optimizations.

BF16 uses a 16-row MMA accumulator per selected expert, FP32 accumulation, and
the original BF16 output cast. Only the first row is stored. IDs/input-repeat
mapping are unchanged. Other storage types retain the original SIMT fallback.

## Validation and emitted code

- Independent exact CPU sums and old-SIMT comparison cover real K2048/N1024
  gate/up and K512/N2048 down, repeated/non-monotonic IDs and both input-repeat
  mappings. Every output matches bitwise in this power-of-two operand fixture.
- Fourteen BF16-focused Qwen3.5 tests passed, including tiny independent
  reference (max 2.38e-7), graph/eager exact logits, chunking, Runner and gated
  lifetime checks. The unchanged FP32 pooled control was excluded this round.
- Real 35B reference passed its existing budget (max 0.66619, RMS 0.12629,
  matching first ID); all four chunked histories exactly match full prefill.
- All-target cargo check and diff checks passed. No tolerance was widened.
- NCU counters are blocked with ERR_NVGPUCTRPERM. A diagnostic-only test enabled
  the cuTile 0.5 filesystem JIT store; header/checksum-validated cubins were
  disassembled with cuobjdump. Both old kernels have zero HMMA; both new kernels
  contain HMMA.16816.F32.BF16 (20/32 static instructions). Registers drop
  255 -> 128/thread; static shared memory grows 1,108 -> 33,956/44,244 bytes;
  new stack frame is 32 bytes. These are static facts, not achieved utilization.
  Instrumentation was removed and is absent from the native bench executable.

Raw artifacts: summary.json, all JSONL/stderr, metadata/commands/binary hashes,
patch.diff, oracle.log, tiny-tests.log, real-reference.log, all-target-check.log,
and compute/ (cubins, SASS, diagnostic instrumentation and NCU rejection logs).
Numerical checks do not establish broad text-quality or serving performance.
