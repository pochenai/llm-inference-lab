# Qwen3.5-35B-A3B: initial single-GPU latency baseline

RTX 6000D, CUDA 13.2, cuTile 0.5.0, SGLang 0.5.9, CPU affinity 24–31 on NUMA 0. Native BF16 checkpoint; SGLang uses explicit Triton attention and FP32 Mamba SSM state. B1, 32 repeated token-9707 inputs, exactly eight outputs, fresh session state. Two warmups and three measured samples per process; three independent processes/engine. Eager engine order alternates. SGLang graph references run separately after eager samples. No profiler, compilation or GPU tests overlap native timing. CPU/GPU clocks and unrelated host load are not controlled.

Each cell is the median of three process aggregates: mean prefill/total, median of sample median decode steps.

| Engine | Prefill ms | Decode ms/step | Total for eight outputs ms |
| --- | ---: | ---: | ---: |
| rust | 59.164 | 29.593 | 264.983 |
| sglang | 94.377 | 67.576 | 540.079 |
| sglang-graph | 95.206 | 7.292 | 146.487 |

## Measurement boundaries

- Rust directly calls the pooled CausalModel forward, with GPU argmax, full-logit D2H/finite-value validation and completed host token feedback. Lazy recurrent-state initialization is included in prefill. Weights, JIT and pool backing storage stay warm. No Scheduler, HTTP, tokenization or text decode is timed.
- SGLang uses the existing static-batch adapter, GPU sampling and explicit Triton full attention/GDN. Graph accelerates decode; prefill remains eager. Vision weights are resident in its conditional-generation wrapper, but these samples contain no images or encoder work. Pool capacities differ, so this is not an allocator-efficiency comparison.
- Rust has no Qwen3.5 graph or GPU-feedback path yet. Its eager boundary differs from the graph reference. The observed decode gap is an optimization target, not an isolated kernel or quality ranking.
- SGLang reports no RTX 6000D-specific MoE tuning file and uses its default MoE config; this is the stock installed baseline, not a claim about the fastest tuned SGLang.
- All nine measured Rust samples reproduce the same complete generated IDs, run exactly eight forwards, and release session resources. SGLang samples retain input checksums and timing records but not generated IDs; cross-engine history/output equivalence is not established.

Rust eager / SGLang graph decode ratio: **4.06x**. Full-cohort ratio: **1.81x**.

## Independent process observations

| Engine | Prefill ms by process | Decode ms by process |
| --- | --- | --- |
| rust | 59.361, 59.071, 59.164 | 31.753, 29.593, 28.150 |
| sglang | 94.377, 95.250, 73.198 | 67.576, 67.648, 49.647 |
| sglang-graph | 95.734, 95.206, 68.741 | 7.293, 7.292, 7.042 |

Process variation is material, especially in eager Python submission; no confidence-bound improvement claim is made. Frozen native Rust source/binary, raw samples, commands and metadata accompany this report. The separate profiling executable adds only profiler-range control; its source/binary and trace are retained separately. Failed language-only and automatic-attention startup attempts are excluded and preserved in failed-*.stderr.

## Separate Nsight attribution

The measured capture contains one prefill and seven decode forwards after two
warmups. Eight `argmax_final_entry` boundaries partition the trace; no kernels
remain outside those eight groups. Profile timings are diagnostics and are not
pooled with native samples or subtracted from native wall times.

- 18,260 kernel launches; 10,852 allocation calls and 10,852 free calls;
  10,844 event create/record/destroy calls each.
- Prefill: 50.99 ms summed GPU kernel time, with expert kernels at 80.8%.
- Decode: 9.74 ms summed kernel time per forward. Linear projection is 51.7%,
  expert kernels 24.7%, and zero-fill kernels 9.3%. This is kernel duration sum,
  not host wall latency or a measured overlap ratio.

The first optimization target is persistent, exclusively leased workspace
storage, following the existing Qwen2 workspace lifetime pattern. Distinguish
fully overwritten scratch from persistent recurrent/KV state; preserve state
and padding initialization contracts. The 0.5 buffer/tensor shared-storage APIs
can supply bounded views without reallocating each intermediate. Verify reuse
and cancellation with warmed Gate-based ownership tests before benchmarking.
Next add GPU-feedback/graph execution with stable addresses, then tune the
linear/expert kernels. Recheck numerical results on common token histories and
extend the workload beyond B1/32 before making broader performance claims.

## Validation

All workspace targets compile. The full migration suite passed 133 tests and
four doc-tests; two opt-in real-Qwen2 regressions and three Qwen3.5 fixture/
reference/pooled-state regressions passed separately. Five invalid-CLI cases
passed. Formatting, Python compilation and diff checks passed. The required
root offline pyright command could not start because the locked CPU PyTorch
wheel is absent from its cache; this is not a passing pyright result.
