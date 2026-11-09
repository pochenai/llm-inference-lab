# Profiling after packed a/b (`0c5dcc0`)

This separate Nsight node trace measures one sample after2 warmups, B1/32/8,
CPU24-31. These durations are attribution evidence, NOT native latency cells.
Native previous result:8.406ms/decode,91.975ms/e2e. Commands, hash, raw trace,
SQLite, analyze.py and analysis.json are preserved. All321 dense projections per
step are mapped by execution order and validated output-grid dimensions.

| Selected cost | Profile ms/decode step |
| --- | ---: |
| Expert MMA (80 kernels) | 1.770 |
| GQA (10 kernels) | 0.341 |
| GDN a/b (30 projections) | 0.212 |
| Shared gate/up fusion | 0.231 |
| Router | 0.112 |
| Scalar shared gate | 0.087 |
| Post-upload CPU prelaunch preparation (median) | 0.290 |
| Post-readback GPU-end -> CPU-validation tail (median) | 0.226 |
| Full finite scan (median) | 0.312 |

817 kernels per decode, down from927 before fusion; GPU kernel sum averages
about7.10ms. Kernel span includes the early KV bind and following CPU preparation,
so do not add prelaunch CPU time to that span. Node tracing increases observer
overhead; driver graph launch0.273ms here is not a native driver estimate.

Earlier prelaunch0.855ms and completion-tail0.536ms already decreased after
scratch identity reuse, but hundreds of immutable weight read leases remain.
Next try factory-only frozen weights retained by the graph capture owner, with
normal mutable-buffer tracking and Gate/context tests. GQA and the full CPU
finite scan are the next material targets. Dense large QKV/z/head remain close
to SGLang; expert GPU time now matches its earlier1.772ms sum. Minor add/norm
fusion is deferred (<1%e2e expected) unless larger work cannot reach parity.
