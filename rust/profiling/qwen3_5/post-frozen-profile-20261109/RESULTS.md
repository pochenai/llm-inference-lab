# Immutable-weight attribution check (`06aa36e`)

Separate Nsight node profile, B1/32/8,2 warmups then1 captured sample. Profile
values are attribution only; native result remains8.111ms/decode and90.146ms/e2e.

| Profile metric | Before frozen | After frozen |
| --- | ---: | ---: |
| Kernels/decode | 817 | 817 |
| GPU kernel sum ms | ~7.10 | ~7.10 |
| Post-upload CPU prelaunch median ms | 0.290 | 0.035 |
| Post-readback tail median ms | 0.226 | 0.093 |
| Full finite validation median ms | 0.312 | 0.312 |

All321 dense projections are order/grid validated. Per-role GPU durations
remain effectively unchanged, consistent with the unchanged GPU math/ABI.
The preparation/tail reduction agrees with the0.393ms native decode point
improvement; do not add profile intervals to native values or kernel spans.
Driver launch in node tracing remains instrumented and is not a native estimate.

Next: GQA0.34ms GPU (32-token tiled reads), then full CPU finite scan0.31ms.
Re-profile remaining costs before sub-1%e2e tuning. Commands/hash, nsys-rep,
SQLite, analyze.py and analysis.json are preserved.
