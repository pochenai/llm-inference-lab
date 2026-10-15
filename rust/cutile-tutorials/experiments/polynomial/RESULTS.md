# FMA and polynomial evaluation experiments

Collected on 2026-10-15 (Asia/Shanghai), RTX 5060 Ti 8 GiB, compute capability 12.0,
CUDA Toolkit 13.3, Nsight Compute 2026.2.1, cuTile Rust 0.4.0 from Cargo's registry.
Rust builds use `--release`. Device compilation uses optimization level 3, device debug
disabled, and line information enabled. No fast-math mode was enabled.

The main finding is that FMA can approach a 2x throughput improvement when arithmetic is
the limiting resource. A one-step vector transform cannot demonstrate that improvement:
it mostly transfers data. The original polynomial configuration was limited by coefficient
redistribution and small output tiles. Increasing the tile and changing coefficient access
exposes a useful FMA improvement, without expanding the coefficients in device memory.

## Files and reproduction

From the `rust/` workspace:

```sh
cargo build --release -p cutile-tutorials \
  --example polynomial_eval --example fma_vs_mul_add

# One vector affine transform: y = alpha*x + bias.
cargo run --release -p cutile-tutorials --example fma_vs_mul_add -- --iterations 1

# One dependent recurrence per input element.
cargo run --release -p cutile-tutorials --example fma_vs_mul_add -- --iterations 512 --chains 1

# Arithmetic-heavy control: four independent chains, 512 updates per chain.
cargo run --release -p cutile-tutorials --example fma_vs_mul_add -- --iterations 512 --chains 4

# Optimized polynomial, GPU graph timing and the previous eager event interval.
cargo run --release -p cutile-tutorials --example polynomial_eval -- --eager
cargo run --release -p cutile-tutorials --example polynomial_eval -- --cache cold

# Save the actual Tile IR, optimized cubin, SASS, embedded debug PTX text, and resource usage.
cargo run --release -p cutile-tutorials --example polynomial_eval -- \
  --inspect-dir cutile-tutorials/artifacts/polynomial/sass
```

`examples/fma_vs_mul_add.rs` isolates arithmetic. The default
`examples/polynomial_eval.rs` uses scalar coefficient loads and a 1024-element output
tile, with the coefficient count still a JIT const generic. CPU-generated coefficients
are reproducible; changing `--coefficients` regenerates the correct number of values.
Both examples depend on `examples/support/mod.rs` for argument parsing, graph timing,
and assembly inspection.

The exploratory variant example and Python sweep/profiling scripts were removed after
the investigation. Their measurements are retained below as historical results. The
commands above reproduce the two retained examples, not the removed coefficient-layout
and storage-cliff experiments.

Raw output and exit codes are in `artifacts/polynomial/runs.jsonl`; parsed tables are
`sweep.csv`, `extended.csv`, and `final.csv`. The journal contains 102 successful benchmark
processes. Additional manual inspection runs and NCU runs are separate. NCU reports,
decoded metrics and source-correlated assembly stay under `artifacts/polynomial/`, which
is ignored by Git. This report is kept outside the ignored artifact directory.
The final source snapshots, SHA-256 hashes and tool version output are archived as
`artifacts/polynomial/final_sources/` and `version_metadata.json`.

## Timing and numerical validation

The paired samples alternate A/B and B/A. Both graphs use the same input values, arithmetic
parameters, output size, coefficient values, and device compilation settings. JIT compilation,
allocation, upload, graph capture and coefficient expansion are outside the timed region.

CUDA timing events are **inside** the captured graph. `CU_EVENT_RECORD_EXTERNAL` forces
actual event record nodes; ordinary event record calls during capture can become only
dependency edges and cannot provide the required elapsed-time result. Events remain alive
until the graph is destroyed.

Warm mode captures 16 repeated kernel nodes and reports the graph's device interval divided
by 16. This is steady-state per-kernel graph throughput and includes inter-node scheduling;
it is not a claim that graph scheduling costs nothing. Cold mode captures one kernel node
and flushes a buffer at least twice the L2 size before each sample. The flush is outside the
timing events. Cold and warm results are therefore reported separately.

The previous `do_bench` protocol records its start event before the Rust launcher closure.
That interval can include GPU idle time while the CPU prepares a launch. A warm-cache
`--eager` comparison reproduced about 34.1/29.4 us for ordinary/FMA Horner, while graph-internal
timing gave 13.1/9.5 us for the same kernels. A preceding cold-cache flush can overlap CPU
submission and change this effect. Accordingly, eager event intervals, graph throughput and
NCU's instrumented kernel duration must not be treated as interchangeable measurements.

Each polynomial run checks **every output element** against CPU f64 Horner using the actual
f32 x values and coefficient values. Tolerance is `2e-5 + 2e-5*abs(reference)`; observed errors
in the final default run were 5.21e-8 for naive, 4.21e-8 for Horner, and 3.82e-8 for FMA.
The standalone arithmetic example checks 4096 sampled positions against a CPU f64 recurrence,
checks every output for finite values, and reports the maximum difference between the two
GPU outputs. For 512 steps and four chains, separate arithmetic's sampled error was about
3.18e-5, FMA's about 2.46e-6. Different rounding is expected; bitwise equality is not required.

## Independent FMA control

N=1,048,576, tile=256, default worker configuration, warm graphs, 100 paired samples:
The recurrence uses runtime alpha=0.99931 and bias=0.000127, with x in [-0.9,0.9].
The four-chain case initializes chain k with x+0.1*k. Polynomial runs use x in [-1,1]
and the fixed CPU generator seed 0x5eed_cafe, with coefficients in [-0.02,0.02].

| Recurrence steps | Independent chains | Separate mul/add (us) | Explicit FMA (us) | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 1 | 5.56 | 5.48 | 1.015x |
| 32 | 1 | 10.41 | 8.25 | 1.263x |
| 32 | 4 | 28.02 | 16.74 | 1.674x |
| 128 | 1 | 26.92 | 20.02 | 1.344x |
| 128 | 4 | 94.40 | 50.36 | 1.875x |
| 512 | 1 | 92.52 | 68.32 | 1.354x |
| 512 | 4 | 359.94 | 183.48 | 1.962x |

A one-step `alpha*x + bias` transform is SAXPY/affine evaluation. Vector addition alone
has no multiplication to fuse. With one step, changing two FP instructions into one barely
changes total time. In the four-chain case, independent arithmetic supplies ILP and substantial
work after the input load. In cold mode the 512-step/four-chain result was 371.39 vs 186.82 us
(1.988x); the one-step result was 22.24 vs 22.27 us (no measurable gain).

NCU confirms the arithmetic-heavy case: the FMA arithmetic pipes operate at about 93-94%
of their peak, DRAM utilization is only 2.4-4.8%, and executed warp instructions decrease from
139,001,856 to 70,303,744. In the one-step case DRAM utilization is about 80%, while arithmetic
pipe utilization is low. The saved SASS contains separate `FMUL`/`FADD` in the baseline and
`FFMA` in the fused version: the compiler did not make the two variants identical.

## Coefficient representation

Same degree-31 polynomial, N=1,048,576, tile=256, warm graph medians:

| Representation / access | Coefficient allocation | Horner (us) | FMA Horner (us) |
|---|---:|---:|---:|
| Original `[C,1]` tile, extract and broadcast | 128 B | 16.28 | 15.58 |
| Scalar `[1,1]` load for each coefficient | 128 B | 17.92 | 16.42 |
| Transposed `[1,C]` coefficient tile | 128 B | 16.28 | 15.61 |
| Pre-expanded `[C,N]`, contiguous coefficient rows | 128 MiB | 337.20 | 337.80 |
| Pre-expanded `[N,C]`, strided coefficient columns | 128 MiB | 438.76 | 438.54 |
| Two independent even/odd Horner chains | 128 B | 18.36 | 16.51 |

The contiguous pre-expansion kernel took about 294 us in an initial one-shot measurement;
its strided preparation was much slower. These preparation times are supplementary single
samples. They are excluded from evaluation timings and still need to be paid when coefficients
change. Expansion removes in-kernel coefficient broadcasts but replaces 32 shared coefficient
values with 32 values **per input element**. It reads 128 MiB of coefficients on every evaluation.
Even excluding preparation, it loses badly; no amortization count makes this measured evaluation
faster than the compact coefficient methods.

The safe scalar and expanded methods were also compared with host-validated unchecked loads.
Unchecked scalar access sometimes saved about 7-9% at smaller tiles and was essentially tied
at tile=1024. The expanded unchecked method remained about 337 us. Device bounds checks were
not the main explanation for the large differences. The final example keeps safe accesses.

Chunked methods load four or eight coefficients at a time. With tile=2048, chunk4 gave
11.56/9.27 us, while chunk8 gave 21.86/8.93 us. These are useful alternatives, but they require
coefficient counts divisible by the chunk size and have configuration-dependent ordinary
Horner performance. Scalar loading was selected for the main example because it accepts
arbitrary positive coefficient counts and gives a stable paired comparison.

## Tile size and compiler storage choices

N=1,048,576, 32 coefficients, warm graph medians:

| Tile | Original coefficient tile: Horner/FMA (us) | Scalar coefficient loads: Horner/FMA (us) |
|---:|---:|---:|
| 64 | 52.54 / 52.48 | 57.47 / 55.96 |
| 128 | 26.70 / 26.65 | 29.17 / 28.45 |
| 256 | 16.39 / 15.67 | 18.17 / 16.63 |
| 512 | 12.31 / 12.33 | 13.21 / 12.04 |
| 1024 | 429.44 / 10.88 | 12.94 / 9.38 |
| 2048 | 223.49 / 222.47 | 14.47 / 9.26 |
| 4096 | not retested | 14.16 / 11.34 |
| 8192 | not retested | 17.94 / 13.11 |

The original 64-element tile incurs coefficient communication for many small CTAs.
Increasing the output tile amortizes this work. Saved SASS for the 32-coefficient tile
contains 31 `SHFL.BFLY` and 62 `FSEL` instructions to redistribute coefficients into registers.
Line information points to coefficient extraction. This is register communication, not a
new HBM/global load for each broadcast. The compiler hoists much of this redistribution
before the polynomial recurrence. Larger output tiles also increase per-thread state.

The 1024-element original ordinary Horner is a **compiler storage cliff**, not a 40x arithmetic
benefit from FMA. Its embedded PTX declares `__local_depot0[256]`; its main path has generic
`ST.E`/`LD.E` to coefficient temporaries. NCU measures 467.84 us and heavy LG queue throttling
(133.72 stalled cycles per issued instruction), while the fused variant measures 16.67 us.
NCU's specific register-spilling-request counters are zero in both. These are compiler-created
local stack temporaries rather than evidence from that counter of late register allocation spills.
The exact backend heuristic deciding to materialize them is not established by this experiment.

Requesting eight worker warps for that same original tile removes the cliff: NCU measures
17.82/17.63 us and 44/48 registers per thread, with a 256-thread block. Requests for one or two
worker warps did not reduce the effective block below 128 threads in this toolchain; the saved
PTX for the one-warp request still says `.reqntid 128`. Hints must be checked against emitted
launch metadata rather than interpreted as guaranteed thread counts.

At 64/128 coefficients, the original full-coefficient tile at output tile=1024 degraded to
about 1.75/6.96 ms. Scalar loading stayed at 22.28/41.08 us for ordinary Horner and
17.61/31.52 us for FMA. A 31-coefficient scalar case also passed numerical validation;
the even/odd case handled its odd final coefficient correctly. Full coefficient tile dimensions
are constrained to powers of two; scalar access does not have that requirement.

## NCU evidence and final selection

Each report captures the first ordinary and fused matched kernel, with identical parameters.
NCU replays/instruments kernels and generally flushes caches, so these durations are reported
separately from the warm graph measurements. Percentages normalize different units and overlap;
they are not exclusive fractions of wall time.
The collection uses NCU's `cache-control=all` and `clock-control=boost` settings.

| Profile case | Duration ordinary/FMA (us) | DRAM ordinary/FMA | L1/TEX ordinary/FMA | LSU ordinary/FMA |
|---|---:|---:|---:|---:|
| Original coefficient tile, B=64 | 60.70 / 60.32 | 15.7% / 15.8% | 93.9% / 94.2% | 91.2% / 91.6% |
| Pre-expanded coefficients, B=256 | 336.90 / 338.43 | 96.0% / 95.5% | 15.2% / 15.3% | 10.9% / 10.8% |
| Final safe scalar loads, B=1024 | 18.78 / 13.95 | 50.9% / 68.7% | 27.0% / 38.5% | 24.1% / 32.6% |

The original configuration spends its budget on the LSU/L1/MIO side, including coefficient
redistribution; its arithmetic pipes are far from the compute-heavy control's saturation.
FMA reduces math instructions while leaving that communication and small-CTA overhead.
The pre-expanded configuration is genuinely device-DRAM-bandwidth limited. The final scalar
configuration removes the coefficient redistribution and reduces CTA count; its lower LSU
pressure lets the reduced arithmetic instruction count affect duration.

The final source uses tile=1024 for both Horner methods. Three independent 200-pair warm runs
gave 13.096/9.516, 13.112/9.526, and 13.098/9.520 us. FMA speedup was 1.376x in every run
(about 27.3% less time). Against the original B=64 coefficient-tile implementation, ordinary
Horner improved about 4.0x and FMA about 5.5x under the same warm graph timing protocol.
The safe main example has no executed register spill requests in its NCU report.

With a cache flush before every one-node replay, the final ordinary/FMA medians were
24.32/22.24 us, only 1.094x. Cache residency, output tile shape and arithmetic intensity are
therefore all material to the observed FMA effect. A general claim that explicit FMA always
halves polynomial runtime is unsupported. Conversely, the lack of a speedup in the original
configuration does not imply that its multiplication and addition were automatically fused.

The measured coefficients, device, compiler and batch sizes define the scope of these results.
The report retains the losing alternatives, cold/warm distinctions, numerical errors and
storage cliffs as evidence for the conclusions and context for future comparisons.

## Validation recorded before cleanup

Before cleanup, all three experiment examples built successfully in release mode. Rust
formatting and Git whitespace checks passed. The two Python scripts passed bytecode
compilation and explicit Pyright checking with no errors or warnings. These statements
describe validation of the recorded experiment sources, including the subsequently removed
variant example and scripts.

The final source was run with 32, 31 and one coefficient. The single-coefficient case evaluates
a constant polynomial and has zero error. Its empty Horner loop needs a `C > 1` const condition:
without that guard this cuTile frontend still checks a negative index inside the zero-iteration
loop. The former variant example skipped incompatible full-tile/chunk layouts with odd
coefficient counts. All compatible variants passed the full CPU reference check in that run.

After these guards were added, a final 200-pair run reproduced 13.102/9.516 us for ordinary/FMA
Horner (1.377x), and the local-array cliff still reproduced 428.21/10.84 us. Their output and
fresh disassembly are retained in `artifacts/polynomial/final_verification.txt`,
`cliff_verification.txt`, and `sass/`. The last independent compute-heavy control also reproduced
1.962x. These checks support the conclusions without treating the CPU-reference tolerance or
SASS static instruction count as a substitute for measured GPU performance.
