# cuTile Tutorials

Small, runnable examples for learning cuTile Rust. Each example is a standalone Cargo example
under `examples/`.

## Run

From this directory:

```sh
cargo run --example hello_world
```

The workspace uses cuTile 0.4.0 from Cargo's registry. A compatible CUDA Toolkit, `tileiras`,
NVIDIA driver, and CUDA GPU are required to compile and run kernels.

## Polynomial evaluation

`polynomial_eval` evaluates a degree-31 polynomial by default for 1,048,576 `f32` inputs in
`[-1, 1]`, using direct powers, Horner, and Horner with explicit FMA. Its loops are specialized
by the coefficient count. Coefficients are generated on the CPU and loaded at runtime.
The measured default uses scalar coefficient loads and a 1024-element output tile.
Run from the Rust workspace root:

```sh
cargo run --release -p cutile-tutorials --example polynomial_eval
cargo run --release -p cutile-tutorials --example polynomial_eval -- --coefficients 64 --tile 1024
cargo run --release -p cutile-tutorials --example polynomial_eval -- --cache cold --eager
```

Timing events are inside CUDA Graphs to exclude the CPU submission gap. Warm mode measures
16 consecutive kernel nodes per graph; cold mode flushes L2 before each one-node graph.
Horner and FMA are measured in alternating A/B and B/A order. Every polynomial output is
checked against a CPU f64 reference. `--eager` also prints the traditional outer-event launch
interval, which can include GPU idle time during Rust launch preparation.

## FMA experiment

`fma_vs_mul_add` isolates the arithmetic. One recurrence step is a vector affine transform;
many steps and four independent chains provide an arithmetic-heavy control:

```sh
cargo run --release -p cutile-tutorials --example fma_vs_mul_add -- --iterations 1
cargo run --release -p cutile-tutorials --example fma_vs_mul_add -- --iterations 512 --chains 1
cargo run --release -p cutile-tutorials --example fma_vs_mul_add -- --iterations 512 --chains 4
```

Both examples use `examples/support/mod.rs` for shared argument parsing, graph timing, and
assembly inspection. `--tile`, `--warps`, and `--cache` select configurations.
`--inspect-dir DIR` saves Tile IR, cubin, SASS, embedded PTX text, and resource usage;
`print_ir` alone prints Tile IR rather than PTX.

See [the experiment results](experiments/polynomial/RESULTS.md) for the 102-run sweep,
repeated final comparison, NCU findings, historical coefficient-access alternatives, and
commands for the two retained examples. The exploratory variant example and Python sweep
scripts have been removed.
Generated data and profiler reports are in the ignored `artifacts/polynomial/` directory.
