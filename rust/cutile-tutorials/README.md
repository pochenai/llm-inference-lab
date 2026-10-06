# cuTile Tutorials

Small, runnable examples for learning cuTile Rust. Each example is a standalone Cargo example
under `examples/`.

## Run

From this directory:

```sh
cargo run --example hello_world
```

The examples use the local checkout at `/home/po/ai/cutile-rs`. Update the `cutile` path in
`Cargo.toml` if that checkout moves. A compatible CUDA Toolkit, `tileiras`, NVIDIA driver, and
CUDA GPU are required to compile and run kernels.
