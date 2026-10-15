# Minimal inference engine

A scheduler batches string requests; an independent model runner encodes inputs,
executes a GPU MLP, and decodes output token IDs into strings. The MLP computes
`ReLU(X * W0) * W1` with dimensions `16 -> 16 -> 1`, without biases.

Run from the repository root:

```bash
cd rust
cargo run --offline -p engine --example export_simple_mlp
cargo run --offline -p engine
cargo test --offline -p engine
```

The explicit export step initializes weights on GPU 0 with fixed seeds 42/43,
then writes `engine/models/simple_mlp/`:

```text
config.json
model.safetensors
generation_config.json
tokenizer.json
```

Generated model files are ignored by Git. Export once before running tests.
Inference loads these files and uploads saved weights to GPU 0. If the model
directory is absent, it prints a warning and initializes random weights without
saving them. An existing incomplete or corrupt directory still produces an error.
Pass another directory as the export command's first argument to change its
destination; callers can select an inference directory through `SimpleMLPEngine::load`.

`runner/loader.rs` accepts a single `model.safetensors` or a
`model.safetensors.index.json` mapping tensor names to shards. An existing index
is authoritative. Missing or malformed indexed shards cause errors. Model
configuration is deserialized into the concrete model's schema. The tokenizer
is loaded from `tokenizer.json`; `generation_config.json` is optional.

The layout follows Hugging Face file conventions. This custom MLP executes in
Rust; it does not provide a Transformers `AutoModel` implementation. Weights
are `layers.0.weight` (`[16, 16]`, input-by-hidden) and `layers.1.weight` (`[16]`).
Numeric input words `0..255` map to token IDs, which are divided by the configured
`input_divisor` (100) before execution. The result is spelled as numeric output
tokens for decoding. Generation settings are loaded and exposed; this MLP is
deterministic and does not implement LLM sampling.
