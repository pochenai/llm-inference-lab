# From Model Files to Predictions

Run the same tiny feed-forward network using Safetensors + PyTorch, GGUF + ggml,
and ONNX + ONNX Runtime. The experiment separates two questions:
**where are the weights stored, and where does the function come from?**

Everything uses CPU and FP32. Inputs are numbers, so there is no tokenizer.
Weights are deterministic random values, not trained parameters; the predicted
class demonstrates execution, not a useful classification task.

## The model

The network has dimensions `4 -> 8 -> 3`:

```python
h = relu(x @ W1.T + b1)
logits = h @ W2.T + b2
prediction = logits.argmax(axis=-1)
```

`x` has shape `[batch, 4]`, and logits have shape `[batch, 3]`.
The four parameters are `fc1.weight [8, 4]`, `fc1.bias [8]`,
`fc2.weight [3, 8]`, and `fc2.bias [3]`: 67 FP32 values, or 268 payload bytes.
File headers, metadata, padding, and graph data add to that size.

## Code structure

```text
src/
  model/
    torch.py           # TinyFFN: handwritten f for Safetensors + PyTorch
    gguf_ggml.py       # Handwritten ggml graph f for GGUF + ggml
  common.py            # Config, shapes, seeded weights, input validation, test reference
  storage.py           # WeightFile: mapped tensor views and eagerly read bytes
  runner/
    __init__.py        # load_runner(): lazily select one backend
    torch.py           # Safetensors/config loading and PyTorch model lifetime
    ggml.py            # GGUF weight attachment and ggml resource lifetime
    onnx.py            # ONNX session lifetime; graph is already in the file
    _ggml.py           # Generated-CFFI loader and native-library paths
    generate_cffi.py   # Generate declarations from pinned public headers
    ggml_api.h          # The public ggml headers exposed to the generator
  cmd/
    export.py          # Generate shared weights and write all three formats
    predict.py         # Parse input, select runner, print logits and class
    inspect.py         # Print tensor metadata and serialized ONNX operators
    mmap_demo.py       # Check memory addresses, sharing, copies, and file hashes
    compare.py         # Compare backend outputs against the NumPy reference
    build_ggml.py      # Fetch/build ggml and generate its CFFI binding
tests/                 # Serialization, inference, storage, and error checks
artifacts/             # Generated model files; ignored by Git
.deps/ggml/            # Native source and build results; ignored by Git
```

Read these files in order:

1. [model/torch.py](src/model/torch.py): understand the PyTorch function before
   looking at storage.
2. [cmd/export.py](src/cmd/export.py): see the same weights saved three ways.
3. [cmd/predict.py](src/cmd/predict.py) and [runner/torch.py](src/runner/torch.py):
   follow one complete file-to-prediction path.
4. [model/gguf_ggml.py](src/model/gguf_ggml.py),
   [runner/ggml.py](src/runner/ggml.py), and [runner/onnx.py](src/runner/onnx.py):
   compare where each backend gets its function and who owns runtime resources.
5. [storage.py](src/storage.py) and [cmd/mmap_demo.py](src/cmd/mmap_demo.py):
   inspect memory ownership after the execution paths make sense.

Skip `_ggml.py` and `generate_cffi.py` on the first pass: they load declarations
generated from public C headers, not the FFN.
`common.py` does not import PyTorch; its NumPy reference is only for verification,
never an execution fallback. Runners do not import command modules.

## Two independent stages

Export once, then load the saved files in a new process for each prediction:

```text
cmd/export.py -> common.make_weights()
              -> artifacts/{config.json, model.safetensors, model.gguf, model.onnx}

cmd/predict.py -> runner.load_runner(backend)
              -> selected runner loads its model file(s)
              -> runner.predict(x) -> logits -> argmax -> predicted_class
```

Predicting does not rerun export or regenerate weights. All runners expose
`predict(x)` and support `with ... as runner:` for resource cleanup.

| Backend | Weight loading | Where the function lives |
| --- | --- | --- |
| PyTorch | Official Safetensors loader; `load_state_dict()` copies into parameters | `model/torch.py`: config selects `TinyFFN`; `forward()` defines the operators |
| ggml | Official GGUF reader; `WeightFile` provides arrays whose addresses are attached to ggml tensors | `model/gguf_ggml.py`: builds matrix multiply, add, ReLU, matrix multiply, add |
| ONNX Runtime | Session loads ONNX initializers | The ONNX file contains `Gemm -> Relu -> Gemm`, written by `cmd/export.py` |

Safetensors and this GGUF file do not contain an executable operator graph.
Config and GGUF metadata describe the supported architecture, but Python code
must implement it. Unknown architectures, unsupported settings, and invalid
tensor shapes are rejected rather than dynamically importing arbitrary code.

ONNX prediction needs neither `config.json` nor the PyTorch model class.
Its graph uses `transB=1` for both Gemm nodes to match `x @ W.T`.
For GGUF/ggml, NumPy shape `[out, in]` becomes dimension metadata `[in, out]`;
this reverses the dimension description, not the stored weight bytes.

## Run the experiment

Run commands from the repository root, the directory containing `pyproject.toml`
and `uv.lock`, not from this README's directory. The package name in commands
is `w1_model_formats.src`.

Requirements: uv, an existing Python 3.12.3 interpreter, Git, CMake, and a C/C++
compiler for ggml. Type checking also needs Node.js available on `PATH` so
Pyright can run without bootstrapping a runtime from the network.
The project supports Python `>=3.12`; `.python-version`
selects `3.12.3`, and uv's automatic Python downloads are disabled.
Dependencies use CPU-only PyTorch. Initial dependency/source downloads require
network access unless already cached.

```bash
uv sync --locked
env -u LD_LIBRARY_PATH uv run --offline python --version
env -u LD_LIBRARY_PATH uv run --offline python -m w1_model_formats.src.cmd.build_ggml
env -u LD_LIBRARY_PATH uv run --offline python -m w1_model_formats.src.cmd.export
env -u LD_LIBRARY_PATH uv run --offline python -m w1_model_formats.src.cmd.inspect
env -u LD_LIBRARY_PATH uv run --offline python -m w1_model_formats.src.cmd.predict --backend torch --x -1 0 0 0
env -u LD_LIBRARY_PATH uv run --offline python -m w1_model_formats.src.cmd.predict --backend ggml --x -1 0 0 0
env -u LD_LIBRARY_PATH uv run --offline python -m w1_model_formats.src.cmd.predict --backend onnx --x -1 0 0 0
```

The `env` prefix prevents an inherited `LD_LIBRARY_PATH` from mixing another
Python environment's PyTorch libraries into this one. It does not change your
shell configuration. It can be omitted on hosts without that conflict.

The native build is pinned to ggml commit
`e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` and reuses its source/build directory.
uv's `--offline` only controls uv: the build command may still fetch ggml source.
After the first successful setup, use the script's own `--offline` too:

```bash
uv sync --locked --offline
env -u LD_LIBRARY_PATH uv run --offline python -m w1_model_formats.src.cmd.build_ggml --offline
env -u LD_LIBRARY_PATH uv run --offline python -m w1_model_formats.src.cmd.compare
env -u LD_LIBRARY_PATH uv run --offline python -m w1_model_formats.src.cmd.mmap_demo
env -u LD_LIBRARY_PATH uv run --offline pyright
env -u LD_LIBRARY_PATH uv run --offline pytest -q w1_model_formats/tests
```

For `[-1, 0, 0, 0]`, expect logits approximately
`[0.208841, 0.247016, 0.057876]` and `predicted_class: [1]` (zero-based class index).
The nonzero class makes an accidental hard-coded `0` easier to spot when comparing backends.
The CLI accepts one sample; the runner API also accepts batches:

```python
from w1_model_formats.src.runner import load_runner

with load_runner("onnx") as runner:
    logits = runner.predict([[1, 2, 3, 4], [-2, 1, -3, 0.5]])
    print(logits.argmax(axis=-1))
```

Export, predict, inspect, compare, and mmap_demo accept `--directory PATH` to use
a different artifact directory. Prediction accepts `--storage mmap|read` only
for `--backend ggml`; use mmap_demo for the Safetensors storage comparison.

## What the mmap experiment shows

`storage.py` separates a tensor's metadata from its backing bytes:

- GGUF uses the official reader's `numpy.memmap` and absolute tensor offsets.
- The educational Safetensors loader reads the length-prefixed JSON header from
  either Python `mmap` or an eagerly read byte buffer, then creates NumPy views
  at the recorded data offsets.
  It supports this experiment's FP32 files, not every possible Safetensors file.
- `mmap` mode keeps file-backed views. `read` mode uses ordinary `readinto()`
  to fill an independent, aligned byte buffer.

The GGUF reader temporarily maps the file to parse its descriptors even in
`read` mode; it is released after parsing. The arrays passed to ggml in that
mode point only into the `readinto()` buffer, which is the storage being compared.

GGUF uses a read-only mapping. The educational Safetensors path uses a private
copy-on-write mapping so `torch.from_numpy()` can safely wrap the views without
writing to the source file. The experiment itself does not modify weights.

Mapping can avoid eager user-space copies and lets the OS populate pages on
demand. But mapping a file does not guarantee zero-copy model parameters:
`torch.from_numpy()` shares the view, while default `load_state_dict()` copies
values into the model's existing parameters. The normal TorchRunner uses the
official loader; the educational loader is a separate inspection/demo path.

The ggml runner directly references the mapped or eagerly read weights.
It keeps their owner alive until its native tensors are released and copies
outputs before freeing per-call activation memory. Borrowed weight views must
not be used after closing their owner.

## ggml runtime constants

`runner/ggml.py` owns file-backed weight setup; `model/gguf_ggml.py` owns graph
construction. Together they give every runtime number a name and a comment:

| Constant | Meaning | Related to TinyFFN dimensions? |
| --- | --- | --- |
| `CPU_BACKING_ALIGNMENT_BYTES = 64` | Host-pointer alignment required by the pinned ggml CPU buffer API | No; backend ABI requirement |
| `WEIGHT_CONTEXT_BYTES = 64 KiB` | Capacity for ggml tensor descriptors when `no_alloc=True` | No; weights remain in mapped/read storage |
| `GRAPH_NODE_CAPACITY = 64` | Maximum graph-node count; this FFN has five operators | No; conservative graph bookkeeping capacity |
| `CPU_THREAD_COUNT = 1` | Fixed CPU execution policy for reproducible teaching output | No; runtime policy |
| `ACTIVATION_CONTEXT_BASE_BYTES = 1 MiB` | Conservative context/scratch headroom | No; runtime allocation policy |
| `activation_context_bytes(config, batch)` | Adds bytes for input, affine intermediates, ReLU, and logits | Yes; derives from dimensions and batch size |

The per-row formula is `(input_dim + 3 * hidden_dim + 2 * output_dim) * 4`.
It counts the graph's FP32 input and intermediate tensor allocations; the 1 MiB
base remains deliberately separate because ggml graph metadata and scratch are
implementation details rather than serialized model parameters.

The demo checks tensor offsets, dtypes, shapes, addresses, shared storage, and
unchanged source hashes. These tiny files may each fit within one OS page:
they cannot demonstrate mmap speedups or independent page faults per layer.

The per-tensor printout is an observation, not a correctness proof. Seeing a
valid `shape` only proves that the loader could create a view using header
metadata; a file can have a plausible shape but wrong values. The demo proves
weight contents with `np.testing.assert_array_equal(value, official[name])`,
then proves execution with `np.testing.assert_allclose(actual, expected, ...)`.
For the PyTorch path, distinct `view` and `parameter` addresses show that
default `load_state_dict()` copied values. For GGUF + ggml, matching `view` and
`ggml` addresses show that ggml directly uses the mapped or eagerly read bytes.

## Validation

Source and test functions have annotated parameters and return values. Shared
types in `common.py` describe FP32 arrays, byte arrays, weight dictionaries,
backends, and storage modes. `runner.Runner` is a structural protocol for the
common execution interface. Array shapes remain runtime-validated: NumPy's dtype
annotations alone cannot prove `[batch, 4]` or `[batch, 3]` shapes.

Pyright checks the source and tests; a test guards against missing function
annotations. The CFFI declarations are generated from the pinned ggml public
headers. ctypes is used only to preload the split `ggml-base` library with global
symbol visibility before CFFI loads `ggml-cpu`; no C API signatures are handwritten.
Casts at third-party boundaries document information that their types cannot
express; a cast itself does not validate runtime data.

Tests reload all four parameters from each format and compare single, batched,
zero, and mixed-sign inputs with `rtol=1e-5`, `atol=1e-6`. They also check
predicted classes, mmap/copy equivalence, repeated inference, resource cleanup,
invalid inputs/configs/weights, and standalone subprocess loading. Missing ggml
libraries produce an explicit build instruction, never a substitute backend.
