# W1: one FFN, three model formats

This experiment traces file bytes into runtime tensors and then into a prediction.
The same fixed `4 -> 8 -> 3` FFN is saved as Safetensors, GGUF, and ONNX.
All computation uses CPU and FP32. There is no training, tokenizer, GPU setup,
model download, or language-model serving stack.

```python
h = relu(x @ W1.T + b1)
logits = h @ W2.T + b2
prediction = logits.argmax(axis=-1)
```

Input has shape `[batch, 4]`; logits have shape `[batch, 3]`. The four tensors
contain 67 parameters, or 268 payload bytes. Weights come from a NumPy generator
with seed 42 and are written identically to all three files.

| File | Where weights live | Where the function lives | Executor |
| --- | --- | --- | --- |
| `model.safetensors` | Named tensor byte ranges | `TinyFFN.forward`, selected using `config.json` | PyTorch |
| `model.gguf` | Named tensor byte ranges plus metadata | Python calls that build a ggml graph | ggml CPU |
| `model.onnx` | Graph initializers | Serialized `Gemm -> Relu -> Gemm` graph | ONNX Runtime CPU |

The GGUF is a custom `tiny_ffn` model. Use this experiment's runner to execute it;
`llama-cli` does not know this architecture. The ggml path executes real native
operations, not a NumPy approximation. NumPy is used separately as a test oracle.

## Setup and quick start

Prerequisites: uv, existing Python 3.12.3, Git, CMake, and a C/C++ build toolchain.
The native library builds from unmodified ggml source; all experiment logic and
bindings are Python. Linux x86-64 is the validated platform.

The project declares `requires-python = ">=3.12"`, while `.python-version` selects
the already-installed 3.12.3 interpreter. Automatic Python downloads are disabled.
The explicit PyTorch CPU index avoids pulling CUDA packages. `uv.lock` fixes the
Python dependencies. The ggml source is fixed at
`e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` (version 0.23.0).

The host used for this experiment exports a Python 3.14 PyTorch library directory
in `LD_LIBRARY_PATH`. That makes even a Python 3.12 virtual environment load the
wrong `libtorch_python.so`, resulting in an undefined Python symbol error.
The commands below clear that variable for the child process only. They do not
modify your shell configuration. In an uncontaminated shell, plain `uv run ...`
works as well.

Run these commands from the repository root using its shared uv environment:

```bash
uv sync --locked
env -u LD_LIBRARY_PATH uv run python -m w1_model_formats.src.cmd.build_ggml
env -u LD_LIBRARY_PATH uv run python -m w1_model_formats.src.cmd.export
env -u LD_LIBRARY_PATH uv run python -m w1_model_formats.src.cmd.inspect
env -u LD_LIBRARY_PATH uv run python -m w1_model_formats.src.cmd.predict --backend torch --x 1 2 3 4
env -u LD_LIBRARY_PATH uv run python -m w1_model_formats.src.cmd.predict --backend ggml --x 1 2 3 4
env -u LD_LIBRARY_PATH uv run python -m w1_model_formats.src.cmd.predict --backend onnx --x 1 2 3 4
env -u LD_LIBRARY_PATH uv run python -m w1_model_formats.src.cmd.mmap_demo
env -u LD_LIBRARY_PATH uv run python -m w1_model_formats.src.cmd.compare
```

Each prediction should select class `0`, with logits approximately
`[0.346988, 0.111663, -0.154423]`.

Artifacts are written under `w1_model_formats/artifacts/`. Pinned ggml source and
build outputs live under `w1_model_formats/.deps/`. The default export command
regenerates its four fixtures. Use `--directory PATH` to select another artifact
directory; all model commands accept this option. Prediction accepts one input
vector on the command line, while the runner API supports a variable-size batch.

To use eagerly read GGUF storage instead of mapped storage:

```bash
env -u LD_LIBRARY_PATH uv run python -m w1_model_formats.src.cmd.predict --backend ggml --storage read --x 1 2 3 4
```

## Read the code in this order

1. [model/torch.py](src/model/torch.py): the PyTorch FFN and its `forward` function.
2. [common.py](src/common.py): tensor shapes, fixed weights, config contract,
   and the NumPy reference function.
3. [export.py](src/cmd/export.py): write the four tensors into both containers;
   separately construct and serialize the ONNX graph. `transB=1` on each `Gemm`
   expresses the transpose used by PyTorch's linear layers.
4. [torch.py](src/runner/torch.py): config selects a Python model class,
   `load_state_dict` fills its parameters, and `forward` defines the computation.
5. [storage.py](src/storage.py): parse offsets and create mapped views or
   independently owned bytes populated through `readinto()`.
6. [gguf_ggml.py](src/model/gguf_ggml.py): construct `f`, compute the graph,
   and copy out logits before releasing temporary storage.
7. [ggml.py](src/runner/ggml.py): attach tensor bytes and own mapped-weight
   resources for the graph model.
8. [onnx.py](src/runner/onnx.py): load the graph and call `session.run`.
   This path never imports PyTorch or reads `config.json`.
8. [mmap_demo.py](src/cmd/mmap_demo.py): check which addresses share storage and
   exactly where PyTorch makes a parameter copy.

[runner/_ggml.py](src/runner/_ggml.py) loads CFFI declarations generated from the
pinned public headers. Tensor, graph, context, and buffer pointers remain opaque
handles. ctypes is used only to preload the split libraries with global symbol
visibility; no C API signatures are maintained by hand. The library revision is
checked before creating tensors.

## What mmap demonstrates

Safetensors stores an 8-byte header length, a JSON header with shapes, dtypes,
and offsets, then tensor bytes. Its offsets are relative to the data section.
The teaching loader computes absolute offsets and creates NumPy views over a
Python `mmap`. It supports only this experiment's dense FP32 fixtures; the normal
PyTorch runner uses the official Safetensors loader.

GGUF stores metadata and tensor descriptors before its aligned data section.
The official `GGUFReader` uses NumPy `memmap`. GGUF dimensions list the
fastest-varying dimension first: a NumPy weight of shape `[8, 4]` has GGUF
dimensions `[4, 8]`. This changes dimension metadata, not the order of its bytes.
In read mode, that temporary reader mapping is released after the descriptors
are parsed; the ggml tensor bytes come only from the eagerly read buffer.

The demo reports byte offsets, shapes, dtypes, addresses, and memory sharing:

| Path | Loader array shares mapped file storage | Runtime parameter shares loader storage |
| --- | --- | --- |
| Safetensors teaching loader, mmap | Yes | No: default `load_state_dict` copies |
| Safetensors teaching loader, read | No | No: default `load_state_dict` copies |
| GGUF + ggml, mmap | Yes | Yes: matching data pointers |
| GGUF + ggml, read | No | Yes: matching data pointers into owned read bytes |

The Safetensors teaching map uses `ACCESS_COPY`: clean pages remain file-backed,
and writes would produce private copy-on-write pages without changing the file.
This also supplies writable views to `torch.from_numpy`, whose tensors do not
enforce NumPy read-only flags. Inference does not modify these mapped weights.
The GGUF map is read-only.

Read mode uses ordinary `readinto()` to fill independently owned, aligned bytes.
It makes the meaningful storage comparison: operating-system file mapping versus
an eager user-space file read. This tiny fixture is still not a performance benchmark.

The ggml runner wraps the mapped bytes in a non-owning CPU backend buffer using
`ggml_backend_cpu_buffer_from_ptr`. `ggml_backend_tensor_alloc` attaches each
tensor at its existing address. Activations use a separate, per-call context.
The runner releases native tensor handles before closing the mapping. Treat
`WeightFile.weights` arrays as borrowed views: do not use them after the owner
closes. Returned logits are independent copies and remain valid after close.

On the validated machine, pages are 4096 bytes and each model file is smaller
than one page. This demonstrates address mappings, not layer-by-layer disk I/O
or a measurable performance gain. Creating a mapping does not load all pages
immediately, but parsing headers can already fault pages in. For large dense
models, repeated inference usually touches nearly all weights. CPU file mapping
also does not eliminate the device transfer required by ordinary GPU inference.

## Verification and offline reuse

```bash
env -u LD_LIBRARY_PATH uv run --offline python -m w1_model_formats.src.cmd.compare
env -u LD_LIBRARY_PATH uv run --offline python -m w1_model_formats.src.cmd.mmap_demo
env -u LD_LIBRARY_PATH uv run --offline pytest -q w1_model_formats/tests
env -u LD_LIBRARY_PATH uv run --offline python -m w1_model_formats.src.cmd.build_ggml --offline
```

The test suite checks exact weight round trips, metadata/layout handling,
single and batched predictions, repeated inference, runtime lifetimes, mmap
sharing, explicit copy behavior, unchanged source hashes, invalid configuration
and input, and missing native libraries. A fresh subprocess verifies that ONNX
prediction works with only `model.onnx` and never imports the Python model class.
Logits must satisfy `rtol=1e-5, atol=1e-6`; predictions must match.

Python downloads remain disabled even during initial setup. Dependency wheels
and pinned ggml source are downloaded once when absent, then reused. Missing
native libraries produce an actionable error; there is no backend fallback.

## Primary references

- [Safetensors file format](https://github.com/safetensors/safetensors#format)
- [Official GGUF Python reader](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/gguf_reader.py)
- [Pinned ggml public API](https://github.com/ggml-org/ggml/blob/e91ded11bdcd78c42f9c8d3978ff6686eb4c1226/include/ggml.h)
- [Pinned ggml CPU buffer attachment](https://github.com/ggml-org/ggml/blob/e91ded11bdcd78c42f9c8d3978ff6686eb4c1226/src/ggml-backend.cpp)
- [ONNX IR](https://onnx.ai/onnx/repo-docs/IR.html)
- [Python mmap](https://docs.python.org/3.12/library/mmap.html)
