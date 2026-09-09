# GGUF, Safetensors, and ONNX: How LLM Model Files Are Loaded and Executed

# 1. Motivation

When browsing open-source models on Hugging Face, one often encounters `model.safetensors`, `model.gguf`, and `model.onnx`. They all look like model files, yet they serve different goals: training or fine-tuning, cross-framework interchange, local CPU inference, high-throughput GPU serving, and deployment in browsers or on mobile devices.

File extensions alone do not explain those differences. The more fundamental question is: given an input $x$ and a model file on disk, what other information does a system need before it can compute a prediction?

This article uses a two-layer feed-forward network (FFN) as a running example. It follows the path from file bytes to in-memory tensors, model semantics, hardware kernels (operator implementations), and finally a prediction. Along the way, it explains the boundaries of three common inference file formats and how they connect to PyTorch, llama.cpp, vLLM, SGLang, and ONNX Runtime.

# 2. A first-principles derivation of the information required for inference

A model can be abstracted as a parameterized function:

$$
y = f_{\theta}(x).
$$

For a classification FFN:

$$
\begin{aligned}
h &= \operatorname{ReLU}(x W_1^{\mathsf{T}} + b_1), \\
\operatorname{logits} &= h W_2^{\mathsf{T}} + b_2, \\
\operatorname{predicted\_class} &= \operatorname*{arg\,max}(\operatorname{logits}).
\end{aligned}
$$

$W_1$, $b_1$, $W_2$, and $b_2$ are persistent parameters $\theta$; $x$, $h$, and $\operatorname{logits}$ are temporary activations for one request. For an LLM, logits usually pass through a sampling policy such as temperature or top-$k$/top-$p$ before producing the next token. Those policies normally belong to serving/runtime configuration rather than the weight file.

For a text model, $x$ is not the original string. It is the token IDs produced by a tokenizer. The tokenizer splits text and maps it to integer IDs in a vocabulary; it also decodes output token IDs back to text. The end-to-end process is:

$$
\text{text prompt}
\xrightarrow{\operatorname{tokenize}}
\text{token IDs}
\xrightarrow{f_{\theta}}
\text{logits}
\xrightarrow{\operatorname{sample}}
\text{next-token ID}
\xrightarrow{\operatorname{decode}}
\text{text}.
$$

This article sets aside tokenizer internals such as BPE and SentencePiece while discussing file formats. In an actual LLM deployment, however, the vocabulary, special tokens, chat template, and tokenizer configuration must accompany the model. Otherwise, text input and output can have the wrong semantics even when the weights and $f_{\theta}$ are correct.

## 2.1 An inference package consists of numerical, semantic, and execution information

The FFN above provides both parameters and a computation. It follows that a runnable model needs at least three layers of information:

| Layer | Content | FFN example |
| --- | --- | --- |
| Numerical layer | Parameter tensors $\theta$ | $W_1$, $b_1$, $W_2$, $b_2$ |
| Semantic/structural layer | How parameters compose $f$ | Two MatMuls, an intervening ReLU, and tensor connections |
| Execution layer | How to run operators on the target hardware | CPU BLAS, CUDA kernels, Metal kernels, and so on |

The numerical layer is organized around tensors. Tensor elements may use `fp16`, `bf16`, `int8`, or quantized bit-packed encodings; block quantization usually also needs associated information such as scales. To reconstruct bytes from a file into a tensor that can participate in computation, a loader needs the raw values as well as dtype, shape, and layout conventions.

A text LLM also needs its tokenizer, vocabulary, special tokens, chat template, context length, RoPE configuration, and other inference settings. A decoder-only LLM additionally maintains a KV cache at runtime; it is dynamic state, not fixed weights published with the model.

Thus, a usable inference package can be summarized as:

```text
Inference package = weights + model semantics + tokenizer + inference metadata
```

This identifies what a runnable model needs, but it does not prescribe how that information must be placed in files. Different deployment goals choose different boundaries across the three layers:

- To distribute weights safely and generally while letting an existing training or inference framework determine the function structure, the file only needs to focus on tensors.
- To support local inference with low memory use, the file should also carry enough model metadata and enable efficient quantization and offset-based weight access.
- To carry function semantics across frameworks and devices, the computation graph itself must become file data.

Safetensors, GGUF, and ONNX correspond to these three choices. They set different packaging boundaries for weights, function semantics, and runtime behavior.

> Note: resuming training also requires optimizer state, possibly retained gradients, scheduler/trainer state, random-number state, and data progress. A training checkpoint answers “how can learning continue without interruption?”, whereas an inference deployment package answers “how can the model respond?”. This article does not cover training-related storage.

## 2.2 Three inference file formats set different packaging boundaries

The previous section established the three layers of information needed to run a model. The next question is which layers the model file stores and which the runtime supplies. Safetensors, GGUF, and ONNX differ at exactly this packaging boundary.

**Safetensors packages parameter tensors.** For the FFN above, it can store the values of $W_1$, $b_1$, $W_2$, and $b_2$, plus the dtype, shape, and other information needed to read them. It does not describe the relation “apply ReLU between two linear transformations.” The loader still needs an FFN implementation and fills its parameters by name. A common Hugging Face model package obtains this behavior from model code outside the file and its accompanying `config.json`; Safetensors itself does not provide a general `forward()`. [Safetensors format specification](https://github.com/safetensors/safetensors#format)

**GGUF packages tensors together with model information for a target runtime.** Examples include an architecture identifier, hyperparameters, a tokenizer, and `ggml_type`, which interprets the weight layout. This often reduces inference-time dependence on external configuration files. The metadata is not a computation graph, however: `general.architecture = "llama"`, for example, tells llama.cpp to select its already-implemented Llama architecture; it does not read an arbitrary function from the file. A runtime still needs a corresponding computation implementation when GGUF packages the FFN above. [GGUF format specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)

**ONNX packages computation relations as a graph.** For the same FFN, weights can be initializers in the graph, while linear transformations, ReLU, and their connections are described by operator nodes. The loader does not need the original `TinyFFN.forward()`, but it must be able to execute the graph's operators. Operator semantics in the graph are distinct from CPU/GPU kernel implementations. [ONNX IR specification](https://onnx.ai/onnx/repo-docs/IR.html)

Mapped onto the three layers above, the formats have the following packaging scopes:

| Format | Numerical layer: parameters $\theta$ | Semantic/structural layer: how parameters compose $f$ | Execution layer: how operators run |
| --- | --- | --- | --- |
| Safetensors | The file stores tensor descriptions and values | Model code and accompanying configuration outside the file | CPU/GPU and other runtime backends |
| GGUF | The file stores tensors, types, and related quantization information | In-file metadata selects and configures an architecture already implemented by the runtime; it does not store a general computation graph | Backends of compatible runtimes such as ggml |
| ONNX | Initializers in the graph, embedded or stored in external tensor data | An in-file computation graph and operator semantics | Backends of compatible runtimes such as ONNX Runtime |

Safetensors and GGUF both require a model implementation to organize parameters; GGUF additionally specifies more about the model and weight representation. ONNX instead gives the computation graph to the runtime for parsing. All three still need appropriate operator implementations to execute. Section 3 expands on the concrete fields and loading paths.

> GGUF and Safetensors solve “how to store and retrieve $\theta$ efficiently and reliably”; a runtime's model implementation or an ONNX graph solves “how $\theta$ composes $f$”; CPU/GPU backend kernels solve “how $f$ actually runs.”

> Note: file count is not part of this packaging boundary. Both Safetensors and GGUF can be distributed as a single file or as shards. Configuration and tokenizer files in a typical Hugging Face model directory are also separate from the Safetensors weight file itself.

## 2.3 The boundary between PyTorch checkpoints and deployment model packages

The previous section compared Safetensors, GGUF, and ONNX by the information layers that each file packages. PyTorch `.pt`/`.pth` files also appear to save models, but they are not another fixed schema: they are conventional extensions for `torch.save()`, and their contents are determined by the Python object supplied by the caller. [PyTorch guide to saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

The lightest case stores a `state_dict`, a mapping from parameter names to tensors, including registered buffers:

$$
\texttt{state\_dict} = \{\texttt{name} \mapsto \texttt{tensor}, \ldots\}.
$$

For the FFN above, a `state_dict` may contain `fc1.weight`, `fc1.bias`, `fc2.weight`, and `fc2.bias`. Loading still constructs `TinyFFN` in PyTorch code first and then fills parameters with matching names. In other words, the file provides $\theta$, while `TinyFFN.forward()` provides $f$. Its boundary at the layer “how parameters compose a function” is therefore close to Safetensors, rather than to ONNX, which contains a computation graph.

A checkpoint intended to resume training commonly stores those weights with additional state:

$$
\begin{aligned}
\text{training checkpoint} ={}& \text{model state} + \text{optimizer state} \\
&+ \text{scheduler / step / data progress} \\
&+ \text{optional RNG, AMP, or other trainer state}.
\end{aligned}
$$

These fields continue training from an interruption and are unnecessary for deployment that only computes $\operatorname{predict}(x) = f_\theta(x)$. Optimizer state also often makes a general checkpoint larger than pure model state. Because actual fields are chosen by training code, a `.pt`/`.pth` suffix alone does not establish whether a file contains an optimizer, gradients, or other training state.

PyTorch can also save a complete `nn.Module`. `torch.save(model, path)` serializes the module object through Python `pickle`, and restoration still depends on the original class definition and module path. PyTorch recommends saving and loading a `state_dict`; it also warns against loading untrusted files in a way that permits arbitrary object deserialization. `weights_only=True` is a more restricted path for loading weight state. [PyTorch serialization notes](https://docs.pytorch.org/docs/stable/notes/serialization.html) [PyTorch `torch.load` security notes](https://docs.pytorch.org/docs/stable/generated/torch.load.html)

For this reason, the article does not treat a PyTorch checkpoint as a fourth deployment format. A pure `state_dict` restores parameters well inside PyTorch; a training checkpoint resumes training; and a complete `nn.Module` suits a tightly coupled Python environment. Their packaging boundary follows PyTorch serialization and the caller's object choice. Safetensors, GGUF, and ONNX instead represent the three formats compared here: a safe tensor container, a model package for local quantized runtimes, and an interchangeable computation graph.

# 3. Schema, loading, and ecosystem support for the three file formats

The previous section established the packaging boundary of each format. This section examines their schemas, loading paths, and ecosystem support in more detail.

## 3.1 Safetensors: a safe, direct tensor container

Safetensors has a simple file layout:

```text
[8-byte little-endian header length][JSON header][contiguous raw tensor bytes]
```

The header stores a name, dtype, shape, and `data_offsets` for each tensor:

```json
{
  "fc1.weight": {"dtype": "F16", "shape": [8, 4], "data_offsets": [0, 64]},
  "fc1.bias":   {"dtype": "F16", "shape": [8], "data_offsets": [64, 80]}
}
```

A loader can read the small header and locate the contiguous byte range for `fc1.weight` in the raw data area. The format does not execute arbitrary Python code, avoiding the deserialization risk of traditional pickle checkpoints; see the [Safetensors format specification](https://github.com/safetensors/safetensors#format).

It is commonly the default weight format in the Hugging Face `transformers` ecosystem. Model semantics still come from `config.json` and Transformers model code, while one or more `.safetensors` shards hold the weights.

A common model directory looks like this:

```text
model/
├── config.json                 # Model architecture and hyperparameters
├── model.safetensors           # Named parameter tensors
├── tokenizer.json              # Or tokenizer.model, depending on the model
└── tokenizer_config.json
```

For the FFN above, a loader reads the parameters from dtype, shape, and data offsets in the header, then gives them to a model with an already-defined computation. Its responsibilities can be summarized as:

```python
model = TinyFFN()                # Defines the computation
weights = load_safetensors(...)  # Reads named tensors
model.load_state_dict(weights)   # Fills the parameters
logits = model(x)
```

The `fc1.weight` example contains $8 \times 4$ F16 elements, so its data area uses $8 \times 4 \times 2 = 64$ bytes. It stores contiguous ordinary floating-point values; a loader can construct a floating-point tensor from its dtype and shape without first decoding a specialized quantized representation. This describes the numerical representation of the parameter. Whether loading copies it, moves it to a device, or registers it as an `nn.Parameter` is determined by loader code.

In serving:

- **vLLM** prefers Safetensors by default and falls back to PyTorch format when it is absent. Its loader is in [vllm/model_executor/model_loader](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/model_loader).
- **SGLang** also primarily accepts Hugging Face weight directories and automatically loads Safetensors or PyTorch checkpoints; see [SGLang model loading](https://docs.sglang.io/advanced_features/model_loading.html).
- Both select model classes using `architectures` in HF configuration rather than deriving an arbitrary computation graph from Safetensors. vLLM documents architecture support in [supported models](https://github.com/vllm-project/vllm/blob/main/docs/models/supported_models.md), while SGLang implementations are in [sglang/srt/models](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models).

## 3.2 GGUF: local inference and quantized distribution

The broad GGUF layout is:

```text
magic "GGUF" + version
number of tensors + number of KV pairs
KV metadata
tensor info table: name + dimensions + ggml type + relative offset
aligned tensor-data blob
```

KV metadata can carry `general.architecture`, layer count, hidden size, RoPE configuration, tokenizer/vocabulary information, and quantization information. Tensor information contains both shape and `ggml_type`, so it can describe ggml quantization layouts such as Q4 and Q8. The complete definition is in the [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) and [llama.cpp GGUF header](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/gguf.h).

The same FP16 weight tensor uses a binary descriptor in GGUF. The following omits concrete dimension values and only shows the fields; tensor names and dimension order must follow the GGML conventions of the target architecture:

```text
tensor info:
  name       = "blk.0.ffn_up.weight"
  n_dims     = 2
  dimensions = [...]
  ggml_type  = GGML_TYPE_F16
  offset     = ...  # Relative to the tensor-data blob
```

With `GGML_TYPE_F16`, the data region still contains ordinary FP16 values. GGUF and Safetensors differ in their descriptors, naming, and layout conventions; both can store quantized tensor data, while GGUF's `ggml_type` further defines concrete encodings such as `Q4_K`.

When the type changes to a quantized type such as `Q4_K`, the data region uses its corresponding block encoding: the low-bit codes for a group of weights are packed with shared scale/minimum and other metadata. For example, `block_q4_K` contains packed four-bit codes, sub-block scale/minimum information, and a super-block scale. The loader must interpret bytes according to this type; kernels then decode or process them in a fused manner. Such data cannot be used directly as an ordinary floating-point parameter array. Q4 and IQ encodings have different layouts; see the [ggml quantization block definitions](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h).

The core llama.cpp path is:

```text
GGUF metadata → architecture enum → C++ model implementation
GGUF named tensors → load_tensors()
C++ build_arch_graph() → ggml graph → backend kernels → logits
```

GGUF does not store a general computation graph. To support a new architecture, llama.cpp must therefore implement parameter loading and a graph builder in source code. The development guide is [HOWTO-add-model](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md); the relevant implementation is concentrated in [src/llama-model.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp) and [src/llama-model-loader.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model-loader.cpp).

GGUF fits local inference environments such as llama.cpp, Ollama, and LM Studio. It is often convenient to distribute as one file, though it can also be sharded; its CPU inference path and quantization choices are mature.

## 3.3 ONNX: an interchangeable representation of weights and computation graphs

The top-level ONNX container is `ModelProto`, whose `GraphProto` stores the graph. Each `NodeProto` is an operator, and weights are commonly graph initializer tensors. See the [ONNX IR specification](https://onnx.ai/onnx/repo-docs/IR.html).

```text
ModelProto
├── GraphProto
│   ├── inputs / outputs
│   ├── initializer tensors
│   └── NodeProto DAG
└── metadata / opset imports
```

This lets ONNX Runtime parse nodes such as `MatMul`, `Add`, and `Relu` and execute the model without the Python source of `TinyFFN.forward()`. Session initialization also performs graph optimization; see [ONNX Runtime graph optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html). The runtime then partitions the graph across Execution Providers such as CPU, CUDA, and TensorRT; see [ONNX Runtime high-level design](https://onnxruntime.ai/docs/reference/high-level-design.html).

ONNX's interchangeable computation graph and support for several Execution Providers make it suitable for cross-framework, cross-device, browser/mobile, and edge deployment. That does not mean every LLM immediately achieves the same portability. Before deployment, four conditions usually need checking:

- **Export completeness**: Can the model's `forward()`, control flow, and operators all become nodes in the target ONNX graph, rather than failing at an unsupported operation or needing a custom implementation? [PyTorch ONNX exporter](https://docs.pytorch.org/docs/stable/onnx_export.html)
- **Dynamic shapes**: Can dimensions such as batch size and sequence length vary at runtime, instead of accepting only the fixed shape of the export example input? [PyTorch dynamic-shape documentation](https://docs.pytorch.org/docs/stable/onnx_export.html)
- **Operator support**: Can the target Execution Provider execute graph nodes? Unsupported nodes may be assigned to another provider, causing extra copies or lower performance. [ONNX Runtime Execution Provider mechanism](https://onnxruntime.ai/docs/execution-providers/)
- **Quantization path**: Do the quantized ONNX graph representation and operators match the target hardware, and do they provide the expected benefit within an acceptable accuracy loss? [ONNX Runtime quantization documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

## 3.4 Deployment trade-offs across size, loading, and ecosystem support

The preceding sections explained what each format stores. This section turns that into a deployment choice: first determine whether you need an interchangeable computation graph, low-memory quantization, or a weight directory for an existing training framework; then compare size and loading cost, which depend jointly on weight precision, model architecture, runtime, and target hardware. A format name does not itself guarantee a fixed file size or loading speed.

| Comparison axis | Safetensors | GGUF | ONNX |
| --- | --- | --- | --- |
| Primary deployment goal | Safely and directly distribute named tensors under existing model code | Distribute quantized weights with model metadata to a compatible runtime | Exchange and deploy a model together with its computation graph |
| Common weight representations and relative size | Commonly BF16/FP16; can also store quantized tensor data. Floating-point weights are close to raw parameter size | Commonly Q2–Q8/F16; lower-bit quantization usually makes files smaller than floating-point weights | Depends on export precision, whether initializers are external, and the quantization scheme; files may also contain the graph and extra constants |
| Main work at first load | Read header to locate tensors, then instantiate parameters in external model code and move them to a device; supported loaders can use mmap (memory mapping) | Read metadata and tensor descriptors, then create weight views or decode according to `ggml_type`; supported runtimes can also use mmap (memory mapping) | Parse, validate, and optimize the graph, then partition it by Execution Provider; cost depends on the graph and target backend |
| Common integration path | Hugging Face `transformers` weight directories and serving runtimes such as vLLM and SGLang | Local inference runtimes such as llama.cpp, Ollama, and LM Studio | ONNX Runtime and runtimes for the web, mobile, and edge devices |
| Source of function $f$ | Python/C++ model implementation outside the file | Target architecture already implemented in the runtime | Operator DAG in the file |
| Fit with standard floating-point training paths | Common floating-point parameters can be passed directly to training frameworks | Requires adaptation to GGML representation; quantized weights also involve decoding or specialized training support | Graph interchange does not restore complete training state |

The “common integration path” row describes how a community commonly loads these artifacts; it does not mean that a runtime supports only one file format. Likewise, distributing one file or multiple shards is a release strategy, not the format boundary between Safetensors and GGUF.

Three direct choices follow. Safetensors is a natural weight container when an existing Hugging Face/PyTorch model implementation will be trained, fine-tuned, or GPU-served. For low-memory local CPU/GPU inference, GGUF's quantization layouts and metadata fit llama.cpp-like runtimes more closely. When computation relations must be handed to different devices and runtimes, ONNX graph representation is more suitable. Actual performance still depends on quantization bit width, graph optimizations, fused operators, the Execution Provider, and target hardware—not on the format name alone.

> Note: “training-friendly” describes the cost of connecting a file's parameter representation to a training system; it does not mean the format itself performs training. A common Safetensors checkpoint stores ordinary FP16/BF16 tensors aligned with PyTorch's floating-point parameter-update path; autograd and the optimizer still come from the framework. GGUF F16 weights can also be used after adaptation, but GGML naming, dimensions, and type conventions must be handled. Quantized GGUF usually additionally requires decoding into floating-point parameters or specialized quantization-aware training. Safetensors can also hold quantized data, so a file extension alone cannot determine training convenience. A common workflow retains a floating-point base for fine-tuning or LoRA (Low-Rank Adaptation). To produce a standalone GGUF, first [merge the LoRA adapter](https://huggingface.co/docs/peft/main/en/package_reference/lora), convert the merged model to a high-precision GGUF, and then quantize that GGUF following [llama.cpp's quantization workflow](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md). This article does not expand on the training workflow.

## 3.5 Safetensors and GGUF mmap loading and its memory advantages

This section returns to the loading cost mentioned in Section 3.4. Safetensors and GGUF can both use mmap to reduce extra memory consumption during loading because they share a property: **a small index is read first, then offsets directly address contiguous tensor bytes**.

```text
metadata/header
  → name → dtype + shape + offset
  → mmap file
  → tensor view = mapped_base + offset
```

On Linux, `mmap()` maps a range of file bytes into a process's virtual address space; it does not immediately copy the entire file into a user-space heap. When the CPU first accesses a non-resident page, the OS brings that page into memory through page-fault handling. Writes through `MAP_PRIVATE` use copy-on-write, whereas `MAP_SHARED` uses shared-modification semantics; see [mmap(2)](https://man7.org/linux/man-pages/man2/mmap.2.html).

This loading strategy has several benefits:

- It avoids an explicit copy from a full `read()` buffer into a new host buffer of the same size.
- The OS page cache manages pages centrally and can let multiple processes reuse cached pages from the same checkpoint.
- It can skip unused tensors, which is useful for MoE, sharding, and partial component loading.
- The mapped file can serve as backing storage for CPU tensors; see llama.cpp's [llama-mmap.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-mmap.cpp).

mmap is not magic. A dense LLM touches most or all layers for every generated token, so after warm-up most weight pages still need to stay resident. Insufficient memory causes extensive paging and sharply lower throughput. More importantly, an ordinary GPU kernel cannot use a CPU file-mapped address directly as VRAM weights: GPU offload usually still copies or rearranges weights.

Safetensors' layout supports tensor indexing and mmap-friendly loading, but this does not guarantee end-to-end zero-copy behavior in every Python binding or GPU loader. vLLM also offers lazy, eager, and prefetch Safetensors loading strategies; prefetch primarily warms the OS page cache. See [vLLM LoadConfig](https://github.com/vllm-project/vllm/blob/main/vllm/config/load.py).

# 4. Summary

Starting from the minimal FFN, model inference reduces to a clear chain:

```text
model file
  → parse metadata / index
  → recover tensors theta
  → recover or load model semantics f
  → choose CPU/GPU kernels
  → logits
  → argmax or sampling
  → predict
```

Safetensors is a mainstream weight container for training, fine-tuning, Hugging Face, and high-performance GPU serving. GGUF is a practical choice for local inference, quantized distribution, and the llama.cpp ecosystem; it is often released as one file but also supports sharding. ONNX makes the computation graph interchangeable data as well, making it suitable for cross-platform deployment.

The central boundary is:

> **GGUF and Safetensors solve “how to store and retrieve $\theta$ efficiently and reliably”; a runtime's model implementation or an ONNX graph solves “how $\theta$ composes $f$”; CPU/GPU backend kernels solve “how $f$ actually runs.”**

## Experiment

[A reproducible experiment for parsing, loading, and running the three file formats](https://github.com/pochenai/llm-inference-lab/blob/main/w1_model_formats/README.md) uses the same $4 \rightarrow 8 \rightarrow 3$ FFN and loads and executes one weight set through Safetensors/PyTorch, GGUF/ggml, and ONNX/ONNX Runtime. It validates the article's two core boundaries: where weights are recovered from the file, and where $f$ comes from runtime code or an ONNX graph. It also uses an `mmap` versus `read()` comparison to check weight storage behavior and whether logits and `predicted_class` agree across all three backends.
