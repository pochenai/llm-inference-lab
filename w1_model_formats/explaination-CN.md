# 大模型推理文件格式GGUF、Safetensors 与 ONNX对比及工作原理

# 1. Motivation

拿到一个开源模型时，我们经常会遇到 `model.safetensors`、`model.gguf` 与 `model.onnx`。它们看起来都像“模型文件”，却服务于不同问题：训练/微调、跨框架交换、本地 CPU 推理、高吞吐 GPU serving，乃至浏览器与移动端部署。

要理解差异，不能只看扩展名，而要追问：给定输入 `x` 和磁盘上的模型文件，系统还缺少什么信息，才能算出最终的 `predict`？

本文以最简单的两层前馈网络（FFN）为统一例子，沿着“文件字节 → 内存 tensor → 模型函数 → 硬件 kernel → `predict`”的路径，理解三种常见推理文件格式的边界，以及它们如何接入 PyTorch、llama.cpp、vLLM、SGLang 和 ONNX Runtime。


# 2. 从第一性原理推导：运行一个模型究竟需要什么？

一个模型可以抽象为参数化函数：

```text
y = f_θ(x)
```

以一个分类用 FFN 为例：

```text
h       = ReLU(x @ W1ᵀ + b1)
logits  =       h @ W2ᵀ + b2
predict = argmax(logits)
```

`W1`、`b1`、`W2`、`b2` 是持久参数 `θ`；`x`、`h`、`logits` 是本次请求的临时 activation。LLM 的 `logits` 通常还会经过 temperature、top-k/top-p 等采样策略，生成 next token；这些策略通常属于 serving/runtime 配置，而非权重文件。

对于用户实际使用的文本模型，$x$ 并不是原始字符串，而是 tokenizer 编码得到的 token IDs。tokenizer 负责将文本切分并映射为词表中的整数 ID；模型输出的 token ID 再由 tokenizer 解码回文本。因此，端到端过程应写作：

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

本文用 FFN 讨论文件格式时暂时忽略 tokenizer 的内部算法（BPE、SentencePiece 等），但在真实 LLM 部署中，词表、special tokens、chat template 与 tokenizer 配置必须随模型一同提供；否则即使权重和 $f_{\theta}$ 都正确，文本输入/输出的语义也可能错误。

## 2.1 最小存储单元的分层

最小物理存储单位可以是一个 `fp16`、`bf16`、`int8` 或量化后的 bit-packed 数值。但孤立数值没有语义；能被正确读取并参与计算的最小结构是：

```text
Tensor = dtype + shape + layout/stride + raw values
```

而可运行模型至少还需要下面三层信息：

| 层 | 内容 | FFN 示例 |
| --- | --- | --- |
| 数值层 | 参数 tensors `θ` | `W1`、`b1`、`W2`、`b2` |
| 语义/结构层 | 如何将参数组成 `f` | 两次 MatMul、中间 ReLU、张量连接关系 |
| 执行层 | 如何在目标硬件跑算子 | CPU BLAS、CUDA kernel、Metal kernel 等 |

文本 LLM 还需要 tokenizer、词表、特殊 token、chat template、上下文长度与 RoPE 等推理配置。decoder-only LLM 运行时还维护 KV cache；它是动态状态，而不是随模型发布的固定权重。

因此，一个可用推理包可概括为：

```text
Inference package = weights + model semantics + tokenizer + inference metadata
```

训练恢复则还要加入 optimizer state、可能暂存的 gradients、scheduler/trainer state、随机数状态与数据进度。训练 checkpoint 解决的是“如何无缝继续学习”，推理部署包解决的是“如何回答”。

到这里，“一个能运行的模型需要什么”已经明确；但它没有规定这些信息必须怎样装进文件。不同部署目标对三层信息做出了不同取舍：

- 如果目标是安全、通用地分发权重，并让既有训练/推理框架决定函数结构，那么只需重点封装 tensors。
- 如果目标是本地推理和低内存运行，那么文件还应带足够的模型 metadata，并能高效量化与按 offset 读取权重。
- 如果目标是跨框架、跨设备地携带函数语义，那么计算图本身也应成为文件数据。

这三种取舍分别对应 Safetensors、GGUF 与 ONNX。它们并不是三种互相竞争的“同一件东西”，而是针对“权重、函数语义与运行时”三层信息的三种封装边界。

## 2.2 为什么会有多种格式：三种封装策略

下表先给出结论，后续第 3 节再展开每种格式的具体 schema、加载路径和生态支持：

| 格式 | 数值层（weights） | 语义/结构层（`f`） | 执行层 |
| --- | --- | --- | --- |
| Safetensors | 文件内 tensor | 文件外：模型代码与配套 config | 文件外：PyTorch / vLLM / SGLang 等 |
| GGUF | 文件内 tensor，常包含量化信息 | 文件外：llama.cpp 等已实现架构；GGUF 含选择架构的 metadata | 文件外：ggml backend |
| ONNX | graph initializer 中的 tensor | 文件内：标准算子 DAG | 文件外：ONNX Runtime Execution Provider |

**关键澄清：Safetensors 与 GGUF 的主要差异不是“多文件 vs 单文件”。** 单个 `.safetensors` 本身就是一个完整二进制 tensor 文件；但一个可用的 Hugging Face 模型包通常还会有 `config.json`、tokenizer 文件，以及模型太大时的多个 Safetensors shards 和索引文件。反过来，GGUF 通常以一个文件分发，但超大模型同样可以拆成多个 GGUF shards。因此，文件数量是发布策略，不是格式的定义边界。

两者真正的区别在于封装范围和目标 runtime：

| | Safetensors | GGUF |
| --- | --- | --- |
| 核心定位 | 安全、通用的 tensor 容器 | 面向 GGML/llama.cpp 推理的模型容器 |
| 文件告诉 loader 什么 | tensor 的名称、dtype、shape 与字节范围 | 除 tensor 外，还包括架构、超参数、tokenizer、量化等推理 metadata |
| 量化表达 | 可保存量化后的数据，但量化语义通常由外部框架/配置解释 | `ggml_type` 是 tensor 描述的一部分，原生表达 Q4、Q8、IQ 等 ggml 量化布局 |
| 如何得到 $f_{\theta}$ | 外部 config 选择 PyTorch/Transformers/vLLM/SGLang 模型代码 | metadata 选择 llama.cpp 已实现的架构图；图实现仍在 runtime 代码中 |
| 典型用途 | 训练、微调、HF 分发、高吞吐 GPU serving | 单机本地推理、CPU/混合 CPU-GPU、量化分发 |

**同一个浮点 parameter 在文件中怎样保存？** 设线性层权重为 $W \in \mathbb{R}^{m \times n}$，并以 FP16 保存；它在两种格式中都需要名称、shape、dtype 和原始数值，但二进制 schema 的抽象层次不同。

Safetensors 的 header 使用 JSON，直接声明该 tensor 的逻辑信息和数据范围：

```json
{
  "fc1.weight": {
    "dtype": "F16",
    "shape": [m, n],
    "data_offsets": [begin, end]
  }
}
```

随后从 `begin` 到 `end` 是连续、row-major 的 FP16 数值：

$$
\left[w_{1,1}, w_{1,2}, \ldots, w_{m,n}\right]_{\mathrm{FP16}}.
$$

也就是说，Safetensors 的 F16 tensor 与 PyTorch 常用的 dense floating-point tensor 在数值层面直接对齐：loader 可以按 dtype、shape 和连续字节构造 `torch.Tensor`，再将它放入 `nn.Parameter`。格式本身不保存梯度或 optimizer；训练能力来自 PyTorch 对该普通浮点 parameter 建立的 autograd 与 optimizer 系统。

GGUF 的 F16 tensor 也可以保存同类浮点数值，但它的 tensor descriptor 是二进制的，并额外使用 GGML 的类型系统：

```text
tensor info:
  name       = "blk.0.ffn_up.weight"
  n_dims     = 2
  dimensions = [...]
  ggml_type  = GGML_TYPE_F16
  offset     = ...  # relative to tensor-data blob
```

对于未量化的 `GGML_TYPE_F16`，数据区仍是 FP16 值；但 GGUF converter/runtime 还要遵守 GGML 的 tensor 命名、维度约定和架构约定。它不是标准的 PyTorch `state_dict` 接口，因而要回到 PyTorch 训练通常需要专用 converter/loader。

差异在量化 GGUF 时变得决定性。若 `ggml_type = Q4_K` 或其他 Q4/IQ 类型，权重不再是长度为 $mn$ 的普通 FP16 数组，而更接近分块近似：

$$
W_{\mathrm{block}} \approx s_{\mathrm{block}} \cdot q_{\mathrm{packed}} + z_{\mathrm{block}},
$$

其中 $q_{\mathrm{packed}}$ 是压入 2/4/8 bit 等位宽的整数 codes，$s_{\mathrm{block}}$ 与 $z_{\mathrm{block}}$ 是每块的 scale/offset；精确 byte layout 由具体 `ggml_type` 决定。llama.cpp 的 kernel 识别该 type，并在矩阵乘中解码或融合处理它，而 PyTorch 不能把这一串字节直接当作一个可更新的普通 `nn.Parameter`。

因此，“Safetensors 更训练友好”并不是因为它神奇地保存了 PyTorch 专属信息；而是常见的 Safetensors checkpoint 将每个参数保留为普通浮点 tensor，恰好与 PyTorch 的 $W \to \nabla_W L \to W'$ 更新模型对齐。GGUF 的 F16 版本理论上可经专用实现参与训练，但量化 GGUF 的主路径是推理：若要微调，通常回到 FP16/BF16 Safetensors 基座训练或训练 LoRA，再转换、合并并量化为 GGUF。

可以把 Safetensors 理解为“安全、可索引的一箱 tensors”；而 GGUF 是“为 GGML 推理准备的一箱 tensors，附带足以选择已支持架构和量化执行方式的运行线索”。二者都不存任意通用计算图；ONNX 才将 $f_{\theta}$ 的算子 DAG 也保存为文件数据。

Safetensors 不包含通用 `forward()`。一个常见 Hugging Face 目录是：

```text
model/
├── config.json                 # 例如 architectures: ["LlamaForCausalLM"]
├── model.safetensors           # 权重 θ
├── tokenizer.json / tokenizer.model
└── tokenizer_config.json
```

加载时，runtime 从 `config.json` 的 architecture 标识找到已实现的模型类，再按 tensor 名称填入参数：

```python
model = TinyFFN()              # Python 类定义 f
weights = load_safetensors(...)
model.load_state_dict(weights) # 填入 θ
logits = model(x)              # 运行 f_θ(x)
```

GGUF 更偏自描述推理包：它通常带有架构、超参数、tokenizer 与量化 metadata，但不是任意的算子图。比如 `general.architecture = "llama"` 会让 llama.cpp 选择内置的 Llama graph builder，再按 GGUF 约定名称找到相应权重。

ONNX 最不同：它可以把 FFN 写成文件内计算图：

```text
initializers: W1, b1, W2, b2

MatMul(x, W1) → Add(..., b1) → Relu → MatMul(..., W2) → Add(..., b2) → logits
```

ONNX 保存的是可移植算子语义，不是 CUDA kernel；ONNX Runtime 仍需选择 CPU、CUDA、TensorRT、WebGPU 等 Execution Provider 来真正执行节点。

> **GGUF/Safetensors 解决“如何高效、可靠地存取 θ”；runtime 的模型实现或 ONNX graph 解决“θ 如何组成 f”；CPU/GPU backend 的 kernels 解决“f 如何真正跑起来”。**


## 2.3 为什么本文不把 PyTorch checkpoint 作为部署格式讨论重点？

这里不是说 PyTorch 不能推理：PyTorch 是最重要的训练与通用推理框架之一；我们要排除的是把传统 `.pt`/`.pth` checkpoint 当作**跨 runtime 的模型分发格式**这一视角。

一个训练 checkpoint 往往会保存远超推理所需的信息：

```text
训练恢复 checkpoint
= model weights
+ optimizer state（如 Adam 的 momentum / variance）
+ gradients（视保存时机而定）
+ learning-rate scheduler、global step、epoch / dataloader progress
+ RNG state、AMP loss scaler 等 trainer state
```

这些状态的目标是“从中断处继续训练”，对只计算 `predict = f_θ(x)` 的部署并不需要，且 optimizer state 经常让 checkpoint 远大于纯权重。

此外，PyTorch 的传统序列化可通过 Python `pickle` 表示对象。若保存的是完整 `nn.Module` 而不是纯 `state_dict`，恢复会依赖原始 Python 类/模块路径，并且不应加载不可信来源的 pickle 文件。PyTorch 官方也建议以 `state_dict` 为常见保存方式，并对 `torch.load()` 的不可信文件给出安全警告。[PyTorch serialization notes](https://docs.pytorch.org/docs/stable/notes/serialization.html) [torch.load](https://docs.pytorch.org/docs/stable/generated/torch.load.html)

需要保留一个重要限定：**`.pt`/`.pth` 不必然包含 optimizer 或任意 Python 对象。** 它也可以只是纯 `state_dict`；这种情形下它与“权重容器”的距离很近。然而它仍主要面向 PyTorch 生态。本文选择 Safetensors、GGUF 与 ONNX，是因为三者分别更清晰地代表了安全 tensor 容器、量化本地推理包与含可交换计算图的部署格式。

# 3. 三种 file format 的 schema、加载方式与生态

## 3.1 Safetensors：安全、直接的 tensor 容器

Safetensors 的文件布局非常简单：

```text
[8-byte little-endian header length][JSON header][contiguous raw tensor bytes]
```

header 为每个 tensor 保存名称、dtype、shape 与 `data_offsets`：

```json
{
  "fc1.weight": {"dtype": "F16", "shape": [8, 4], "data_offsets": [0, 64]},
  "fc1.bias":   {"dtype": "F16", "shape": [8], "data_offsets": [64, 80]}
}
```

loader 只需读取很小的 header，便能定位 `fc1.weight` 在原始数据区的连续字节范围。格式不执行任意 Python 代码，避免传统 pickle checkpoint 的反序列化风险；见 [Safetensors format](https://github.com/safetensors/safetensors#format)。

它通常是 Hugging Face `transformers` 生态的默认权重格式：模型语义仍由 `config.json` 与 Transformers 模型代码给出，权重以一个或多个 `.safetensors` shard 存放。

在 serving 中：

- **vLLM** 默认优先选择 Safetensors，缺失时再回退 PyTorch 格式；其 loader 在 [vllm/model_executor/model_loader](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/model_loader)。
- **SGLang** 也以 Hugging Face 权重目录为主要输入，自动加载 Safetensors 或 PyTorch checkpoint；见 [SGLang model loading](https://docs.sglang.io/advanced_features/model_loading.html)。
- 二者都用 HF config 的 `architectures` 选择模型类，而不是从 Safetensors 推导任意计算图。vLLM 的架构支持说明在 [supported models](https://github.com/vllm-project/vllm/blob/main/docs/models/supported_models.md)，SGLang 模型实现目录在 [sglang/srt/models](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models)。

## 3.2 GGUF：面向本地推理与量化分发

GGUF 的大体布局是：

```text
magic "GGUF" + version
number of tensors + number of KV pairs
KV metadata
tensor info table: name + dimensions + ggml type + relative offset
aligned tensor-data blob
```

KV metadata 可以携带 `general.architecture`、层数、hidden size、RoPE 配置、tokenizer/词表与量化信息。tensor 信息除了 shape 之外，还带 `ggml_type`，因此能描述 Q4、Q8 等 ggml 量化布局。完整规范见 [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) 与 [llama.cpp GGUF 头文件](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/gguf.h)。

llama.cpp 的核心路径是：

```text
GGUF metadata → architecture enum → C++ model implementation
GGUF named tensors → load_tensors()
C++ build_arch_graph() → ggml graph → backend kernels → logits
```

所以 GGUF 不保存通用计算图。llama.cpp 要支持新架构，必须在源码中实现参数加载与 graph builder；官方开发说明见 [HOWTO-add-model](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md)，实现代码集中在 [src/llama-model.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp) 与 [src/llama-model-loader.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model-loader.cpp)。

GGUF 特别适合 llama.cpp、Ollama、LM Studio 等本地推理环境：它常以单文件形式易于分发，也可按需要分片；CPU 推理和多种量化方案成熟。它不适合作为常规训练/微调 checkpoint，因为修改 tensor 或 metadata 通常意味着重写整个产物。

## 3.3 ONNX：权重与计算图的可交换表示

ONNX 的顶层容器是 `ModelProto`，其中的 `GraphProto` 保存图；每个 `NodeProto` 是一个算子，weights 通常作为 graph initializer tensors。标准说明见 [ONNX IR specification](https://onnx.ai/onnx/repo-docs/IR.html)。

```text
ModelProto
├── GraphProto
│   ├── inputs / outputs
│   ├── initializer tensors
│   └── NodeProto DAG
└── metadata / opset imports
```

这让 ONNX Runtime 可以在不拥有 `TinyFFN.forward()` Python 源码的前提下，解析 `MatMul`、`Add`、`Relu` 等节点并运行模型。session 初始化阶段还会进行图优化；见 [ONNX Runtime graph optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)。运行时再将图分区给 CPU/CUDA/TensorRT 等 Execution Provider，具体机制见 [ONNX Runtime high-level design](https://onnxruntime.ai/docs/reference/high-level-design.html)。

ONNX 因而尤其适合跨框架、跨设备、浏览器/移动端与边缘部署。对于复杂且快速演化的大语言模型，导出是否完整、动态 shape、算子支持和量化路径仍是实际工程约束。

## 3.4 对比：大小、加载与社区支持

不存在脱离具体模型和量化方案的固定文件大小或加载速度。以下是正确的比较维度：

| 维度 | Safetensors | GGUF | ONNX |
| --- | --- | --- | --- |
| 权重精度/大小 | 常见 BF16/FP16；也可存量化 tensor | 常见 Q2–Q8/F16，面向小内存推理 | 与导出精度、外置权重/量化方案有关 |
| 相对大小 | FP16/BF16 时接近原始参数大小 | 量化后通常显著小于 FP16/BF16 | 可接近权重大小，也可能含图与额外常量 |
| 首次加载 | tensor header 可快速索引；还要实例化模型、传入设备 | 通常单文件、也可分片；metadata 完整、常可 mmap | 需解析/校验/优化图，初始化成本取决于图和 EP |
| 主推 runtime | Transformers、PyTorch、vLLM、SGLang | llama.cpp、Ollama 等 | ONNX Runtime、Web/移动端 runtimes |
| `f` 的表达 | 外部 Python/C++ 模型实现 | 外部 runtime 的架构实现 | 文件内 DAG |
| 训练/微调友好性 | 高 | 低 | 通常偏部署，不是完整训练 checkpoint |

“GGUF 更小、更快”并非格式的绝对属性：它通常更小，是因为发布者选择了更低 bit 的量化；“ONNX 更慢”也不成立，实际性能取决于图优化、支持的融合算子、Execution Provider 和目标硬件。

## 3.5 为什么 GGUF 与 Safetensors 都适合 mmap

两种格式都有共同性质：**先读小索引，再通过 offset 直接寻址连续 tensor bytes**。

```text
metadata/header
  → name → dtype + shape + offset
  → mmap 文件
  → tensor view = mapped_base + offset
```

Linux 中，`mmap()` 将文件的一段字节映射到进程虚拟地址空间；它并不立即把整个文件复制到用户态 heap。CPU 首次访问未驻留页时，OS 再通过缺页处理将该页带入内存。`MAP_PRIVATE` 的写入采用 copy-on-write，`MAP_SHARED` 则使用共享修改语义；见 [mmap(2)](https://man7.org/linux/man-pages/man2/mmap.2.html)。

模型加载的好处包括：

- 避免先完整 `read()` 再新建同等大小 host buffer 的显式复制。
- 由 OS page cache 统一管理，并让多个进程复用同一 checkpoint 的缓存页。
- 可跳过不用的 tensor；对 MoE、分片与只加载部分组件有价值。
- 映射文件可作为 CPU tensor 的 backing storage；llama.cpp 的实现见 [llama-mmap.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-mmap.cpp)。

但 mmap 不是魔法：dense LLM 每生成一个 token 都会访问多数甚至全部层，因此热身后大部分权重页仍需驻留；内存不足时会产生大量换页，吞吐会急剧下降。更重要的是，普通 GPU kernel 不能把 CPU 文件映射地址直接当作 VRAM 权重使用：GPU offload 通常仍需复制或重排权重。

Safetensors 的布局支持按 tensor 索引与 mmap-friendly loading，但这不保证每个 Python binding 和 GPU loader 都是端到端零拷贝。vLLM 也提供 lazy、eager 与 prefetch 等 Safetensors 加载策略；其中 prefetch 的重点是预热 OS page cache。参见 [vLLM LoadConfig](https://github.com/vllm-project/vllm/blob/main/vllm/config/load.py)。

# 4. 总结

从一个最小 FFN 出发，每一次模型推理都可还原为一条清晰链路：

```text
model file
  → parse metadata / index
  → recover tensors θ
  → recover or load model semantics f
  → choose CPU/GPU kernels
  → logits
  → argmax or sampling
  → predict
```

Safetensors 是训练、微调、Hugging Face 与高性能 GPU serving 的主流权重容器；GGUF 是本地、量化分发与 llama.cpp 生态的实用选择，常以单文件发布但也支持分片；ONNX 则将计算图也变成可交换数据，适合跨平台部署。

最后回到最重要的边界：

> **GGUF/Safetensors 解决“如何高效、可靠地存取 θ”；runtime 的模型实现或 ONNX graph 解决“θ 如何组成 f”；CPU/GPU backend 的 kernels 解决“f 如何真正跑起来”。**
