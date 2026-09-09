# 大模型推理文件格式GGUF、Safetensors 与 ONNX对比及工作原理

# 1. Motivation

看到一个Hugging face上的开源模型时，我们经常会遇到 `model.safetensors`、`model.gguf` 与 `model.onnx`。它们看起来都像“模型文件”，却服务于不同问题：训练/微调、跨框架交换、本地 CPU 推理、高吞吐 GPU serving，乃至浏览器与移动端部署。

要理解差异，不能只看扩展名，而要追问：给定输入 `x` 和磁盘上的模型文件，系统还缺少什么信息，才能算出最终的 `predict`？

本文以最简单的两层前馈网络（FFN）为统一例子，沿着“文件字节格式 → 内存 tensor → 模型函数 → 硬件 kernel → `predict`”的路径，理解三种常见推理文件格式的边界，以及它们如何接入 PyTorch、llama.cpp、vLLM、SGLang 和 ONNX Runtime。


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

根据上述 FFN 的参数与计算过程，可运行模型至少需要下面三层信息：

| 层 | 内容 | FFN 示例 |
| --- | --- | --- |
| 数值层 | 参数 tensors `θ` | `W1`、`b1`、`W2`、`b2` |
| 语义/结构层 | 如何将参数组成 `f` | 两次 MatMul、中间 ReLU、张量连接关系 |
| 执行层 | 如何在目标硬件跑算子 | CPU BLAS、CUDA kernel、Metal kernel 等 |

其中，数值层以 tensor 为基本组织单位。tensor 的元素可以用 `fp16`、`bf16`、`int8` 等格式表示，也可以采用量化后的 bit-packed 编码；块量化通常还需要配套的 scale 等信息。读取时，除了原始数值，还需要 dtype、shape 和布局约定，才能将文件字节还原为可参与计算的 tensor。

文本 LLM 还需要 tokenizer、词表、特殊 token、chat template、上下文长度与 RoPE 等推理配置。decoder-only LLM 运行时还维护 KV cache；它是动态状态，而不是随模型发布的固定权重。

因此，一个可用推理包可概括为：

```text
Inference package = weights + model semantics + tokenizer + inference metadata
```

到这里，“一个能运行的模型需要什么”已经明确；但它没有规定这些信息必须怎样装进文件。不同部署目标对三层信息做出了不同取舍：

- 如果目标是安全、通用地分发权重，并让既有训练/推理框架决定函数结构，那么只需重点封装 tensors。
- 如果目标是本地推理和低内存运行，那么文件还应带足够的模型 metadata，并能高效量化与按 offset 读取权重。
- 如果目标是跨框架、跨设备地携带函数语义，那么计算图本身也应成为文件数据。

这三种取舍分别对应 Safetensors、GGUF 与 ONNX。它们并不是三种互相竞争的“同一件东西”，而是针对“权重、函数语义与运行时”三层信息的三种封装边界。

> 说明：训练恢复则还要加入 optimizer state、可能暂存的 gradients、scheduler/trainer state、随机数状态与数据进度。训练 checkpoint 解决的是“如何无缝继续学习”，推理部署包解决的是“如何回答”。本文不考虑训练相关存储。

## 2.2 三种推理文件格式及其封装策略

上一节明确了运行模型所需的三层信息。接下来要比较的是：哪些信息由模型文件保存，哪些由 runtime 提供？Safetensors、GGUF 与 ONNX 的区别，就在这条封装边界上。

**Safetensors 以参数 tensor 为封装核心。** 对前面的 FFN，它可以保存 $W_1$、$b_1$、$W_2$、$b_2$ 的数值及读取这些数值所需的 dtype、shape 等信息，但不描述“两次线性变换之间接 ReLU”的计算关系。加载端仍需提供 FFN 实现，再按名称填入参数。常见 Hugging Face 模型包通过文件外的模型代码与配套 `config.json` 完成这一步；Safetensors 本身不提供通用 `forward()`。[Safetensors 格式规范](https://github.com/safetensors/safetensors#format)

**GGUF 在 tensor 之外，还封装面向目标 runtime 的模型信息。** 例如架构标识、超参数、tokenizer，以及用于解释权重布局的 `ggml_type`。因此，它通常能减少推理时对外部配置文件的依赖。但这些 metadata 不是计算图：例如 `general.architecture = "llama"` 是让 llama.cpp 选择已实现的 Llama 架构，而不是从文件中读出任意函数。如果用 GGUF 封装前面的 FFN，加载端同样需要有对应的计算实现。[GGUF 格式规范](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)

**ONNX 将计算关系也封装为图。** 对同一个 FFN，权重可以作为图中的 initializer，线性变换、ReLU 及它们的连接关系则由算子节点描述。加载端不必拥有原来的 `TinyFFN.forward()`，但必须能够执行图中使用的算子；图的算子语义不等于 CPU/GPU kernel 实现。[ONNX IR 规范](https://onnx.ai/onnx/repo-docs/IR.html)

对应到上一节的三层信息，三种格式的分工如下：

| 格式 | 数值层：参数 $\theta$ | 语义/结构层：参数如何组成 $f$ | 执行层：算子如何运行 |
| --- | --- | --- | --- |
| Safetensors | 文件保存 tensor 的描述与数值 | 文件外的模型代码与配套配置 | runtime 的 CPU/GPU 等后端 |
| GGUF | 文件保存 tensor、类型及相关量化信息 | 文件内 metadata 选择和配置 runtime 已实现的架构；不保存通用计算图 | ggml 等兼容 runtime 的后端 |
| ONNX | 图中的 initializer，可内嵌或引用外部 tensor 数据 | 文件内的计算图与算子语义 | ONNX Runtime 等兼容 runtime 的后端 |

因此，Safetensors 与 GGUF 都需要模型实现来组织参数；GGUF 额外约定了更多模型与权重表示信息。ONNX 则把计算图交给 runtime 解析。三者都不能只凭模型文件完成执行，还需要相应的算子实现。具体字段与加载路径在第 3 节展开。

> GGUF/Safetensors 解决“如何高效、可靠地存取 $\theta$”；runtime 的模型实现或 ONNX graph 解决“$\theta$ 如何组成 $f$”；CPU/GPU backend 的 kernels 解决“$f$ 如何真正跑起来”。

> 说明：文件数量不是上述封装边界。Safetensors 与 GGUF 都可以单文件或分片发布；常见 Hugging Face 模型目录中的配置与 tokenizer 文件，也不属于 Safetensors 权重文件本身。


## 2.3 PyTorch checkpoint 与部署模型包的边界

上一节按“文件封装了哪几层信息”比较 Safetensors、GGUF 与 ONNX。PyTorch 的 `.pt`/`.pth` 文件看似也保存模型，但它们不是另一种固定 schema：这是 `torch.save()` 的常见文件扩展名，文件实际保存什么由调用者交给它的 Python 对象决定。[PyTorch 保存与恢复模型指南](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

最轻量的情形是保存 `state_dict`。它是参数名到 tensor（以及已注册 buffer）的映射：

$$
\texttt{state\_dict} = \{\texttt{name} \mapsto \texttt{tensor}, \ldots\}
$$

对前面的 FFN，`state_dict` 可以包含 `fc1.weight`、`fc1.bias`、`fc2.weight` 与 `fc2.bias`。加载时仍要先在 PyTorch 代码中构造 `TinyFFN`，再把同名参数填入其中；也就是说，文件给出 $\theta$，而 `TinyFFN.forward()` 给出 $f$。在“参数如何组成函数”这一层，它与 Safetensors 的边界相近，而不等同于包含计算图的 ONNX。

训练恢复常用的 checkpoint 则会将这组权重与额外状态一起保存，例如：

$$
\begin{aligned}
\text{training checkpoint} ={}& \text{model state} + \text{optimizer state} \\
&+ \text{scheduler / step / data progress} \\
&+ \text{optional RNG, AMP, or other trainer state}
\end{aligned}
$$

这些字段用于从中断处继续训练，对只计算 $\operatorname{predict}(x) = f_\theta(x)$ 的部署不是必要信息；optimizer state 也常使一般 checkpoint 大于纯模型状态。实际字段由训练代码决定，因此 `.pt`/`.pth` 后缀本身不能说明文件是否含 optimizer、gradient 或其他训练状态。

还存在保存完整 `nn.Module` 的情形。`torch.save(model, path)` 会借助 Python `pickle` 序列化整个模块对象；恢复时仍依赖原始类定义及其模块路径。PyTorch 将保存/加载 `state_dict` 作为推荐方式，并提醒不可信来源的文件不应以允许任意对象反序列化的方式加载；`weights_only=True` 是加载权重状态时更受限的路径。[PyTorch 序列化说明](https://docs.pytorch.org/docs/stable/notes/serialization.html) [PyTorch 的 torch.load 安全说明](https://docs.pytorch.org/docs/stable/generated/torch.load.html)

因此，本文不把 PyTorch checkpoint 作为第四种部署格式，并不是因为 PyTorch 不能推理：纯 `state_dict` 很适合在 PyTorch 内恢复参数，训练 checkpoint 适合恢复训练，完整 `nn.Module` 适合紧耦合的 Python 环境。它们的封装边界由 PyTorch 序列化与调用者的对象选择决定；Safetensors、GGUF 与 ONNX 则分别代表了本文要比较的安全 tensor 容器、面向本地量化 runtime 的模型包，以及可交换计算图。

# 3. 三种 file format 的 schema、加载方式与生态

上一节明确了三种格式的封装边界；本节详细介绍它们的 schema、加载方式与社区生态支持。

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

一个常见的模型目录是：

```text
model/
├── config.json                 # Model architecture and hyperparameters
├── model.safetensors           # Named parameter tensors
├── tokenizer.json              # Or tokenizer.model, depending on the model
└── tokenizer_config.json
```

以前面的 FFN 为例，loader 根据 header 中的 dtype、shape 与数据偏移读取参数，再交给已定义计算过程的模型。下面是加载职责的伪代码：

```python
model = TinyFFN()              # Defines the computation
weights = load_safetensors(...)  # Reads named tensors
model.load_state_dict(weights)   # Fills the parameters
logits = model(x)
```

上面的 `fc1.weight` 示例包含 $8 \times 4$ 个 F16 元素，数据区占 $8 \times 4 \times 2 = 64$ 字节。它保存的是连续的普通浮点数值；loader 可以按 dtype 和 shape 构造浮点 tensor，而不必先解码一种专用量化表示。这里说明的是参数的数值表示，是否复制、迁移设备或注册为 `nn.Parameter` 由加载代码决定。

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

同一个 FP16 权重 tensor，在 GGUF 中由二进制 descriptor 描述。下面省略具体维度值，仅展示字段；名称与维度顺序需要遵守目标架构的 GGML 约定：

```text
tensor info:
  name       = "blk.0.ffn_up.weight"
  n_dims     = 2
  dimensions = [...]
  ggml_type  = GGML_TYPE_F16
  offset     = ...  # Relative to the tensor-data blob
```

当类型是 `GGML_TYPE_F16` 时，数据区仍是普通 FP16 数值。GGUF 与 Safetensors 的区别在于 descriptor、命名与布局约定；两者都能保存量化后的 tensor 数据，而 GGUF 的 `ggml_type` 进一步定义 `Q4_K` 等具体编码布局。

当类型改为 `Q4_K` 等量化类型时，数据区才变为对应的分块编码（将一组权重的低位 codes 与共享 scale/min 等元数据一起打包存储）。例如 `block_q4_K` 包含打包的四位 codes、子块 scale/min 信息以及 super-block scale；加载端必须按该类型解释字节，kernel 再解码或融合处理它们。它不能直接作为普通浮点参数数组使用。不同 Q4/IQ 类型的编码并不相同，具体布局见 [ggml 量化块定义](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h)。

llama.cpp 的核心路径是：

```text
GGUF metadata → architecture enum → C++ model implementation
GGUF named tensors → load_tensors()
C++ build_arch_graph() → ggml graph → backend kernels → logits
```

可以看出，GGUF 不保存通用计算图。因此，llama.cpp 若要支持新架构，必须在源码中实现参数加载与 graph builder；官方开发说明见 [HOWTO-add-model](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md)，实现代码集中在 [src/llama-model.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp) 与 [src/llama-model-loader.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model-loader.cpp)。

GGUF 特别适合 llama.cpp、Ollama、LM Studio 等本地推理环境：它常以单文件形式易于分发，也可按需要分片；CPU 推理和多种量化方案成熟。

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

ONNX 的可交换计算图和多种 Execution Provider 支持，使其适合跨框架、跨设备、浏览器/移动端与边缘部署。但这不表示任意大语言模型都能直接获得同样的可移植性；部署前通常要确认以下四点：

- **导出是否完整**：模型的 `forward()`、控制流与所用算子能否都转换成目标 ONNX 图，而不是在某个不支持的操作处失败或需要自定义实现。[PyTorch ONNX 导出器](https://docs.pytorch.org/docs/stable/onnx_export.html)
- **动态 shape**：batch size、sequence length 等维度能否在运行时变化，而不是只接受导出时示例输入的固定大小。[PyTorch 动态 shape 说明](https://docs.pytorch.org/docs/stable/onnx_export.html)
- **算子支持**：目标 Execution Provider 能否执行图中的节点；不支持的节点会被分给其他 provider，可能带来额外拷贝或性能下降。[ONNX Runtime Execution Provider 机制](https://onnxruntime.ai/docs/execution-providers/)
- **量化路径**：量化后的 ONNX 图表示、算子与目标硬件是否匹配，并在可接受的精度损失下带来预期收益。[ONNX Runtime 量化说明](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

## 3.4 三种格式在大小、加载与生态中的部署取舍

前面三节说明了每种格式分别保存什么。本节进一步回答部署时如何选择：先确定是否需要可交换计算图、低内存量化或既有训练框架的权重目录；再比较由权重精度、模型架构、runtime 与目标硬件共同决定的大小和加载成本。格式名称本身不承诺固定的文件大小或加载速度。

| 比较轴 | Safetensors | GGUF | ONNX |
| --- | --- | --- | --- |
| 首要部署目标 | 在既有模型代码下安全、直接地分发 named tensors | 在兼容 runtime 中分发带模型 metadata 的量化权重 | 连同计算图一起交换和部署模型 |
| 常见权重表示与相对大小 | 常见 BF16/FP16；也可保存量化后的 tensor 数据。浮点权重时接近原始参数大小 | 常见 Q2–Q8/F16；较低 bit 的量化通常使文件小于浮点权重 | 取决于导出精度、initializer 是否外置与量化方案；文件还可能包含图和额外常量 |
| 首次加载的主要工作 | 读取 header 定位 tensor，再由外部模型代码实例化参数并迁移设备；支持的 loader 可用 mmap（内存映射）读取 | 读取 metadata/tensor descriptor，按 `ggml_type` 建立权重视图或解码；支持的 runtime 也可用 mmap（内存映射）读取 | 解析、校验和优化计算图，再按 Execution Provider 分区；成本取决于图和目标后端 |
| 常见集成路线 | Hugging Face `transformers` 权重目录，以及 vLLM、SGLang 等 serving runtime | llama.cpp、Ollama、LM Studio 等本地推理 runtime | ONNX Runtime，以及 Web、移动端和边缘设备 runtime |
| 函数 $f$ 的来源 | 文件外的 Python/C++ 模型实现 | runtime 中已实现的目标架构 | 文件内的算子 DAG |
| 与常规浮点训练路径的衔接 | 常见浮点参数可直接交给训练框架 | 需适配 GGML 表示；量化权重还涉及解码或专用训练支持 | 图交换不等于完整训练状态恢复 |

表中的“常见集成路线”描述的是社区通常如何加载这些产物，并不表示某个 runtime 只支持这一种文件格式；同样，单文件或分片是发布策略，不是 Safetensors 与 GGUF 的格式分界。

由此可以得到三个直接的选择：已有 Hugging Face/PyTorch 模型代码并希望训练、微调或 GPU serving 时，Safetensors 是自然的权重容器；目标是低内存的本地 CPU/GPU 推理时，GGUF 的量化布局与 metadata 更贴近 llama.cpp 类 runtime；需要把计算关系一起交给不同设备和 runtime 时，ONNX 的图表示更合适。实际性能仍取决于量化 bit 数、图优化、融合算子、Execution Provider 与目标硬件，而不是格式名称。

> 说明：“训练友好”描述的是文件中的参数表示与训练系统的衔接成本，不是格式自带训练能力。常见 Safetensors checkpoint 保存普通 FP16/BF16 tensor，与 PyTorch 的浮点参数更新路径对齐；autograd 和 optimizer 仍由框架提供。GGUF 的 F16 权重也可经适配后使用，但需要处理 GGML 的命名、维度和类型约定；量化 GGUF 则通常还需解码成浮点参数，或配套专用的量化训练方法。Safetensors 也能存放量化数据，因此不能只看扩展名判断训练便利性。一种常见工作流是保留浮点基座进行微调或训练 LoRA（Low-Rank Adaptation，低秩适配）；若要得到独立的 GGUF，先[合并 LoRA adapter](https://huggingface.co/docs/peft/main/en/package_reference/lora)，再将合并后的模型转换为高精度 GGUF，最后按 [llama.cpp 的量化流程](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)量化该 GGUF。本文不展开训练流程。

## 3.5 Safetensors 与 GGUF 的 mmap 加载及其内存优势

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

## 实验

[三种文件格式解析、加载和 runtime 的可复现实验](https://github.com/pochenai/llm-inference-lab/blob/main/w1_model_formats/README.md) 使用同一个 $4 \rightarrow 8 \rightarrow 3$ 的 FFN，将同一组权重分别交给 Safetensors/PyTorch、GGUF/ggml 与 ONNX/ONNX Runtime 加载和执行。它可验证本文的两个核心边界：权重从文件的何处恢复，$f$ 从运行时代码或 ONNX graph 的何处获得；同时通过 `mmap` 与 `read()` 的对照检查权重存储方式，以及三个后端的 logits 与 prediction 是否一致。
