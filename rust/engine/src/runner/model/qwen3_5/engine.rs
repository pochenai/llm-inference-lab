//! Typed text-only Qwen3.5 MoE decoder with explicit per-session hybrid state.
//!
//! Qwen3.5-35B-A3B (T = number of uncached input tokens):
//! ```text
//! Token IDs [T]
//!     |
//! Embedding [248320, 2048] -> hidden [T, 2048]
//!     |
//! +---------------- Decoder block x 40 ----------------+
//! | hidden --------------------------------------+    |
//! |   |                                          |    |
//! | Zero-centered RMSNorm                        |    |
//! |   |                                          |    |
//! | Mixer: [DeltaNet, DeltaNet, DeltaNet, GQA] x 10     |
//! |   |                                          |    |
//! | O projection -> hidden [T, 2048]              |    |
//! |   |                                          |    |
//! | Add <----------------------------------------+    |
//! |   |                                               |
//! |   +------------------------------------------+    |
//! | Zero-centered RMSNorm                        |    |
//! |   |                                          |    |
//! | MoE: top 8 of 256 routed SwiGLU experts       |    |
//! |      + sigmoid-gated shared SwiGLU expert     |    |
//! |      Each expert: 2048 -> 512 -> 2048         |    |
//! |   |                                          |    |
//! | Add <----------------------------------------+    |
//! +--------------|------------------------------------+
//!                |
//! Last-token hidden -> final zero-centered RMSNorm
//!                |
//! Untied LM head [248320, 2048] -> FP32 logits [248320]
//!                |
//! Greedy argmax -> next token ID
//!
//! DeltaNet mixer (30 layers):
//!   Q/K/V -> causal Conv1d(4) + SiLU -> Q/K L2 norm
//!        -> gated delta recurrence -> RMSNorm * SiLU(z)
//!   Q/K: 16 x 128; V: 32 x 128 (repeat each Q/K head twice)
//!   Session state: 3 raw convolution inputs + FP32 [32, 128, 128]
//!
//! GQA mixer (10 layers):
//!   Q/K/V -> per-head zero-centered Q/K RMSNorm + partial RoPE
//!        -> causal GQA -> sigmoid output gate
//!   Q: 16 x 256; K/V: 2 x 256; 64 rotary channels/head
//!   Session state: rotated BF16 K + BF16 V
//! ```

use std::{ffi::CString, path::Path, sync::Arc};

use crate::profiling::ScopedRange;

use anyhow::{ensure, Result};
use cuda_async::{device_operation::*, scheduling_policies::SchedulingPolicy};
use cutile::{
    api,
    core::{bf16, ElementType},
    cuda_core::DeviceCopy,
    tensor::{PartitionMut, Reshape, Tensor},
    tile_kernel::TileKernel,
    DType,
};
use tokenizers::Tokenizer;

use super::{
    config::{Qwen3_5Config, Qwen3_5LayerType},
    constants::*,
    graph::{
        ComputeOutput, CudaGraphMetrics, DecodeGraph, DecodeGraphs, Execution, GraphCacheStorage,
        GraphData,
    },
    kernels,
    state::{retained_pointer, AttentionView, Qwen3_5Context, Qwen3_5ExecutionView},
    workspace::{Scratch, ScratchTensor, WorkspaceMetrics, WorkspacePool},
    Qwen3_5StatePool,
};
use crate::runner::{
    kv_pool::NaiveKVPoolConfig,
    loader::{EosTokenId, WeightElement, WeightLoader},
    model::{
        qwen2::kernels as shared, readback::Readback, CausalModel, ModelError, ModelInput,
        NextToken,
    },
};

type Weight<T> = Arc<Tensor<T>>;
type Norm = Arc<Tensor<f32>>;

struct Delta<T: DType> {
    qkv: Weight<T>,
    z: Weight<T>,
    a: Weight<T>,
    b: Weight<T>,
    out: Weight<T>,
    conv: Weight<T>,
    a_log: Norm,
    dt: Norm,
    norm: Norm,
}
struct Attention<T: DType> {
    q: Weight<T>,
    k: Weight<T>,
    v: Weight<T>,
    out: Weight<T>,
    q_norm: Norm,
    k_norm: Norm,
}
enum Mixer<T: DType> {
    Delta(Delta<T>),
    Attention(Attention<T>),
}
struct Layer<T: DType> {
    input_norm: Norm,
    post_norm: Norm,
    mixer: Mixer<T>,
    router: Weight<T>,
    gate_up: Weight<T>,
    down: Weight<T>,
    shared_gate: Weight<T>,
    shared_up: Weight<T>,
    shared_down: Weight<T>,
    shared_router: Weight<T>,
}
/// Typed text backbone; T selects floating-point weight/activation/state storage.
/// Norm metadata, reductions, recurrence and diagnostic logits remain FP32.
pub struct Qwen3_5Engine<T: DType = bf16> {
    config: Qwen3_5Config,
    tokenizer: Tokenizer,
    embedding: Weight<T>,
    layers: Vec<Layer<T>>,
    layer_ranges: Vec<CString>,
    norm: Norm,
    lm_head: Weight<T>,
    frequencies: Norm,
    eos: Vec<u32>,
    policy: Arc<dyn SchedulingPolicy>,
    pub(super) owner: Arc<()>,
    weight_bytes: usize,
    workspaces: Arc<WorkspacePool<T>>,
    pub(super) graphs: Option<DecodeGraphs<T>>,
    uploads: Arc<crate::runner::upload::UploadPool<u32>>,
    readbacks: crate::runner::model::readback::ReadbackCache<f32>,
    token_readbacks: crate::runner::model::readback::ReadbackCache<i32>,
}

impl<T: WeightElement + ElementType> Qwen3_5Engine<T> {
    pub async fn load(path: &Path, policy: &Arc<dyn SchedulingPolicy>) -> Result<Self> {
        let loader = WeightLoader::new(path)?;
        let config: Qwen3_5Config = loader.config()?;
        config.validate()?;
        let c = &config.text_config;
        let tokenizer = loader.tokenizer()?;
        let mut weight_bytes = 0;
        async fn weight<W: WeightElement>(
            loader: &WeightLoader,
            name: &str,
            shape: &[usize],
            policy: &Arc<dyn SchedulingPolicy>,
            bytes: &mut usize,
        ) -> Result<Arc<Tensor<W>>> {
            let host = loader.load_host::<W>(name)?;
            ensure!(
                host.shape == shape,
                "{name}: expected {shape:?}, got {:?}",
                host.shape
            );
            *bytes += host.data.len() * size_of::<W>();
            Ok(Arc::new(
                api::copy_host_vec_to_device(&Arc::new(host.data))
                    .schedule(policy)?
                    .await?
                    .reshape(shape)?,
            ))
        }
        macro_rules! load {
            ($name:expr, $shape:expr) => {
                weight(&loader, &$name, &$shape, policy, &mut weight_bytes).await?
            };
        }
        let h = c.hidden_size;
        let i = c.moe_intermediate_size;
        let sh = c.shared_expert_intermediate_size;
        let kh = c.linear_num_key_heads;
        let vh = c.linear_num_value_heads;
        let channels = kh * GDN_QK_CHANNELS_PER_HEAD + vh * GDN_VALUE_HEAD_DIM;
        let embedding = load!(
            "model.language_model.embed_tokens.weight",
            [c.vocab_size, h]
        );
        let lm_head = load!("lm_head.weight", [c.vocab_size, h]);
        let norm = load!("model.language_model.norm.weight", [h]);
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for (index, kind) in c.layer_types.iter().enumerate() {
            let p = format!("model.language_model.layers.{index}");
            let mixer = match kind {
                Qwen3_5LayerType::LinearAttention => {
                    // Transpose the small convolution once to make channel tiles contiguous.
                    let host = loader.load_host::<T>(&format!("{p}.linear_attn.conv1d.weight"))?;
                    ensure!(
                        host.shape == [channels, 1, CONV_KERNEL_SIZE],
                        "invalid convolution weight shape"
                    );
                    let mut transposed: Vec<T> = Vec::with_capacity(host.data.len());
                    transposed.extend(
                        (0..CONV_KERNEL_SIZE)
                            .flat_map(|tap| (0..channels).map(move |channel| (tap, channel)))
                            .map(|(tap, channel)| host.data[channel * CONV_KERNEL_SIZE + tap]),
                    );
                    weight_bytes += host.data.len() * size_of::<T>();
                    let conv = Arc::new(
                        api::copy_host_vec_to_device(&Arc::new(transposed))
                            .schedule(policy)?
                            .await?
                            .reshape(&[CONV_KERNEL_SIZE, channels])?,
                    );
                    Mixer::Delta(Delta {
                        qkv: load!(format!("{p}.linear_attn.in_proj_qkv.weight"), [channels, h]),
                        z: load!(
                            format!("{p}.linear_attn.in_proj_z.weight"),
                            [vh * GDN_VALUE_HEAD_DIM, h]
                        ),
                        a: load!(format!("{p}.linear_attn.in_proj_a.weight"), [vh, h]),
                        b: load!(format!("{p}.linear_attn.in_proj_b.weight"), [vh, h]),
                        out: load!(
                            format!("{p}.linear_attn.out_proj.weight"),
                            [h, vh * GDN_VALUE_HEAD_DIM]
                        ),
                        conv,
                        a_log: load!(format!("{p}.linear_attn.A_log"), [vh]),
                        dt: load!(format!("{p}.linear_attn.dt_bias"), [vh]),
                        norm: load!(format!("{p}.linear_attn.norm.weight"), [GDN_VALUE_HEAD_DIM]),
                    })
                }
                Qwen3_5LayerType::FullAttention => Mixer::Attention(Attention {
                    q: load!(
                        format!("{p}.self_attn.q_proj.weight"),
                        [c.num_attention_heads * GATED_QUERY_HEAD_DIM, h]
                    ),
                    k: load!(
                        format!("{p}.self_attn.k_proj.weight"),
                        [c.num_key_value_heads * ATTENTION_HEAD_DIM, h]
                    ),
                    v: load!(
                        format!("{p}.self_attn.v_proj.weight"),
                        [c.num_key_value_heads * ATTENTION_HEAD_DIM, h]
                    ),
                    out: load!(
                        format!("{p}.self_attn.o_proj.weight"),
                        [h, c.num_attention_heads * ATTENTION_HEAD_DIM]
                    ),
                    q_norm: load!(format!("{p}.self_attn.q_norm.weight"), [ATTENTION_HEAD_DIM]),
                    k_norm: load!(format!("{p}.self_attn.k_norm.weight"), [ATTENTION_HEAD_DIM]),
                }),
            };
            let gu: Weight<T> = load!(
                format!("{p}.mlp.experts.gate_up_proj"),
                [c.num_experts, 2 * i, h]
            );
            let down: Weight<T> =
                load!(format!("{p}.mlp.experts.down_proj"), [c.num_experts, h, i]);
            layers.push(Layer {
                input_norm: load!(format!("{p}.input_layernorm.weight"), [h]),
                post_norm: load!(format!("{p}.post_attention_layernorm.weight"), [h]),
                mixer,
                router: load!(format!("{p}.mlp.gate.weight"), [c.num_experts, h]),
                gate_up: gu.reshape(&[c.num_experts * 2 * i, h])?,
                down: down.reshape(&[c.num_experts * h, i])?,
                shared_gate: load!(format!("{p}.mlp.shared_expert.gate_proj.weight"), [sh, h]),
                shared_up: load!(format!("{p}.mlp.shared_expert.up_proj.weight"), [sh, h]),
                shared_down: load!(format!("{p}.mlp.shared_expert.down_proj.weight"), [h, sh]),
                shared_router: load!(format!("{p}.mlp.shared_expert_gate.weight"), [1, h]),
            });
        }
        let freq: Vec<f32> = (0..ROTARY_DIM)
            .map(|j| {
                c.rope_parameters
                    .rope_theta
                    .powf(-2.0 * j as f32 / ROTARY_DIM as f32)
            })
            .collect();
        let frequencies = Arc::new(
            api::copy_host_vec_to_device(&Arc::new(freq))
                .schedule(policy)?
                .await?,
        );
        let mut eos = vec![c.eos_token_id];
        if let Some(generation) = loader.generation_config()? {
            if let Some(ids) = generation.eos_token_id {
                eos = match ids {
                    EosTokenId::One(id) => vec![id],
                    EosTokenId::Many(ids) => ids,
                };
            }
        }
        ensure!(
            eos.iter().all(|id| (*id as usize) < c.vocab_size),
            "invalid EOS token"
        );
        let layer_ranges = (0..layers.len())
            .map(|index| {
                CString::new(format!("qwen3_5.layer.{index}")).expect("layer label has no NUL")
            })
            .collect();
        Ok(Self {
            layer_ranges,
            config,
            tokenizer,
            embedding,
            layers,
            norm,
            lm_head,
            frequencies,
            eos,
            policy: policy.clone(),
            owner: Arc::new(()),
            weight_bytes,
            workspaces: Arc::default(),
            graphs: None,
            uploads: Arc::default(),
            readbacks: Default::default(),
            token_readbacks: Default::default(),
        })
    }

    /// Capture and replay single-token pooled decode; prefill remains eager.
    pub fn with_cuda_graphs(mut self) -> Self {
        self.graphs = Some(DecodeGraphs::default());
        self
    }
    pub fn cuda_graph_metrics(&self) -> CudaGraphMetrics {
        self.graphs
            .as_ref()
            .map_or_else(Default::default, DecodeGraphs::metrics)
    }

    pub fn workspace_metrics(&self) -> WorkspaceMetrics {
        self.workspaces.metrics()
    }

    pub fn config(&self) -> &Qwen3_5Config {
        &self.config
    }
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    pub fn is_eos(&self, token: u32) -> bool {
        self.eos.contains(&token)
    }
    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    fn validate_max_seq_len(&self, max_seq_len: usize) -> Result<()> {
        let c = &self.config.text_config;
        ensure!(
            max_seq_len > 0
                && max_seq_len <= c.max_position_embeddings
                && max_seq_len
                    .checked_mul(c.num_key_value_heads * ATTENTION_HEAD_DIM)
                    .is_some_and(|n| n <= i32::MAX as usize),
            "invalid maximum session sequence length"
        );
        Ok(())
    }

    pub fn kv_pool_config(&self, max_seq_len: usize) -> Result<NaiveKVPoolConfig> {
        self.validate_max_seq_len(max_seq_len)?;
        Ok(NaiveKVPoolConfig {
            num_layers: self
                .config
                .text_config
                .layer_types
                .iter()
                .filter(|&&kind| kind == Qwen3_5LayerType::FullAttention)
                .count(),
            num_slots: max_seq_len,
            max_rows: 1,
            max_seq_len,
            head_dim: self.config.text_config.num_key_value_heads * ATTENTION_HEAD_DIM,
        })
    }

    /// Allocate standalone state for at most max_seq_len total tokens.
    pub async fn new_context(&self, max_seq_len: usize) -> Result<Qwen3_5Context<T>> {
        self.validate_max_seq_len(max_seq_len)?;
        with_context(|exec| {
            value(unsafe {
                Qwen3_5Context::allocate(
                    &self.config.text_config,
                    self.owner.clone(),
                    max_seq_len,
                    false,
                    exec,
                )
            })
        })
        .schedule(&self.policy)?
        .await?
    }

    /// Append the uncached suffix, completing GPU work before publishing history.
    pub async fn step_with_logits(
        &self,
        tokens: &[u32],
        state: &mut Qwen3_5Context<T>,
    ) -> Result<(u32, Vec<f32>)> {
        self.validate_input(tokens)?;
        state.validate(&self.owner, tokens)?;
        let suffix = Arc::new(tokens[state.ready_len()..].to_vec());
        state
            .execute(
                &self.owner,
                tokens,
                &self.policy,
                |view, exec| unsafe { self.submit(&suffix, view, exec) },
                |(logits, token)| {
                    let next = validate_prediction(token.as_slice()[0], logits.as_slice())?;
                    Ok((next, logits.as_slice().to_vec()))
                },
            )
            .await
    }

    pub(super) fn prepare_decode_graph(
        &self,
        input: &ModelInput<'_, Qwen3_5StatePool<T>>,
    ) -> Result<Option<Arc<DecodeGraph<T>>>> {
        let Some(graphs) = &self.graphs else {
            return Ok(None);
        };
        if input.uncached_len() != 1 || input.kv.ready_len() == 0 {
            return Ok(None);
        }
        let view = input.kv.read_view(0)?;
        // SAFETY: retain backing storage only; each replay grants its own pending write lease.
        let table = unsafe { view.tensors().2.clone() };
        let key = Arc::as_ptr(&table) as usize;
        Ok(Some(graphs.acquire(key, || {
            let _capture = ScopedRange::new(c"qwen3_5.decode.graph_capture");
            let mut layers = Vec::new();
            for index in 0..self
                .config
                .text_config
                .layer_types
                .iter()
                .filter(|&&kind| kind == Qwen3_5LayerType::FullAttention)
                .count()
            {
                let view = input.kv.read_view(index)?;
                // SAFETY: no request ownership is captured in the graph cache.
                let (k, v, _) = unsafe { view.tensors() };
                layers.push((k.clone(), v.clone()));
            }
            let cache = GraphCacheStorage { layers, table };
            let c = &self.config.text_config;
            ensure!(
                cache.table.context().cu_ctx() == self.embedding.context().cu_ctx()
                    && cache.layers.iter().all(|(k, v)| k.shape() == v.shape()
                        && k.shape()[1] as usize == c.num_key_value_heads * ATTENTION_HEAD_DIM),
                "graph KV geometry or CUDA context mismatch"
            );
            let stream = self.embedding.context().new_stream()?;
            let (data, warm, mut scratch, _conv, _recurrent) = with_context(|exec| {
                value((|| -> Result<_> {
                    // SAFETY: private warmup state never aliases a live request's mutable storage.
                    unsafe {
                        let mut metadata = vec![0u64; 2 + self.layers.len() * 2];
                        let delta_layers = self
                            .layers
                            .iter()
                            .enumerate()
                            .filter(|(_, layer)| matches!(layer.mixer, Mixer::Delta(_)));
                        let count = delta_layers.clone().count();
                        let mut conv = Vec::with_capacity(count);
                        let mut recurrent = Vec::with_capacity(count);
                        for (index, _) in delta_layers {
                            let cv = api::zeros::<T>(&[
                                CONV_HISTORY_LEN,
                                c.linear_num_key_heads * GDN_QK_CHANNELS_PER_HEAD
                                    + c.linear_num_value_heads * GDN_VALUE_HEAD_DIM,
                            ])
                            .execute(exec)?;
                            let st = api::zeros::<f32>(&[
                                c.linear_num_value_heads * GDN_KEY_HEAD_DIM,
                                GDN_VALUE_HEAD_DIM,
                            ])
                            .execute(exec)?;
                            metadata[2 + index * 2] = cv.device_pointer().cu_deviceptr();
                            metadata[3 + index * 2] = st.device_pointer().cu_deviceptr();
                            ensure!(
                                metadata[2 + index * 2] % 16 == 0
                                    && metadata[3 + index * 2] % 16 == 0,
                                "graph warmup state requires 16-byte alignment"
                            );
                            conv.push(cv);
                            recurrent.push(st);
                        }
                        let shape: Vec<usize> = cache.layers[0]
                            .0
                            .shape()
                            .iter()
                            .map(|&n| n as usize)
                            .collect();
                        let k = Arc::new(api::zeros::<T>(&shape).execute(exec)?);
                        let v = Arc::new(api::zeros::<T>(&shape).execute(exec)?);
                        let table_shape: Vec<usize> =
                            cache.table.shape().iter().map(|&n| n as usize).collect();
                        let warm = GraphData {
                            ids: api::copy_host_vec_to_device(&Arc::new(vec![1u32]))
                                .execute(exec)?,
                            metadata: api::copy_host_vec_to_device(&Arc::new(metadata))
                                .execute(exec)?,
                            cache: GraphCacheStorage {
                                layers: (0..cache.layers.len())
                                    .map(|_| (k.clone(), v.clone()))
                                    .collect(),
                                table: Arc::new(api::zeros::<i32>(&table_shape).execute(exec)?),
                            },
                        };
                        let data = GraphData {
                            ids: api::zeros::<u32>(&[1]).execute(exec)?,
                            metadata: api::zeros::<u64>(&[2 + self.layers.len() * 2])
                                .execute(exec)?,
                            cache,
                        };
                        Ok((
                            data,
                            warm,
                            self.workspaces.acquire(1, exec)?,
                            conv,
                            recurrent,
                        ))
                    }
                })())
            })
            .sync_on(&stream)??;
            with_context(|exec| {
                value(unsafe {
                    self.compute(&warm.ids, &mut scratch, Execution::Graph(&warm), exec)
                })
            })
            .sync_on(&stream)??;
            // Keep the warmed checkout exclusive through capture, including concurrent forwards.
            let mut graph = cuda_async::cuda_graph::CudaGraph::capture(
                stream.clone(),
                with_context(|exec| {
                    value(unsafe {
                        self.compute(&data.ids, &mut scratch, Execution::Graph(&data), exec)
                    })
                }),
            )?;
            let (logits, token) = graph.take_output().expect("captured output")?;
            // SAFETY: an exclusive graph checkout protects these outputs through readback.
            let (logits, token) =
                unsafe { (logits.into_shared_alias(), token.into_shared_alias()) };
            let context = self.embedding.context();
            let host_metadata =
                cutile::cuda_core::PinnedHostBuffer::zeroed(context, data.metadata.size())?;
            let host_id = cutile::cuda_core::PinnedHostBuffer::zeroed(context, 1)?;
            Ok(DecodeGraph {
                key,
                data,
                graph,
                logits,
                token,
                host_metadata,
                host_id,
            })
        })?))
    }

    pub(super) unsafe fn submit_pooled(
        &self,
        suffix: &Arc<Vec<u32>>,
        view: &mut Qwen3_5ExecutionView<'_, T>,
        graph: Option<&Arc<DecodeGraph<T>>>,
        exec: &ExecutionContext,
    ) -> Result<(Readback<f32>, Readback<i32>)> {
        if let Some(graph) = graph {
            let _range = ScopedRange::new(c"qwen3_5.decode.graph_replay");
            let metadata = view.graph_metadata()?;
            graph.replay(suffix[0], &metadata, exec)?;
            self.graphs.as_ref().unwrap().replayed();
            self.enqueue_readback(graph.logits.clone(), graph.token.clone(), exec)
        } else {
            self.submit(suffix, view, exec)
        }
    }

    unsafe fn submit(
        &self,
        suffix: &Arc<Vec<u32>>,
        state: &mut Qwen3_5ExecutionView<'_, T>,
        exec: &ExecutionContext,
    ) -> Result<(Readback<f32>, Readback<i32>)> {
        let mut scratch = self.workspaces.acquire(suffix.len(), exec)?;
        let ids = {
            let _range = ScopedRange::new(c"qwen3_5.input.upload");
            self.uploads.upload(suffix.clone(), &[suffix.len()], exec)?
        };
        let (logits, token) = self.compute(&ids, &mut scratch, Execution::Eager(state), exec)?;
        self.enqueue_readback(logits, token, exec)
    }

    pub(super) unsafe fn compute(
        &self,
        ids: &Tensor<u32>,
        scratch: &mut Scratch<T>,
        state: Execution<'_, '_, T>,
        exec: &ExecutionContext,
    ) -> Result<ComputeOutput> {
        scratch.retain(exec)?;
        scratch.reset();
        let (mut eager, graph) = match state {
            Execution::Eager(view) => (Some(view), None),
            Execution::Graph(data) => (None, Some(data)),
        };
        let c = &self.config.text_config;
        let start_pos = eager.as_ref().map_or(0, |view| view.start_pos());
        let rows = ids.shape()[0] as usize;
        let _submit = ScopedRange::new(if graph.is_none() && (start_pos == 0 || rows > 1) {
            c"qwen3_5.prefill.submit"
        } else {
            c"qwen3_5.decode.submit"
        });
        let h = c.hidden_size;
        let mut attention_index = 0;
        let mut x = scratch.activation(&[rows, h], exec)?;
        {
            let _range = ScopedRange::new(c"qwen3_5.embedding");
            shared::embedding(
                (&mut x).partition([1, SHARED_POINTWISE_COLUMNS]),
                ids,
                self.embedding.clone(),
            )
            .generics(vec![T::DTYPE.to_string()])
            .execute(exec)?;
        }
        let mut residual = scratch.activation(&[rows, h], exec)?;
        for (index, layer) in self.layers.iter().enumerate() {
            let _layer = ScopedRange::new(&self.layer_ranges[index]);
            scratch.layer(matches!(layer.mixer, Mixer::Delta(_)));
            let n = normalize(scratch, &*x, layer.input_norm.clone(), c.rms_norm_eps, exec)?;
            let mixed = match &layer.mixer {
                Mixer::Delta(w) => {
                    let _mixer = ScopedRange::new(c"qwen3_5.mixer.delta");
                    let delta = if let Some(view) = eager.as_deref_mut() {
                        Some(view.delta(index)?)
                    } else {
                        None
                    };
                    let qkv = linear(scratch, &*n, w.qkv.clone(), exec)?;
                    let mut convolved =
                        scratch.activation(&[rows, qkv.shape()[1] as usize], exec)?;

                    {
                        let _range = ScopedRange::new(c"qwen3_5.delta.conv");
                        let out = retained_pointer(&mut convolved, exec)?;
                        let generics = vec![T::DTYPE.to_string(), qkv.shape()[1].to_string()];
                        let grid = (1, (qkv.shape()[1] as usize / POINTWISE_COLUMNS) as u32, 1);
                        if let Some(data) = graph {
                            kernels::conv_graph(
                                out,
                                &data.metadata,
                                index as i32,
                                &*qkv,
                                w.conv.clone(),
                            )
                            .generics(generics)
                            .grid(grid)
                            .execute(exec)?;
                        } else {
                            kernels::conv(out, delta.as_ref().unwrap().conv, &*qkv, w.conv.clone())
                                .generics(generics)
                                .grid(grid)
                                .execute(exec)?;
                        }
                    }
                    let a = linear(scratch, &*n, w.a.clone(), exec)?;
                    let b = linear(scratch, &*n, w.b.clone(), exec)?;
                    let z = linear(scratch, &*n, w.z.clone(), exec)?;
                    let vh = c.linear_num_value_heads;
                    let mut core = scratch.activation(&[rows, vh * GDN_VALUE_HEAD_DIM], exec)?;

                    {
                        let _range = ScopedRange::new(c"qwen3_5.delta.recurrence");
                        let out = retained_pointer(&mut core, exec)?;
                        let generics = vec![
                            T::DTYPE.to_string(),
                            c.linear_num_key_heads.to_string(),
                            vh.to_string(),
                            GDN_KEY_HEAD_DIM.to_string(),
                            GDN_VALUE_HEAD_DIM.to_string(),
                        ];
                        let grid = (
                            vh as u32,
                            (GDN_VALUE_HEAD_DIM / DELTA_VALUE_COLUMNS) as u32,
                            1,
                        );
                        if let Some(data) = graph {
                            kernels::delta_graph(
                                out,
                                &data.metadata,
                                index as i32,
                                &*convolved,
                                &*a,
                                &*b,
                                w.a_log.clone(),
                                w.dt.clone(),
                            )
                            .generics(generics)
                            .grid(grid)
                            .execute(exec)?;
                        } else {
                            kernels::delta(
                                out,
                                delta.as_ref().unwrap().recurrent,
                                &*convolved,
                                &*a,
                                &*b,
                                w.a_log.clone(),
                                w.dt.clone(),
                            )
                            .generics(generics)
                            .grid(grid)
                            .execute(exec)?;
                        }
                    }
                    let core = core.reshape(&[rows * vh, GDN_VALUE_HEAD_DIM])?;
                    let z = z.reshape(&[rows * vh, GDN_VALUE_HEAD_DIM])?;
                    let mut gated = scratch.activation(&[rows * vh, GDN_VALUE_HEAD_DIM], exec)?;

                    {
                        let _range = ScopedRange::new(c"qwen3_5.delta.gated_norm");
                        kernels::gated_norm(
                            (&mut gated).partition([1, GDN_VALUE_HEAD_DIM]),
                            &*core,
                            &*z,
                            w.norm.clone(),
                            c.rms_norm_eps,
                        )
                        .generics(vec![T::DTYPE.to_string(), GDN_VALUE_HEAD_DIM.to_string()])
                        .execute(exec)?;
                    }
                    linear(
                        scratch,
                        &*gated.reshape(&[rows, vh * GDN_VALUE_HEAD_DIM])?,
                        w.out.clone(),
                        exec,
                    )?
                }
                Mixer::Attention(w) => {
                    let _mixer = ScopedRange::new(c"qwen3_5.mixer.gqa");
                    let qg = linear(scratch, &*n, w.q.clone(), exec)?;
                    let raw_k = linear(scratch, &*n, w.k.clone(), exec)?;
                    let raw_v = linear(scratch, &*n, w.v.clone(), exec)?;
                    let mut q = scratch
                        .activation(&[rows, c.num_attention_heads * ATTENTION_HEAD_DIM], exec)?;

                    {
                        let _range = ScopedRange::new(c"qwen3_5.gqa.q_rope");
                        let out = retained_pointer(&mut q, exec)?;
                        let generics = vec![
                            T::DTYPE.to_string(),
                            ATTENTION_HEAD_DIM.to_string(),
                            ROTARY_DIM.to_string(),
                            "true".into(),
                        ];
                        let grid = (rows as u32, c.num_attention_heads as u32, 1);
                        if let Some(data) = graph {
                            kernels::qk_graph(
                                out,
                                &*qg,
                                w.q_norm.clone(),
                                self.frequencies.clone(),
                                c.rms_norm_eps,
                                &data.metadata,
                                c.num_attention_heads as i32,
                            )
                            .generics(generics)
                            .grid(grid)
                            .execute(exec)?;
                        } else {
                            kernels::qk(
                                out,
                                &*qg,
                                w.q_norm.clone(),
                                self.frequencies.clone(),
                                c.rms_norm_eps,
                                start_pos as i32,
                                c.num_attention_heads as i32,
                                0,
                            )
                            .generics(generics)
                            .grid(grid)
                            .execute(exec)?;
                        }
                    }
                    let mut rotated_k = scratch
                        .activation(&[rows, c.num_key_value_heads * ATTENTION_HEAD_DIM], exec)?;
                    {
                        let _range = ScopedRange::new(c"qwen3_5.gqa.k_rope");
                        let out = retained_pointer(&mut rotated_k, exec)?;
                        let generics = vec![
                            T::DTYPE.to_string(),
                            ATTENTION_HEAD_DIM.to_string(),
                            ROTARY_DIM.to_string(),
                            "false".into(),
                        ];
                        let grid = (rows as u32, c.num_key_value_heads as u32, 1);
                        if let Some(data) = graph {
                            kernels::qk_graph(
                                out,
                                &*raw_k,
                                w.k_norm.clone(),
                                self.frequencies.clone(),
                                c.rms_norm_eps,
                                &data.metadata,
                                c.num_key_value_heads as i32,
                            )
                            .generics(generics)
                            .grid(grid)
                            .execute(exec)?;
                        } else {
                            kernels::qk(
                                out,
                                &*raw_k,
                                w.k_norm.clone(),
                                self.frequencies.clone(),
                                c.rms_norm_eps,
                                start_pos as i32,
                                c.num_key_value_heads as i32,
                                0,
                            )
                            .generics(generics)
                            .grid(grid)
                            .execute(exec)?;
                        }
                    }

                    let mut attended = scratch
                        .activation(&[rows, c.num_attention_heads * ATTENTION_HEAD_DIM], exec)?;
                    let generics = vec![
                        T::DTYPE.to_string(),
                        ATTENTION_HEAD_DIM.to_string(),
                        c.num_attention_heads.to_string(),
                        c.num_key_value_heads.to_string(),
                    ];
                    let cache = {
                        let _range = ScopedRange::new(c"qwen3_5.gqa.kv_store");
                        if let Some(data) = graph {
                            let (k, v) = &data.cache.layers[attention_index];
                            kernels::cache_decode(
                                k.device_pointer(),
                                v.device_pointer(),
                                &*rotated_k,
                                &*raw_v,
                                &data.cache.table,
                                &data.metadata,
                            )
                            .generics(vec![T::DTYPE.to_string(), ATTENTION_HEAD_DIM.to_string()])
                            .grid((1, c.num_key_value_heads as u32, 1))
                            .execute(exec)?;
                            AttentionView {
                                k: k.clone(),
                                v: v.clone(),
                                table: data.cache.table.device_pointer(),
                                table_offset: 0,
                                pooled: true,
                            }
                        } else {
                            eager
                                .as_deref_mut()
                                .unwrap()
                                .attention(index, &*rotated_k, &*raw_v)?
                        }
                    };
                    {
                        let _range = ScopedRange::new(c"qwen3_5.gqa.attention");
                        if let Some(data) = graph {
                            kernels::gqa_graph(
                                (&mut attended).partition([1, ATTENTION_HEAD_DIM]),
                                &*q,
                                cache.k,
                                cache.v,
                                &*qg,
                                &data.metadata,
                                cache.table,
                            )
                            .generics(generics)
                            .execute(exec)?;
                        } else {
                            kernels::gqa_attention(
                                (&mut attended).partition([1, ATTENTION_HEAD_DIM]),
                                &*q,
                                cache.k,
                                cache.v,
                                &*qg,
                                start_pos as i32,
                                cache.table,
                                cache.table_offset as i32,
                            )
                            .generics(
                                generics
                                    .into_iter()
                                    .chain([cache.pooled.to_string()])
                                    .collect(),
                            )
                            .execute(exec)?;
                        }
                        attention_index += 1;
                    }
                    linear(scratch, &*attended, w.out.clone(), exec)?
                }
            };
            add(&mut residual, &*x, &*mixed, exec)?;
            let _moe = ScopedRange::new(c"qwen3_5.moe");
            let n = normalize(
                scratch,
                &*residual,
                layer.post_norm.clone(),
                c.rms_norm_eps,
                exec,
            )?;
            let logits = {
                let _range = ScopedRange::new(c"qwen3_5.moe.router");
                linear(scratch, &*n, layer.router.clone(), exec)?
            };
            let top = c.num_experts_per_tok;
            let i = c.moe_intermediate_size;
            let mut selected = scratch.integer(&[rows, top], exec)?;
            let mut scores = scratch.activation(&[rows, top], exec)?;

            {
                let _range = ScopedRange::new(c"qwen3_5.moe.route");
                kernels::route(
                    (&mut selected).partition([1, top]),
                    (&mut scores).partition([1, top]),
                    &*logits,
                )
                .generics(vec![
                    T::DTYPE.to_string(),
                    c.num_experts.to_string(),
                    top.to_string(),
                ])
                .grid((rows as u32, 1, 1))
                .execute(exec)?;
            }
            let mut gu = scratch.activation(&[rows * top, 2 * i], exec)?;

            {
                let _range = ScopedRange::new(c"qwen3_5.moe.expert.gate_up");
                expert_project(&mut gu, &n, layer.gate_up.clone(), &selected, top, exec)?;
            }
            let mut activated = scratch.activation(&[rows * top, i], exec)?;

            {
                let _range = ScopedRange::new(c"qwen3_5.moe.expert.swiglu");
                kernels::expert_swiglu((&mut activated).partition([1, POINTWISE_COLUMNS]), &*gu)
                    .generics(vec![T::DTYPE.to_string(), i.to_string()])
                    .execute(exec)?;
            }
            let mut routed = scratch.activation(&[rows * top, h], exec)?;

            {
                let _range = ScopedRange::new(c"qwen3_5.moe.expert.down");
                expert_project(
                    &mut routed,
                    &activated,
                    layer.down.clone(),
                    &selected,
                    1,
                    exec,
                )?;
            }
            let (shared_out, shared_gate) = {
                let _range = ScopedRange::new(c"qwen3_5.moe.shared");
                let mut shared_act =
                    scratch.activation(&[rows, c.shared_expert_intermediate_size], exec)?;
                if T::DTYPE == bf16::DTYPE {
                    let _range = ScopedRange::new(c"qwen3_5.moe.shared.gate_up_swiglu");
                    kernels::shared_gate_up_vector(
                        (&mut shared_act).partition([1, 4]),
                        &*n,
                        layer.shared_gate.clone(),
                        layer.shared_up.clone(),
                    )
                    .generics(vec![T::DTYPE.to_string(), h.to_string()])
                    .execute(exec)?;
                } else {
                    let gate = linear(scratch, &n, layer.shared_gate.clone(), exec)?;
                    let up = linear(scratch, &n, layer.shared_up.clone(), exec)?;
                    let _range = ScopedRange::new(c"qwen3_5.moe.shared.swiglu");
                    shared::swiglu(
                        (&mut shared_act).partition([1, SHARED_POINTWISE_COLUMNS]),
                        &*gate,
                        &*up,
                    )
                    .generics(vec![T::DTYPE.to_string()])
                    .execute(exec)?;
                }
                let shared_out = linear(scratch, &*shared_act, layer.shared_down.clone(), exec)?;
                let shared_gate = linear(scratch, &*n, layer.shared_router.clone(), exec)?;
                (shared_out, shared_gate)
            };
            let mut merged = scratch.activation(&[rows, h], exec)?;

            {
                let _range = ScopedRange::new(c"qwen3_5.moe.merge");
                kernels::moe_merge(
                    (&mut merged).partition([1, POINTWISE_COLUMNS]),
                    &*routed,
                    &*scores,
                    &*shared_out,
                    &*shared_gate,
                )
                .generics(vec![T::DTYPE.to_string(), top.to_string()])
                .execute(exec)?;
            }
            add(&mut x, &*residual, &*merged, exec)?;
        }
        let _head = ScopedRange::new(c"qwen3_5.head");
        scratch.head();
        let mut last = scratch.activation(&[1, h], exec)?;

        kernels::last_row((&mut last).partition([1, POINTWISE_COLUMNS]), &*x)
            .generics(vec![T::DTYPE.to_string()])
            .execute(exec)?;
        let n = normalize(scratch, &*last, self.norm.clone(), c.rms_norm_eps, exec)?;
        let logits = linear_f32(scratch, &*n, self.lm_head.clone(), exec)?;
        let blocks = c.vocab_size.div_ceil(shared::ARGMAX_TILE);
        let mut maxima = scratch.float(&[1, blocks], exec)?;
        let mut indices = scratch.integer(&[1, blocks], exec)?;
        let mut token = scratch.integer(&[1], exec)?;
        {
            let _range = ScopedRange::new(c"qwen3_5.argmax.blocks");
            shared::argmax_blocks(
                (&mut maxima).partition([1, 1]),
                (&mut indices).partition([1, 1]),
                &*logits,
                c.vocab_size as i32,
            )
            .generics(vec![shared::ARGMAX_TILE.to_string()])
            .execute(exec)?;
        }
        {
            let _range = ScopedRange::new(c"qwen3_5.argmax.final");
            shared::argmax_final(
                (&mut token).partition([1]),
                &*maxima,
                &*indices,
                blocks as i32,
            )
            .generics(vec![shared::ARGMAX_FINAL_TILE.to_string()])
            .execute(exec)?;
        }
        Ok((logits.escape(), token.escape()))
    }

    unsafe fn enqueue_readback(
        &self,
        logits: impl std::borrow::Borrow<Tensor<f32>> + Send + 'static,
        token: impl std::borrow::Borrow<Tensor<i32>> + Send + 'static,
        exec: &ExecutionContext,
    ) -> Result<(Readback<f32>, Readback<i32>)> {
        let _range = ScopedRange::new(c"qwen3_5.readback.enqueue");
        Ok((
            self.readbacks.enqueue(logits, exec)?,
            self.token_readbacks.enqueue(token, exec)?,
        ))
    }
}

unsafe fn expert_project<T: DType>(
    out: &mut Tensor<T>,
    x: &Tensor<T>,
    weight: Weight<T>,
    selected: &Tensor<i32>,
    input_repeat: usize,
    exec: &ExecutionContext,
) -> Result<()> {
    let generics = vec![
        T::DTYPE.to_string(),
        x.shape()[1].to_string(),
        out.shape()[1].to_string(),
        selected.shape()[1].to_string(),
    ];
    if T::DTYPE == bf16::DTYPE {
        kernels::expert_mma(
            out.partition([1, EXPERT_OUTPUT_COLUMNS]),
            x,
            weight,
            selected,
            input_repeat as i32,
        )
        .generics(generics)
        .execute(exec)?;
    } else {
        kernels::expert(
            out.partition([1, EXPERT_OUTPUT_COLUMNS]),
            x,
            weight,
            selected,
            input_repeat as i32,
        )
        .generics(generics)
        .execute(exec)?;
    }
    Ok(())
}

unsafe fn normalize<T: DType + DeviceCopy>(
    scratch: &mut Scratch<T>,
    x: &Tensor<T>,
    w: Norm,
    eps: f32,
    exec: &ExecutionContext,
) -> Result<ScratchTensor<T, T>> {
    let _range = ScopedRange::new(c"qwen3_5.rms_norm");
    let rows = x.shape()[0] as usize;
    let d = x.shape()[1] as usize;
    let mut out = scratch.activation(&[rows, d], exec)?;
    kernels::norm((&mut out).partition([1, d]), x, w, eps)
        .generics(vec![T::DTYPE.to_string(), d.to_string(), "true".into()])
        .execute(exec)?;
    Ok(out)
}
unsafe fn linear<T: DType + DeviceCopy>(
    scratch: &mut Scratch<T>,
    x: &Tensor<T>,
    w: Weight<T>,
    exec: &ExecutionContext,
) -> Result<ScratchTensor<T, T>> {
    let out = scratch.activation(&[x.shape()[0] as usize, w.shape()[0] as usize], exec)?;
    let bias = scratch.activation(&[1], exec)?;
    project(out, x, w, &bias, exec)
}
unsafe fn linear_f32<T: DType + DeviceCopy>(
    scratch: &mut Scratch<T>,
    x: &Tensor<T>,
    w: Weight<T>,
    exec: &ExecutionContext,
) -> Result<ScratchTensor<T, f32>> {
    let out = scratch.float(&[x.shape()[0] as usize, w.shape()[0] as usize], exec)?;
    let bias = scratch.activation(&[1], exec)?;
    project(out, x, w, &bias, exec)
}
unsafe fn project<T: DType, O: DType>(
    mut out: ScratchTensor<T, O>,
    x: &Tensor<T>,
    w: Weight<T>,
    bias: &Tensor<T>,
    exec: &ExecutionContext,
) -> Result<ScratchTensor<T, O>> {
    let _range = ScopedRange::new(c"qwen3_5.linear");
    // Preserve GDN a/b MMA rounding; vectorize scalar and 256-way router shapes.
    // Keep the reduction tree independent of token count for chunked prefill.
    if T::DTYPE == bf16::DTYPE && O::DTYPE == bf16::DTYPE && matches!(w.shape()[0], 1 | 256) {
        kernels::linear_vector((&mut out).partition([1, 4]), x, w)
            .generics(vec![
                T::DTYPE.to_string(),
                O::DTYPE.to_string(),
                x.shape()[1].to_string(),
            ])
            .execute(exec)?;
    } else {
        shared::linear(
            (&mut out).partition([LINEAR_TILE_ROWS, LINEAR_TILE_COLUMNS]),
            x,
            w,
            bias,
        )
        .generics(vec![
            T::DTYPE.to_string(),
            O::DTYPE.to_string(),
            x.shape()[1].to_string(),
            "false".into(),
            LINEAR_TILE_ROWS.to_string(),
            LINEAR_TILE_COLUMNS.to_string(),
            LINEAR_TILE_REDUCTION.to_string(),
        ])
        .execute(exec)?;
    }
    Ok(out)
}
unsafe fn add<T: DType>(
    out: &mut Tensor<T>,
    a: &Tensor<T>,
    b: &Tensor<T>,
    exec: &ExecutionContext,
) -> Result<()> {
    let _range = ScopedRange::new(c"qwen3_5.residual.add");
    shared::add(out.partition([1, SHARED_POINTWISE_COLUMNS]), a, b)
        .generics(vec![T::DTYPE.to_string(), "1".into()])
        .execute(exec)?;
    Ok(())
}

fn validate_prediction(token: i32, logits: &[f32]) -> Result<u32> {
    let _range = ScopedRange::new(c"qwen3_5.host_validation");
    ensure!(logits.iter().all(|v| v.is_finite()), "non-finite logits");
    ensure!(
        token >= 0 && (token as usize) < logits.len(),
        "invalid greedy prediction"
    );
    Ok(token as u32)
}

impl<T: WeightElement + ElementType> CausalModel<Qwen3_5StatePool<T>> for Qwen3_5Engine<T> {
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    fn is_eos(&self, token: u32) -> bool {
        self.is_eos(token)
    }
    fn validate_input(&self, tokens: &[u32]) -> Result<(), ModelError> {
        if tokens
            .iter()
            .any(|&id| id as usize >= self.config.text_config.vocab_size)
        {
            return Err(ModelError::InvalidInput("token outside vocabulary".into()));
        }
        Ok(())
    }
    async fn forward(
        &self,
        batch: &mut [ModelInput<'_, Qwen3_5StatePool<T>>],
        policy: Arc<dyn SchedulingPolicy>,
    ) -> Vec<Result<NextToken, ModelError>> {
        let mut outputs = Vec::with_capacity(batch.len());
        // shortcut: serial sessions and host token feedback, use packed kernels and
        // submission-first GPU predictions when optimizing Runner throughput.
        for input in batch {
            let mut submitted = false;
            let result: Result<NextToken, ModelError> = async {
                if !input.kv.has_owned_pool_state() {
                    return Err(ModelError::InvalidInput(
                        "context has been invalidated".into(),
                    ));
                }
                let tokens = input.token_ids.host()?;
                self.validate_input(tokens)?;
                if input.kv.is_pending()
                    || tokens.len() <= input.kv.ready_len()
                    || tokens.len() > input.kv.max_seq_len()
                {
                    return Err(ModelError::InvalidInput(
                        "expected an idle context and extending history".into(),
                    ));
                }
                input
                    .kv
                    .state()
                    .validate(&self.owner, tokens, input.kv.ready_len())?;
                let start_pos = input.kv.ready_len();
                let suffix = Arc::new(tokens[start_pos..].to_vec());
                let graph = self.prepare_decode_graph(input)?;
                let (logits, token) =
                    Self::execute_append_batch(
                        std::slice::from_mut(input),
                        &policy,
                        |batch, exec| {
                            submitted = true;
                            let mut view = batch[0]
                                .kv
                                .execution_view(batch[0].token_ids.host()?, exec)?;
                            // SAFETY: view grants this append's resources and retains GPU owners.
                            Ok(unsafe {
                                self.submit_pooled(&suffix, &mut view, graph.as_ref(), exec)?
                            })
                        },
                    )
                    .await?;
                let next = validate_prediction(token.as_slice()[0], logits.as_slice())?;
                Ok(NextToken::Host(next))
            }
            .await;
            if result.is_err()
                && input.kv.has_owned_pool_state()
                && (input.kv.is_pending() || submitted)
            {
                input.kv.invalidate();
            }
            outputs.push(result);
        }
        outputs
    }
}
