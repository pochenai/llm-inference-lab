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

use std::{path::Path, sync::Arc};

use anyhow::{ensure, Result};
use cuda_async::{device_operation::*, scheduling_policies::SchedulingPolicy};
use cutile::{
    api,
    core::{bf16, ElementType},
    tensor::{PartitionMut, Reshape, Tensor},
    tile_kernel::TileKernel,
    DType,
};
use tokenizers::Tokenizer;

use super::{
    config::{Qwen3_5Config, Qwen3_5LayerType},
    constants::*,
    kernels,
    state::{retained_pointer, Qwen3_5Context, Qwen3_5ExecutionView},
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
    norm: Norm,
    lm_head: Weight<T>,
    frequencies: Norm,
    eos: Vec<u32>,
    policy: Arc<dyn SchedulingPolicy>,
    pub(super) owner: Arc<()>,
    weight_bytes: usize,
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
        Ok(Self {
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
            readbacks: Default::default(),
            token_readbacks: Default::default(),
        })
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

    unsafe fn submit(
        &self,
        suffix: &Arc<Vec<u32>>,
        state: &mut Qwen3_5ExecutionView<'_, T>,
        exec: &ExecutionContext,
    ) -> Result<(Readback<f32>, Readback<i32>)> {
        let c = &self.config.text_config;
        let start_pos = state.start_pos();
        let rows = suffix.len();
        let h = c.hidden_size;
        let ids = api::copy_host_vec_to_device(suffix).execute(exec)?;
        let mut x = api::zeros::<T>(&[rows, h]).execute(exec)?;
        shared::embedding(
            (&mut x).partition([1, SHARED_POINTWISE_COLUMNS]),
            &ids,
            self.embedding.clone(),
        )
        .generics(vec![T::DTYPE.to_string()])
        .execute(exec)?;
        for (index, layer) in self.layers.iter().enumerate() {
            let n = normalize(&x, layer.input_norm.clone(), c.rms_norm_eps, exec)?;
            let mixed = match &layer.mixer {
                Mixer::Delta(w) => {
                    let delta = state.delta(index)?;
                    let qkv = linear(&n, w.qkv.clone(), exec)?;
                    let mut convolved =
                        api::zeros::<T>(&[rows, qkv.shape()[1] as usize]).execute(exec)?;

                    kernels::conv(
                        retained_pointer(&mut convolved, exec)?,
                        delta.conv,
                        &qkv,
                        w.conv.clone(),
                    )
                    .generics(vec![T::DTYPE.to_string(), qkv.shape()[1].to_string()])
                    .grid((1, (qkv.shape()[1] as usize / POINTWISE_COLUMNS) as u32, 1))
                    .execute(exec)?;
                    let a = linear(&n, w.a.clone(), exec)?;
                    let b = linear(&n, w.b.clone(), exec)?;
                    let z = linear(&n, w.z.clone(), exec)?;
                    let vh = c.linear_num_value_heads;
                    let mut core =
                        api::zeros::<T>(&[rows, vh * GDN_VALUE_HEAD_DIM]).execute(exec)?;

                    kernels::delta(
                        retained_pointer(&mut core, exec)?,
                        delta.recurrent,
                        &convolved,
                        &a,
                        &b,
                        w.a_log.clone(),
                        w.dt.clone(),
                    )
                    .generics(vec![
                        T::DTYPE.to_string(),
                        c.linear_num_key_heads.to_string(),
                        vh.to_string(),
                        GDN_KEY_HEAD_DIM.to_string(),
                        GDN_VALUE_HEAD_DIM.to_string(),
                    ])
                    .grid((
                        vh as u32,
                        (GDN_VALUE_HEAD_DIM / DELTA_VALUE_COLUMNS) as u32,
                        1,
                    ))
                    .execute(exec)?;
                    let core = core.reshape(&[rows * vh, GDN_VALUE_HEAD_DIM])?;
                    let z = z.reshape(&[rows * vh, GDN_VALUE_HEAD_DIM])?;
                    let mut gated =
                        api::zeros::<T>(&[rows * vh, GDN_VALUE_HEAD_DIM]).execute(exec)?;

                    kernels::gated_norm(
                        (&mut gated).partition([1, GDN_VALUE_HEAD_DIM]),
                        &core,
                        &z,
                        w.norm.clone(),
                        c.rms_norm_eps,
                    )
                    .generics(vec![T::DTYPE.to_string(), GDN_VALUE_HEAD_DIM.to_string()])
                    .execute(exec)?;
                    linear(
                        &gated.reshape(&[rows, vh * GDN_VALUE_HEAD_DIM])?,
                        w.out.clone(),
                        exec,
                    )?
                }
                Mixer::Attention(w) => {
                    let qg = linear(&n, w.q.clone(), exec)?;
                    let raw_k = linear(&n, w.k.clone(), exec)?;
                    let raw_v = linear(&n, w.v.clone(), exec)?;
                    let mut q =
                        api::zeros::<T>(&[rows, c.num_attention_heads * ATTENTION_HEAD_DIM])
                            .execute(exec)?;

                    kernels::qk(
                        retained_pointer(&mut q, exec)?,
                        &qg,
                        w.q_norm.clone(),
                        self.frequencies.clone(),
                        c.rms_norm_eps,
                        start_pos as i32,
                        c.num_attention_heads as i32,
                        0,
                    )
                    .generics(vec![
                        T::DTYPE.to_string(),
                        ATTENTION_HEAD_DIM.to_string(),
                        ROTARY_DIM.to_string(),
                        "true".into(),
                    ])
                    .grid((rows as u32, c.num_attention_heads as u32, 1))
                    .execute(exec)?;

                    let mut rotated_k =
                        api::zeros::<T>(&[rows, c.num_key_value_heads * ATTENTION_HEAD_DIM])
                            .execute(exec)?;
                    kernels::qk(
                        retained_pointer(&mut rotated_k, exec)?,
                        &raw_k,
                        w.k_norm.clone(),
                        self.frequencies.clone(),
                        c.rms_norm_eps,
                        start_pos as i32,
                        c.num_key_value_heads as i32,
                        0,
                    )
                    .generics(vec![
                        T::DTYPE.to_string(),
                        ATTENTION_HEAD_DIM.to_string(),
                        ROTARY_DIM.to_string(),
                        "false".into(),
                    ])
                    .grid((rows as u32, c.num_key_value_heads as u32, 1))
                    .execute(exec)?;

                    let mut attended =
                        api::zeros::<T>(&[rows, c.num_attention_heads * ATTENTION_HEAD_DIM])
                            .execute(exec)?;
                    let generics = vec![
                        T::DTYPE.to_string(),
                        ATTENTION_HEAD_DIM.to_string(),
                        c.num_attention_heads.to_string(),
                        c.num_key_value_heads.to_string(),
                    ];
                    let cache = state.attention(index, &rotated_k, &raw_v)?;
                    kernels::gqa_attention(
                        (&mut attended).partition([1, ATTENTION_HEAD_DIM]),
                        &q,
                        cache.k,
                        cache.v,
                        &qg,
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
                    linear(&attended, w.out.clone(), exec)?
                }
            };
            x = add(&x, &mixed, exec)?;
            let n = normalize(&x, layer.post_norm.clone(), c.rms_norm_eps, exec)?;
            let logits = linear(&n, layer.router.clone(), exec)?;
            let top = c.num_experts_per_tok;
            let i = c.moe_intermediate_size;
            let mut selected = api::zeros::<i32>(&[rows, top]).execute(exec)?;
            let mut scores = api::zeros::<T>(&[rows, top]).execute(exec)?;

            kernels::route(
                (&mut selected).partition([1, top]),
                (&mut scores).partition([1, top]),
                &logits,
            )
            .generics(vec![
                T::DTYPE.to_string(),
                c.num_experts.to_string(),
                top.to_string(),
            ])
            .grid((rows as u32, 1, 1))
            .execute(exec)?;
            let mut gu = api::zeros::<T>(&[rows * top, 2 * i]).execute(exec)?;

            kernels::expert(
                (&mut gu).partition([1, EXPERT_OUTPUT_COLUMNS]),
                &n,
                layer.gate_up.clone(),
                &selected,
                top as i32,
            )
            .generics(vec![
                T::DTYPE.to_string(),
                h.to_string(),
                (2 * i).to_string(),
                top.to_string(),
            ])
            .execute(exec)?;
            let mut activated = api::zeros::<T>(&[rows * top, i]).execute(exec)?;

            kernels::expert_swiglu((&mut activated).partition([1, POINTWISE_COLUMNS]), &gu)
                .generics(vec![T::DTYPE.to_string(), i.to_string()])
                .execute(exec)?;
            let mut routed = api::zeros::<T>(&[rows * top, h]).execute(exec)?;

            kernels::expert(
                (&mut routed).partition([1, EXPERT_OUTPUT_COLUMNS]),
                &activated,
                layer.down.clone(),
                &selected,
                1,
            )
            .generics(vec![
                T::DTYPE.to_string(),
                i.to_string(),
                h.to_string(),
                top.to_string(),
            ])
            .execute(exec)?;
            let gate = linear(&n, layer.shared_gate.clone(), exec)?;
            let up = linear(&n, layer.shared_up.clone(), exec)?;
            let mut shared_act =
                api::zeros::<T>(&[rows, c.shared_expert_intermediate_size]).execute(exec)?;
            shared::swiglu(
                (&mut shared_act).partition([1, SHARED_POINTWISE_COLUMNS]),
                &gate,
                &up,
            )
            .generics(vec![T::DTYPE.to_string()])
            .execute(exec)?;
            let shared_out = linear(&shared_act, layer.shared_down.clone(), exec)?;
            let shared_gate = linear(&n, layer.shared_router.clone(), exec)?;
            let mut merged = api::zeros::<T>(&[rows, h]).execute(exec)?;

            kernels::moe_merge(
                (&mut merged).partition([1, POINTWISE_COLUMNS]),
                &routed,
                &scores,
                &shared_out,
                &shared_gate,
            )
            .generics(vec![T::DTYPE.to_string(), top.to_string()])
            .execute(exec)?;
            x = add(&x, &merged, exec)?;
        }
        let mut last = api::zeros::<T>(&[1, h]).execute(exec)?;

        kernels::last_row((&mut last).partition([1, POINTWISE_COLUMNS]), &x)
            .generics(vec![T::DTYPE.to_string()])
            .execute(exec)?;
        let n = normalize(&last, self.norm.clone(), c.rms_norm_eps, exec)?;
        let logits = linear_f32(&n, self.lm_head.clone(), exec)?;
        let blocks = c.vocab_size.div_ceil(shared::ARGMAX_TILE);
        let mut maxima = api::zeros::<f32>(&[1, blocks]).execute(exec)?;
        let mut indices = api::zeros::<i32>(&[1, blocks]).execute(exec)?;
        let mut token = api::zeros::<i32>(&[1]).execute(exec)?;
        shared::argmax_blocks(
            (&mut maxima).partition([1, 1]),
            (&mut indices).partition([1, 1]),
            &logits,
            c.vocab_size as i32,
        )
        .generics(vec![shared::ARGMAX_TILE.to_string()])
        .execute(exec)?;
        shared::argmax_final(
            (&mut token).partition([1]),
            &maxima,
            &indices,
            blocks as i32,
        )
        .generics(vec![shared::ARGMAX_FINAL_TILE.to_string()])
        .execute(exec)?;
        Ok((
            self.readbacks.enqueue(logits, exec)?,
            self.token_readbacks.enqueue(Arc::new(token), exec)?,
        ))
    }
}

unsafe fn normalize<T: DType>(
    x: &Tensor<T>,
    w: Norm,
    eps: f32,
    exec: &ExecutionContext,
) -> Result<Tensor<T>> {
    let rows = x.shape()[0] as usize;
    let d = x.shape()[1] as usize;
    let mut out = api::zeros::<T>(&[rows, d]).execute(exec)?;

    kernels::norm((&mut out).partition([1, d]), x, w, eps)
        .generics(vec![T::DTYPE.to_string(), d.to_string(), "true".into()])
        .execute(exec)?;
    Ok(out)
}
unsafe fn linear<T: DType>(
    x: &Tensor<T>,
    w: Weight<T>,
    exec: &ExecutionContext,
) -> Result<Tensor<T>> {
    project(x, w, exec)
}
unsafe fn linear_f32<T: DType>(
    x: &Tensor<T>,
    w: Weight<T>,
    exec: &ExecutionContext,
) -> Result<Tensor<f32>> {
    project(x, w, exec)
}
unsafe fn project<T: DType, O: DType>(
    x: &Tensor<T>,
    w: Weight<T>,
    exec: &ExecutionContext,
) -> Result<Tensor<O>> {
    let mut out = api::zeros::<O>(&[x.shape()[0] as usize, w.shape()[0] as usize]).execute(exec)?;
    let bias = api::zeros::<T>(&[1]).execute(exec)?;
    shared::linear(
        (&mut out).partition([LINEAR_TILE_ROWS, LINEAR_TILE_COLUMNS]),
        x,
        w,
        &bias,
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
    Ok(out)
}
unsafe fn add<T: DType>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    exec: &ExecutionContext,
) -> Result<Tensor<T>> {
    let mut out = api::zeros::<T>(&[a.shape()[0] as usize, a.shape()[1] as usize]).execute(exec)?;
    shared::add((&mut out).partition([1, SHARED_POINTWISE_COLUMNS]), a, b)
        .generics(vec![T::DTYPE.to_string(), "1".into()])
        .execute(exec)?;
    Ok(out)
}

fn validate_prediction(token: i32, logits: &[f32]) -> Result<u32> {
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
                let (logits, token) = Self::execute_append_batch(
                    std::slice::from_mut(input),
                    &policy,
                    |batch, exec| {
                        submitted = true;
                        let mut view = batch[0]
                            .kv
                            .execution_view(batch[0].token_ids.host()?, exec)?;
                        // SAFETY: view grants this append's resources and retains GPU owners.
                        Ok(unsafe { self.submit(&suffix, &mut view, exec)? })
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
