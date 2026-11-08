//! BF16 text-only Qwen3.5 MoE decoder with explicit per-session hybrid state.
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

use anyhow::{ensure, Context, Result};
use cuda_async::{device_operation::*, scheduling_policies::SchedulingPolicy};
use cutile::{
    api,
    core::bf16,
    tensor::{PartitionMut, Reshape, Tensor},
    tile_kernel::TileKernel,
    DType,
};
use tokenizers::Tokenizer;

use super::{
    config::{Qwen3_5Config, Qwen3_5LayerType},
    constants::*,
    kernels,
};
use crate::runner::{
    loader::{EosTokenId, WeightElement, WeightLoader},
    model::qwen2::kernels as shared,
};

type Weight = Arc<Tensor<bf16>>;
type Norm = Arc<Tensor<f32>>;

struct Delta {
    qkv: Weight,
    z: Weight,
    a: Weight,
    b: Weight,
    out: Weight,
    conv: Weight,
    a_log: Norm,
    dt: Norm,
    norm: Norm,
}
struct Attention {
    q: Weight,
    k: Weight,
    v: Weight,
    out: Weight,
    q_norm: Norm,
    k_norm: Norm,
}
enum Mixer {
    Delta(Delta),
    Attention(Attention),
}
struct Layer {
    input_norm: Norm,
    post_norm: Norm,
    mixer: Mixer,
    router: Weight,
    gate_up: Weight,
    down: Weight,
    shared_gate: Weight,
    shared_up: Weight,
    shared_down: Weight,
    shared_router: Weight,
}
enum LayerState {
    Delta {
        conv: Tensor<bf16>,
        recurrent: Tensor<f32>,
    },
    Attention {
        k: Tensor<bf16>,
        v: Tensor<bf16>,
    },
}

/// Exclusive, append-only hybrid state. A failed/cancelled forward invalidates it.
/// Recurrent state cannot be sliced or shared as a KV prefix.
pub struct Qwen3_5Context {
    owner: Arc<()>,
    layers: Vec<LayerState>,
    history: Vec<u32>,
    /// Maximum total token history for this session, including prompt and generated tokens.
    max_seq_len: usize,
    valid: bool,
}
impl Qwen3_5Context {
    pub fn ready_len(&self) -> usize {
        self.history.len()
    }
    pub fn is_valid(&self) -> bool {
        self.valid
    }
}

/// BF16 text backbone with cuTile operators and explicit per-session hybrid state.
pub struct Qwen3_5Engine {
    config: Qwen3_5Config,
    tokenizer: Tokenizer,
    embedding: Weight,
    layers: Vec<Layer>,
    norm: Norm,
    lm_head: Weight,
    frequencies: Norm,
    eos: Vec<u32>,
    policy: Arc<dyn SchedulingPolicy>,
    owner: Arc<()>,
    weight_bytes: usize,
    readbacks: crate::runner::model::readback::ReadbackCache<f32>,
}

impl Qwen3_5Engine {
    pub async fn load(path: &Path, policy: &Arc<dyn SchedulingPolicy>) -> Result<Self> {
        let loader = WeightLoader::new(path)?;
        let config: Qwen3_5Config = loader.config()?;
        config.validate()?;
        let c = &config.text_config;
        let tokenizer = loader.tokenizer()?;
        let mut weight_bytes = 0;
        async fn weight<T: WeightElement>(
            loader: &WeightLoader,
            name: &str,
            shape: &[usize],
            policy: &Arc<dyn SchedulingPolicy>,
            bytes: &mut usize,
        ) -> Result<Arc<Tensor<T>>> {
            let host = loader.load_host::<T>(name)?;
            ensure!(
                host.shape == shape,
                "{name}: expected {shape:?}, got {:?}",
                host.shape
            );
            *bytes += host.data.len() * size_of::<T>();
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
                    let host =
                        loader.load_host::<bf16>(&format!("{p}.linear_attn.conv1d.weight"))?;
                    ensure!(
                        host.shape == [channels, 1, CONV_KERNEL_SIZE],
                        "invalid convolution weight shape"
                    );
                    let mut transposed: Vec<bf16> = Vec::with_capacity(host.data.len());
                    transposed.extend(
                        (0..CONV_KERNEL_SIZE)
                            .flat_map(|tap| (0..channels).map(move |channel| (tap, channel)))
                            .map(|(tap, channel)| host.data[channel * CONV_KERNEL_SIZE + tap]),
                    );
                    weight_bytes += host.data.len() * size_of::<bf16>();
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
            let gu: Weight = load!(
                format!("{p}.mlp.experts.gate_up_proj"),
                [c.num_experts, 2 * i, h]
            );
            let down: Weight = load!(format!("{p}.mlp.experts.down_proj"), [c.num_experts, h, i]);
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

    /// Allocate session state for at most max_seq_len total tokens (prompt plus generation).
    pub async fn new_context(&self, max_seq_len: usize) -> Result<Qwen3_5Context> {
        let c = &self.config.text_config;
        ensure!(
            max_seq_len > 0
                && max_seq_len <= c.max_position_embeddings
                && max_seq_len
                    .checked_mul(c.num_key_value_heads * ATTENTION_HEAD_DIM)
                    .is_some_and(|n| n <= i32::MAX as usize),
            "invalid maximum session sequence length"
        );
        let layers = with_context(|exec| {
            value((|| -> Result<_> {
                let mut states = Vec::with_capacity(self.layers.len());
                for layer in &self.layers {
                    // SAFETY: allocation operations are completed before returning the context.
                    states.push(unsafe {
                        match layer.mixer {
                            Mixer::Delta(_) => LayerState::Delta {
                                conv: api::zeros(&[
                                    CONV_HISTORY_LEN,
                                    c.linear_num_key_heads * GDN_QK_CHANNELS_PER_HEAD
                                        + c.linear_num_value_heads * GDN_VALUE_HEAD_DIM,
                                ])
                                .execute(exec)?,
                                recurrent: api::zeros(&[
                                    c.linear_num_value_heads * GDN_KEY_HEAD_DIM,
                                    GDN_VALUE_HEAD_DIM,
                                ])
                                .execute(exec)?,
                            },
                            Mixer::Attention(_) => LayerState::Attention {
                                k: api::zeros(&[
                                    max_seq_len,
                                    c.num_key_value_heads * ATTENTION_HEAD_DIM,
                                ])
                                .execute(exec)?,
                                v: api::zeros(&[
                                    max_seq_len,
                                    c.num_key_value_heads * ATTENTION_HEAD_DIM,
                                ])
                                .execute(exec)?,
                            },
                        }
                    });
                }
                Ok(states)
            })())
        })
        .schedule(&self.policy)?
        .await??;
        Ok(Qwen3_5Context {
            owner: self.owner.clone(),
            layers,
            history: Vec::with_capacity(max_seq_len),
            max_seq_len,
            valid: true,
        })
    }

    /// Append the uncached suffix of a complete token history and return next-token logits.
    /// All GPU updates complete before the committed history advances.
    pub async fn step_with_logits(
        &self,
        tokens: &[u32],
        state: &mut Qwen3_5Context,
    ) -> Result<(u32, Vec<f32>)> {
        ensure!(
            Arc::ptr_eq(&state.owner, &self.owner),
            "context belongs to another engine"
        );
        ensure!(
            state.valid,
            "context was invalidated by a cancelled or failed forward"
        );
        ensure!(
            tokens.len() > state.history.len()
                && tokens.len() <= state.max_seq_len
                && tokens.starts_with(&state.history),
            "expected an extending history within the maximum session sequence length"
        );
        ensure!(
            tokens
                .iter()
                .all(|id| (*id as usize) < self.config.text_config.vocab_size),
            "token outside vocabulary"
        );
        state.valid = false;
        let start_pos = state.history.len();
        let suffix = Arc::new(tokens[start_pos..].to_vec());
        let logits =
            with_context(|exec| value(unsafe { self.submit(&suffix, start_pos, state, exec) }))
                .schedule(&self.policy)?
                .await??;
        let logits = logits.as_slice().to_vec();
        ensure!(logits.iter().all(|v| v.is_finite()), "non-finite logits");
        let next = logits
            .iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
            .context("empty logits")?
            .0 as u32;
        state.history.extend_from_slice(&tokens[start_pos..]);
        state.valid = true;
        Ok((next, logits))
    }

    unsafe fn submit(
        &self,
        suffix: &Arc<Vec<u32>>,
        start_pos: usize,
        state: &mut Qwen3_5Context,
        exec: &ExecutionContext,
    ) -> Result<crate::runner::model::readback::Readback<f32>> {
        let c = &self.config.text_config;
        let rows = suffix.len();
        let h = c.hidden_size;
        let ids = api::copy_host_vec_to_device(suffix).execute(exec)?;
        let mut x = api::zeros::<bf16>(&[rows, h]).execute(exec)?;
        shared::embedding(
            (&mut x).partition([1, SHARED_POINTWISE_COLUMNS]),
            &ids,
            self.embedding.clone(),
        )
        .generics(vec![bf16::DTYPE.to_string()])
        .execute(exec)?;
        for (layer, state) in self.layers.iter().zip(&mut state.layers) {
            let n = normalize(&x, layer.input_norm.clone(), c.rms_norm_eps, exec)?;
            let mixed = match (&layer.mixer, state) {
                (Mixer::Delta(w), LayerState::Delta { conv, recurrent }) => {
                    let qkv = linear(&n, w.qkv.clone(), exec)?;
                    let mut convolved =
                        api::zeros::<bf16>(&[rows, qkv.shape()[1] as usize]).execute(exec)?;

                    kernels::conv(
                        output_pointer(&mut convolved, exec)?,
                        output_pointer(conv, exec)?,
                        &qkv,
                        w.conv.clone(),
                    )
                    .generics(vec![bf16::DTYPE.to_string(), qkv.shape()[1].to_string()])
                    .grid((1, (qkv.shape()[1] as usize / POINTWISE_COLUMNS) as u32, 1))
                    .execute(exec)?;
                    let a = linear(&n, w.a.clone(), exec)?;
                    let b = linear(&n, w.b.clone(), exec)?;
                    let z = linear(&n, w.z.clone(), exec)?;
                    let vh = c.linear_num_value_heads;
                    let mut core =
                        api::zeros::<bf16>(&[rows, vh * GDN_VALUE_HEAD_DIM]).execute(exec)?;

                    kernels::delta(
                        output_pointer(&mut core, exec)?,
                        output_pointer(recurrent, exec)?,
                        &convolved,
                        &a,
                        &b,
                        w.a_log.clone(),
                        w.dt.clone(),
                    )
                    .generics(vec![
                        bf16::DTYPE.to_string(),
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
                        api::zeros::<bf16>(&[rows * vh, GDN_VALUE_HEAD_DIM]).execute(exec)?;

                    kernels::gated_norm(
                        (&mut gated).partition([1, GDN_VALUE_HEAD_DIM]),
                        &core,
                        &z,
                        w.norm.clone(),
                        c.rms_norm_eps,
                    )
                    .generics(vec![GDN_VALUE_HEAD_DIM.to_string()])
                    .execute(exec)?;
                    linear(
                        &gated.reshape(&[rows, vh * GDN_VALUE_HEAD_DIM])?,
                        w.out.clone(),
                        exec,
                    )?
                }
                (Mixer::Attention(w), LayerState::Attention { k, v }) => {
                    let qg = linear(&n, w.q.clone(), exec)?;
                    let raw_k = linear(&n, w.k.clone(), exec)?;
                    let raw_v = linear(&n, w.v.clone(), exec)?;
                    let mut q =
                        api::zeros::<bf16>(&[rows, c.num_attention_heads * ATTENTION_HEAD_DIM])
                            .execute(exec)?;

                    kernels::qk(
                        output_pointer(&mut q, exec)?,
                        &qg,
                        w.q_norm.clone(),
                        self.frequencies.clone(),
                        c.rms_norm_eps,
                        start_pos as i32,
                        c.num_attention_heads as i32,
                        0,
                    )
                    .generics(vec![
                        bf16::DTYPE.to_string(),
                        ATTENTION_HEAD_DIM.to_string(),
                        ROTARY_DIM.to_string(),
                        "true".into(),
                    ])
                    .grid((rows as u32, c.num_attention_heads as u32, 1))
                    .execute(exec)?;

                    kernels::qk(
                        output_pointer(k, exec)?,
                        &raw_k,
                        w.k_norm.clone(),
                        self.frequencies.clone(),
                        c.rms_norm_eps,
                        start_pos as i32,
                        c.num_key_value_heads as i32,
                        start_pos as i32,
                    )
                    .generics(vec![
                        bf16::DTYPE.to_string(),
                        ATTENTION_HEAD_DIM.to_string(),
                        ROTARY_DIM.to_string(),
                        "false".into(),
                    ])
                    .grid((rows as u32, c.num_key_value_heads as u32, 1))
                    .execute(exec)?;

                    kernels::cache_v(output_pointer(v, exec)?, &raw_v, start_pos as i32)
                        .generics(vec![
                            bf16::DTYPE.to_string(),
                            ATTENTION_HEAD_DIM.to_string(),
                        ])
                        .grid((rows as u32, c.num_key_value_heads as u32, 1))
                        .execute(exec)?;
                    let mut attended =
                        api::zeros::<bf16>(&[rows, c.num_attention_heads * ATTENTION_HEAD_DIM])
                            .execute(exec)?;

                    kernels::gqa_attention(
                        (&mut attended).partition([1, ATTENTION_HEAD_DIM]),
                        &q,
                        &*k,
                        &*v,
                        &qg,
                        start_pos as i32,
                    )
                    .generics(vec![
                        ATTENTION_HEAD_DIM.to_string(),
                        c.num_attention_heads.to_string(),
                        c.num_key_value_heads.to_string(),
                    ])
                    .execute(exec)?;
                    linear(&attended, w.out.clone(), exec)?
                }
                _ => anyhow::bail!("hybrid layer/state mismatch"),
            };
            x = add(&x, &mixed, exec)?;
            let n = normalize(&x, layer.post_norm.clone(), c.rms_norm_eps, exec)?;
            let logits = linear(&n, layer.router.clone(), exec)?;
            let top = c.num_experts_per_tok;
            let i = c.moe_intermediate_size;
            let mut selected = api::zeros::<i32>(&[rows, top]).execute(exec)?;
            let mut scores = api::zeros::<bf16>(&[rows, top]).execute(exec)?;

            kernels::route(
                (&mut selected).partition([1, top]),
                (&mut scores).partition([1, top]),
                &logits,
            )
            .generics(vec![
                bf16::DTYPE.to_string(),
                c.num_experts.to_string(),
                top.to_string(),
            ])
            .grid((rows as u32, 1, 1))
            .execute(exec)?;
            let mut gu = api::zeros::<bf16>(&[rows * top, 2 * i]).execute(exec)?;

            kernels::expert(
                (&mut gu).partition([1, EXPERT_OUTPUT_COLUMNS]),
                &n,
                layer.gate_up.clone(),
                &selected,
                top as i32,
            )
            .generics(vec![
                bf16::DTYPE.to_string(),
                h.to_string(),
                (2 * i).to_string(),
                top.to_string(),
            ])
            .execute(exec)?;
            let mut activated = api::zeros::<bf16>(&[rows * top, i]).execute(exec)?;

            kernels::expert_swiglu((&mut activated).partition([1, POINTWISE_COLUMNS]), &gu)
                .generics(vec![i.to_string()])
                .execute(exec)?;
            let mut routed = api::zeros::<bf16>(&[rows * top, h]).execute(exec)?;

            kernels::expert(
                (&mut routed).partition([1, EXPERT_OUTPUT_COLUMNS]),
                &activated,
                layer.down.clone(),
                &selected,
                1,
            )
            .generics(vec![
                bf16::DTYPE.to_string(),
                i.to_string(),
                h.to_string(),
                top.to_string(),
            ])
            .execute(exec)?;
            let gate = linear(&n, layer.shared_gate.clone(), exec)?;
            let up = linear(&n, layer.shared_up.clone(), exec)?;
            let mut shared_act =
                api::zeros::<bf16>(&[rows, c.shared_expert_intermediate_size]).execute(exec)?;
            shared::swiglu(
                (&mut shared_act).partition([1, SHARED_POINTWISE_COLUMNS]),
                &gate,
                &up,
            )
            .generics(vec![bf16::DTYPE.to_string()])
            .execute(exec)?;
            let shared_out = linear(&shared_act, layer.shared_down.clone(), exec)?;
            let shared_gate = linear(&n, layer.shared_router.clone(), exec)?;
            let mut merged = api::zeros::<bf16>(&[rows, h]).execute(exec)?;

            kernels::moe_merge(
                (&mut merged).partition([1, POINTWISE_COLUMNS]),
                &routed,
                &scores,
                &shared_out,
                &shared_gate,
            )
            .generics(vec![top.to_string()])
            .execute(exec)?;
            x = add(&x, &merged, exec)?;
        }
        let mut last = api::zeros::<bf16>(&[1, h]).execute(exec)?;

        kernels::last_row((&mut last).partition([1, POINTWISE_COLUMNS]), &x).execute(exec)?;
        let n = normalize(&last, self.norm.clone(), c.rms_norm_eps, exec)?;
        let logits = linear_f32(&n, self.lm_head.clone(), exec)?;
        Ok(self.readbacks.enqueue(logits, exec)?)
    }
}

unsafe fn normalize(
    x: &Tensor<bf16>,
    w: Norm,
    eps: f32,
    exec: &ExecutionContext,
) -> Result<Tensor<bf16>> {
    let rows = x.shape()[0] as usize;
    let d = x.shape()[1] as usize;
    let mut out = api::zeros::<bf16>(&[rows, d]).execute(exec)?;

    kernels::norm((&mut out).partition([1, d]), x, w, eps)
        .generics(vec![d.to_string(), "true".into()])
        .execute(exec)?;
    Ok(out)
}
unsafe fn linear(x: &Tensor<bf16>, w: Weight, exec: &ExecutionContext) -> Result<Tensor<bf16>> {
    project(x, w, exec)
}
unsafe fn linear_f32(x: &Tensor<bf16>, w: Weight, exec: &ExecutionContext) -> Result<Tensor<f32>> {
    project(x, w, exec)
}
unsafe fn project<T: DType>(
    x: &Tensor<bf16>,
    w: Weight,
    exec: &ExecutionContext,
) -> Result<Tensor<T>> {
    let mut out = api::zeros::<T>(&[x.shape()[0] as usize, w.shape()[0] as usize]).execute(exec)?;
    let bias = api::zeros::<bf16>(&[1]).execute(exec)?;
    shared::linear(
        (&mut out).partition([LINEAR_TILE_ROWS, LINEAR_TILE_COLUMNS]),
        x,
        w,
        &bias,
    )
    .generics(vec![
        bf16::DTYPE.to_string(),
        T::DTYPE.to_string(),
        x.shape()[1].to_string(),
        "false".into(),
        LINEAR_TILE_ROWS.to_string(),
        LINEAR_TILE_COLUMNS.to_string(),
        LINEAR_TILE_REDUCTION.to_string(),
    ])
    .execute(exec)?;
    Ok(out)
}
unsafe fn add(a: &Tensor<bf16>, b: &Tensor<bf16>, exec: &ExecutionContext) -> Result<Tensor<bf16>> {
    let mut out =
        api::zeros::<bf16>(&[a.shape()[0] as usize, a.shape()[1] as usize]).execute(exec)?;
    shared::add((&mut out).partition([1, SHARED_POINTWISE_COLUMNS]), a, b)
        .generics(vec![bf16::DTYPE.to_string(), "1".into()])
        .execute(exec)?;
    Ok(out)
}

/// Keep raw destinations alive through completion, including cancellation. The
/// caller owns exclusive state/scratch and orders every access on this exec.
unsafe fn output_pointer<T: DType>(
    tensor: &mut Tensor<T>,
    exec: &ExecutionContext,
) -> Result<cuda_async::device_buffer::DevicePointer<T>> {
    exec.retain(tensor.into_shared_alias())?;
    Ok(tensor.device_pointer())
}
