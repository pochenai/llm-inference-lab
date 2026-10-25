//! Correctness-first Qwen2 causal decoder. All numerical operators use cuTile kernels.
//!
//! Qwen2.5-0.5B-Instruct (T = number of input tokens):
//! ```text
//! Token IDs [T]
//!     |
//! Embedding [151936, 896] -> hidden [T, 896]
//!     |
//! +---------------- Decoder block x 24 ----------------+
//! | hidden --------------------------------------+    |
//! |   |                                          |    |
//! | RMSNorm                                      |    |
//! |   |                                          |    |
//! | Q/K/V projections + bias                      |    |
//! |   Q: 14 heads x 64; K/V: 2 heads x 64         |    |
//! |   |                                          |    |
//! | RoPE(Q,K); cache rotated K and V              |    |
//! |   |                                          |    |
//! | Causal GQA (7 Q heads per KV head)            |    |
//! |   |                                          |    |
//! | O projection [896 -> 896]                     |    |
//! |   |                                          |    |
//! | Add <----------------------------------------+    |
//! |   |                                               |
//! |   +------------------------------------------+    |
//! |   |                                          |    |
//! | RMSNorm                                      |    |
//! |   |                                          |    |
//! | gate: 896 -> 4864    up: 896 -> 4864          |    |
//! |       SiLU(gate) * up                         |    |
//! |              |                               |    |
//! |       down: 4864 -> 896                       |    |
//! |              |                               |    |
//! | Add <----------------------------------------+    |
//! +--------------|------------------------------------+
//!                |
//! Final RMSNorm -> last-token hidden [1, 896]
//!                |
//! Language Model head (tied to embedding) -> logits [151936]
//!                |
//! Greedy argmax -> next token ID
//! ```

use super::{
    kernels,
    readback::{enqueue_readback, Readback, ReadbackCache},
    workspace::WorkspacePool,
};
use crate::runner::{
    kv_pool::{KVContext, NaiveKVPool, NaiveKVPoolConfig, NaiveReadView},
    loader::{EosTokenId, WeightLoader},
    upload::UploadPool,
};
use crate::{AttentionBackend, CausalModel, ModelError, ModelInput};
use cuda_async::{device_operation::*, scheduling_policies::SchedulingPolicy};
use cutile::{
    api,
    tensor::{PartitionMut, Reshape, Tensor},
    tile_kernel::TileKernel,
};
use serde::Deserialize;
use std::{path::Path, sync::Arc};
use tokenizers::Tokenizer;

type Weight = Arc<Tensor<f32>>;

#[derive(Clone, Debug, Deserialize)]
pub struct Qwen2Config {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub eos_token_id: u32,
    pub hidden_act: String,
    #[serde(default)]
    use_sliding_window: bool,
    #[serde(default)]
    rope_scaling: Option<serde_json::Value>,
}

struct Layer {
    input_norm: Weight,
    post_norm: Weight,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    q_bias: Weight,
    k_bias: Weight,
    v_bias: Weight,
    gate: Weight,
    up: Weight,
    down: Weight,
}

pub struct Qwen2Engine {
    config: Qwen2Config,
    tokenizer: Tokenizer,
    embedding: Weight,
    norm: Weight,
    lm_head: Weight,
    inv_freq: Weight,
    layers: Vec<Layer>,
    eos: Vec<u32>,
    attention: GqaAttention,
    workspaces: Arc<WorkspacePool>,
    input_uploads: Arc<UploadPool<u32>>,
    token_readbacks: ReadbackCache<i32>,
    logit_readbacks: ReadbackCache<f32>,
}

impl Qwen2Engine {
    /// Load the supplied local checkpoint; missing or incompatible weights are errors.
    /// BF16 checkpoint values are converted once to FP32 by the existing WeightLoader.
    pub async fn load(path: &Path, policy: &Arc<dyn SchedulingPolicy>) -> Result<Self, ModelError> {
        let loader = WeightLoader::new(path)?;
        let config: Qwen2Config = loader.config()?;
        if !(config.model_type == "qwen2"
            && config.hidden_act == "silu"
            && !config.use_sliding_window
            && config.rope_scaling.is_none())
        {
            return Err(ModelError::InvalidInput(
                "expected Qwen2 with SiLU, unscaled RoPE, and full causal attention".into(),
            ));
        }
        if !(config.num_attention_heads > 0
            && config.num_key_value_heads > 0
            && config.hidden_size % config.num_attention_heads == 0
            && config.hidden_size / config.num_attention_heads == 64
            && config.num_attention_heads % config.num_key_value_heads == 0
            && config.hidden_size % 32 == 0
            && config.intermediate_size % 32 == 0
            && config.vocab_size > 0
            && config.vocab_size <= 256 * 1024
            && config.rms_norm_eps.is_finite()
            && config.rms_norm_eps > 0.0
            && config.rope_theta.is_finite()
            && config.rope_theta > 0.0
            && config.num_hidden_layers > 0)
        {
            return Err(ModelError::InvalidInput(
                "unsupported Qwen2 dimensions (initial backend requires 64-dimensional heads)"
                    .into(),
            ));
        }
        let h = config.hidden_size;
        let k = config.num_key_value_heads * 64;
        let embedding = weight(
            &loader,
            "model.embed_tokens.weight",
            &[config.vocab_size, h],
            policy,
        )
        .await?;
        let norm = weight(&loader, "model.norm.weight", &[h], policy).await?;
        let lm_head = if config.tie_word_embeddings {
            embedding.clone()
        } else {
            weight(&loader, "lm_head.weight", &[config.vocab_size, h], policy).await?
        };
        let inv_freq_host: Vec<f32> = (0..32)
            .map(|i| config.rope_theta.powf(-((2 * i) as f32) / 64.0))
            .collect();
        let inv_freq = Arc::new(
            api::copy_host_vec_to_device(&Arc::new(inv_freq_host))
                .schedule(policy)?
                .await?,
        );
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let name = format!("model.layers.{i}");
            layers.push(Layer {
                input_norm: weight(
                    &loader,
                    &format!("{name}.input_layernorm.weight"),
                    &[h],
                    policy,
                )
                .await?,
                post_norm: weight(
                    &loader,
                    &format!("{name}.post_attention_layernorm.weight"),
                    &[h],
                    policy,
                )
                .await?,
                q: weight(
                    &loader,
                    &format!("{name}.self_attn.q_proj.weight"),
                    &[h, h],
                    policy,
                )
                .await?,
                k: weight(
                    &loader,
                    &format!("{name}.self_attn.k_proj.weight"),
                    &[k, h],
                    policy,
                )
                .await?,
                v: weight(
                    &loader,
                    &format!("{name}.self_attn.v_proj.weight"),
                    &[k, h],
                    policy,
                )
                .await?,
                o: weight(
                    &loader,
                    &format!("{name}.self_attn.o_proj.weight"),
                    &[h, h],
                    policy,
                )
                .await?,
                q_bias: weight(
                    &loader,
                    &format!("{name}.self_attn.q_proj.bias"),
                    &[h],
                    policy,
                )
                .await?,
                k_bias: weight(
                    &loader,
                    &format!("{name}.self_attn.k_proj.bias"),
                    &[k],
                    policy,
                )
                .await?,
                v_bias: weight(
                    &loader,
                    &format!("{name}.self_attn.v_proj.bias"),
                    &[k],
                    policy,
                )
                .await?,
                gate: weight(
                    &loader,
                    &format!("{name}.mlp.gate_proj.weight"),
                    &[config.intermediate_size, h],
                    policy,
                )
                .await?,
                up: weight(
                    &loader,
                    &format!("{name}.mlp.up_proj.weight"),
                    &[config.intermediate_size, h],
                    policy,
                )
                .await?,
                down: weight(
                    &loader,
                    &format!("{name}.mlp.down_proj.weight"),
                    &[h, config.intermediate_size],
                    policy,
                )
                .await?,
            });
            eprintln!("Loaded Qwen2 layer {}/{}", i + 1, config.num_hidden_layers);
        }
        let eos = match loader.generation_config()?.and_then(|g| g.eos_token_id) {
            Some(EosTokenId::One(id)) => vec![id],
            Some(EosTokenId::Many(ids)) => ids,
            None => vec![config.eos_token_id],
        };
        let attention = GqaAttention {
            q_heads: config.num_attention_heads,
            kv_heads: config.num_key_value_heads,
        };
        Ok(Self {
            workspaces: Arc::default(),
            input_uploads: Arc::default(),
            token_readbacks: Default::default(),
            logit_readbacks: Default::default(),
            attention,
            tokenizer: loader.tokenizer()?,
            config,
            embedding,
            norm,
            lm_head,
            inv_freq,
            layers,
            eos,
        })
    }

    pub fn config(&self) -> &Qwen2Config {
        &self.config
    }

    /// Flatten KV heads into NaiveKVPool's per-token vector; attention restores GQA heads.
    pub fn kv_pool_config(&self, max_seq_len: usize) -> Result<NaiveKVPoolConfig, ModelError> {
        if !(max_seq_len > 0 && max_seq_len <= self.config.max_position_embeddings) {
            return Err(ModelError::InvalidInput(
                "context length exceeds model limit".into(),
            ));
        }
        Ok(NaiveKVPoolConfig {
            num_layers: self.layers.len(),
            num_slots: max_seq_len,
            max_rows: 1,
            max_seq_len,
            head_dim: self.config.num_key_value_heads * 64,
        })
    }

    /// Diagnostic step: compute the suffix, commit its KV, and read last-position logits.
    pub async fn step_with_logits(
        &self,
        tokens: &[u32],
        kv: &mut KVContext<NaiveKVPool>,
        policy: Arc<dyn SchedulingPolicy>,
    ) -> Result<(u32, Vec<f32>), ModelError> {
        let (token, logits) = self.step(tokens, kv, policy, true).await?;
        Ok((token, logits.unwrap()))
    }

    async fn step(
        &self,
        tokens: &[u32],
        kv: &mut KVContext<NaiveKVPool>,
        policy: Arc<dyn SchedulingPolicy>,
        read_logits: bool,
    ) -> Result<(u32, Option<Vec<f32>>), ModelError> {
        if kv.is_pending()
            || tokens.len() <= kv.ready_len()
            || tokens.len() > kv.max_seq_len()
            || tokens.len() > self.config.max_position_embeddings
            || tokens[kv.ready_len()..]
                .iter()
                .any(|&id| id as usize >= self.config.vocab_size)
        {
            return Err(ModelError::InvalidInput(
                "expected valid tokens with an uncached suffix and idle context".into(),
            ));
        }
        let input = ModelInput {
            token_ids: tokens,
            kv,
        };
        let (token, logits) = Self::execute_append_batch(&mut [input], &policy, |batch, exec| {
            let input = &mut batch[0];
            // SAFETY: every GPU access, including D2H, belongs to this submission.
            unsafe { self.submit(input.token_ids, input.kv, read_logits, exec) }
        })
        .await?;
        // Outer completion includes D2H; only now expose host references.
        let token = token.as_slice()[0] as u32;
        if token as usize >= self.config.vocab_size {
            return Err(ModelError::Execution(
                "non-finite logits or invalid argmax".into(),
            ));
        }
        let logits = logits.map(|values| values.as_slice().to_vec());
        Ok((token, logits))
    }

    unsafe fn submit(
        &self,
        tokens: &[u32],
        kv: &mut KVContext<NaiveKVPool>,
        read_logits: bool,
        exec: &ExecutionContext,
    ) -> Result<(Readback<i32>, Option<Readback<f32>>), ModelError> {
        let start = kv.ready_len();
        let rows = tokens.len() - start;
        // reuse the same buffer
        let mut scratch = self.workspaces.acquire(rows, &self.config, exec)?;
        let ids = self.input_uploads.upload_slice(&tokens[start..], exec)?;
        kernels::embedding(
            (&mut scratch.x).partition([1, 256]),
            ids,
            self.embedding.clone(),
        )
        .execute(exec)?;
        // All layer accesses share exec's stream. A buffer is overwritten only after
        // its last consumer; no scratch allocation grows with the number of layers.
        for (index, layer) in self.layers.iter().enumerate() {
            self.normalize(
                &mut scratch.norm,
                &scratch.x,
                layer.input_norm.clone(),
                0,
                exec,
            )?;
            self.linear(&mut scratch.q, &scratch.norm, layer.q.clone(), None, exec)?;
            self.linear(&mut scratch.k, &scratch.norm, layer.k.clone(), None, exec)?;
            self.linear(
                &mut scratch.v,
                &scratch.norm,
                layer.v.clone(),
                Some(layer.v_bias.clone()),
                exec,
            )?;
            self.rope(
                &mut scratch.rotated_q,
                &scratch.q,
                layer.q_bias.clone(),
                start,
                exec,
            )?;
            self.rope(
                &mut scratch.rotated_k,
                &scratch.k,
                layer.k_bias.clone(),
                start,
                exec,
            )?;
            kv.store(
                index,
                scratch.rotated_k.slice(&[0..rows])?,
                scratch.v.slice(&[0..rows])?,
                exec,
            )?;
            let view = kv.read_view(index)?;
            self.attention.forward_into(
                &mut scratch.attention,
                &scratch.rotated_q,
                &[view],
                &[0..rows],
                exec,
            )?;
            // Q projection has no remaining consumers: reuse its buffer for O/down.
            self.linear(
                &mut scratch.q,
                &scratch.attention,
                layer.o.clone(),
                None,
                exec,
            )?;
            add(&mut scratch.residual, &scratch.x, &scratch.q, exec)?;
            self.normalize(
                &mut scratch.norm,
                &scratch.residual,
                layer.post_norm.clone(),
                0,
                exec,
            )?;
            self.linear(
                &mut scratch.gate,
                &scratch.norm,
                layer.gate.clone(),
                None,
                exec,
            )?;
            self.linear(&mut scratch.up, &scratch.norm, layer.up.clone(), None, exec)?;
            kernels::swiglu(
                (&mut scratch.activated).partition([1, 256]),
                &scratch.gate,
                &scratch.up,
            )
            .execute(exec)?;
            self.linear(
                &mut scratch.q,
                &scratch.activated,
                layer.down.clone(),
                None,
                exec,
            )?;
            add(&mut scratch.x, &scratch.residual, &scratch.q, exec)?;
        }
        self.normalize(
            &mut scratch.final_norm,
            &scratch.x,
            self.norm.clone(),
            rows - 1,
            exec,
        )?;
        self.linear(
            &mut scratch.logits,
            &scratch.final_norm,
            self.lm_head.clone(),
            None,
            exec,
        )?;
        let flat = scratch.logits.reshape(&[self.config.vocab_size])?;
        let blocks = self.config.vocab_size.div_ceil(256);
        kernels::argmax_blocks(
            (&mut scratch.maxima).partition([1]),
            (&mut scratch.indices).partition([1]),
            &flat,
            self.config.vocab_size as i32,
        )
        .execute(exec)?;
        kernels::argmax_final(
            (&mut scratch.winner).partition([1]),
            &scratch.maxima,
            &scratch.indices,
            blocks as i32,
        )
        .execute(exec)?;
        let token = enqueue_readback(scratch.winner, &self.token_readbacks, exec)?;
        let values = if read_logits {
            Some(enqueue_readback(flat, &self.logit_readbacks, exec)?)
        } else {
            None
        };
        Ok((token, values))
    }

    unsafe fn linear(
        &self,
        out: &mut Tensor<f32>,
        x: &Tensor<f32>,
        w: Weight,
        bias: Option<Weight>,
        exec: &ExecutionContext,
    ) -> Result<(), ModelError> {
        let k = x.shape()[1] as usize;
        let has_bias = bias.is_some();
        // Bias-free specializations never read this existing rank-one weight.
        let bias = bias.unwrap_or_else(|| self.norm.clone());
        kernels::linear(out.partition([4, 32]), x, w, bias)
            .generics(vec![k.to_string(), has_bias.to_string()])
            .execute(exec)?;
        Ok(())
    }

    unsafe fn normalize(
        &self,
        out: &mut Tensor<f32>,
        x: &Tensor<f32>,
        w: Weight,
        offset: usize,
        exec: &ExecutionContext,
    ) -> Result<(), ModelError> {
        let width = self.config.hidden_size;
        kernels::rms_norm(
            out.partition([1, width.next_power_of_two()]),
            x,
            w,
            self.config.rms_norm_eps,
            width as i32,
            offset as i32,
        )
        .generics(vec![width.next_power_of_two().to_string()])
        .execute(exec)?;
        Ok(())
    }

    unsafe fn rope(
        &self,
        out: &mut Tensor<f32>,
        x: &Tensor<f32>,
        bias: Weight,
        start: usize,
        exec: &ExecutionContext,
    ) -> Result<(), ModelError> {
        kernels::rope(
            out.partition([1, 64]),
            x,
            bias,
            self.inv_freq.clone(),
            start as i32,
        )
        .execute(exec)?;
        Ok(())
    }
}

/// Compatible attention backend for flat [token, kv_head * 64] Naive storage.
struct GqaAttention {
    q_heads: usize,
    kv_heads: usize,
}

impl GqaAttention {
    unsafe fn forward_into(
        &self,
        out: &mut Tensor<f32>,
        q: &Tensor<f32>,
        views: &[NaiveReadView<'_>],
        query_ranges: &[std::ops::Range<usize>],
        exec: &ExecutionContext,
    ) -> Result<(), ModelError> {
        let [view] = views else {
            return Err(ModelError::InvalidInput(
                "initial Qwen GQA backend expects one context".into(),
            ));
        };
        let rows = q.shape()[0] as usize;
        if q.shape().len() != 2
            || q.shape()[1] as usize != self.q_heads * 64
            || view.head_dim() != self.kv_heads * 64
            || query_ranges != [0..rows]
            || rows > view.positions().len()
        {
            return Err(ModelError::InvalidInput(
                "incompatible GQA query or KV layout".into(),
            ));
        }
        view.retain(exec)?;
        // SAFETY: view retains storage and allocation leases; writes precede reads on exec.
        let (k, v, table) = view.tensors();
        kernels::attention(
            out.partition([1, 64]),
            q,
            k.clone(),
            v.clone(),
            table.clone(),
            view.row() as i32,
            (view.positions().end - rows) as i32,
            self.q_heads as i32,
            self.kv_heads as i32,
        )
        .execute(exec)?;
        Ok(())
    }
}

impl AttentionBackend<NaiveKVPool> for GqaAttention {
    fn forward(
        &self,
        q: Weight,
        views: &[NaiveReadView<'_>],
        query_ranges: &[std::ops::Range<usize>],
        exec: &ExecutionContext,
    ) -> Result<Weight, ModelError> {
        // SAFETY: the attention kernel writes every valid output before any read.
        unsafe {
            let mut out = Tensor::<f32>::uninitialized(q.size())
                .execute(exec)?
                .assume_init()
                .reshape(&[q.shape()[0] as usize, self.q_heads * 64])?;
            self.forward_into(&mut out, &q, views, query_ranges, exec)?;
            Ok(Arc::new(out))
        }
    }
}

impl CausalModel<NaiveKVPool> for Qwen2Engine {
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn validate_input(&self, token_ids: &[u32]) -> Result<(), ModelError> {
        if token_ids
            .iter()
            .any(|&id| id as usize >= self.config.vocab_size)
        {
            return Err(ModelError::InvalidInput(
                "token ID outside model vocabulary".into(),
            ));
        }
        Ok(())
    }

    async fn next_tokens(
        &self,
        batch: &mut [ModelInput<'_, NaiveKVPool>],
        policy: Arc<dyn SchedulingPolicy>,
    ) -> Vec<Result<u32, ModelError>> {
        // shortcut: serialize Qwen batch items until the attention kernel supports packed contexts.
        let mut outputs = Vec::with_capacity(batch.len());
        for input in batch {
            outputs.push(
                self.step(input.token_ids, input.kv, policy.clone(), false)
                    .await
                    .map(|(token, _)| token),
            );
        }
        outputs
    }

    fn is_eos(&self, token: u32) -> bool {
        self.eos.contains(&token)
    }
}

async fn weight(
    loader: &WeightLoader,
    name: &str,
    shape: &[usize],
    policy: &Arc<dyn SchedulingPolicy>,
) -> Result<Weight, ModelError> {
    let w = loader.load_device::<f32>(name, policy).await?;
    if !(w
        .shape()
        .iter()
        .map(|&v| v as usize)
        .eq(shape.iter().copied()))
    {
        return Err(ModelError::InvalidInput(format!(
            "unexpected shape for {name}"
        )));
    }
    Ok(w)
}

unsafe fn add(
    out: &mut Tensor<f32>,
    x: &Tensor<f32>,
    y: &Tensor<f32>,
    exec: &ExecutionContext,
) -> Result<(), ModelError> {
    kernels::add(out.partition([1, 256]), x, y).execute(exec)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::kv_pool::KVPool;
    use anyhow::Context;
    use cuda_async::scheduling_policies::SingleStream;
    use cutile::cuda_core::Device;
    use cutile::tensor::{IntoPartition, Unpartition};

    #[tokio::test]
    async fn linear_epilogue_bias_matches_cpu_with_partial_row_tile() -> Result<(), ModelError> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = cuda_async::device_context::global_policy(0)?;
        let cache = ReadbackCache::default();
        let input = Arc::new((0..96).map(|i| (i / 32 + 1) as f32).collect());
        let weights = Arc::new((0..64 * 32).map(|i| (i / 32 % 7 + 1) as f32).collect());
        let biases = Arc::new((0..64).map(|i| i as f32 * 0.25).collect());
        for has_bias in [false, true] {
            let actual = with_context(|exec| {
                value((|| -> Result<_, ModelError> {
                    // SAFETY: producers and D2H share exec; host reads follow await.
                    unsafe {
                        let x = api::copy_host_vec_to_device(&input)
                            .execute(exec)?
                            .reshape(&[3, 32])?;
                        let w = api::copy_host_vec_to_device(&weights)
                            .execute(exec)?
                            .reshape(&[64, 32])?;
                        let bias = api::copy_host_vec_to_device(&biases).execute(exec)?;
                        let out = kernels::linear(
                            Tensor::<f32>::uninitialized(3 * 64)
                                .execute(exec)?
                                .assume_init()
                                .reshape(&[3, 64])?
                                .partition([4, 32]),
                            x,
                            w,
                            bias,
                        )
                        .generics(vec!["32".into(), has_bias.to_string()])
                        .map(|(out, ..)| out)
                        .unpartition()
                        .execute(exec)?;
                        Ok(enqueue_readback(out, &cache, exec)?)
                    }
                })())
            })
            .schedule(&policy)?
            .await??;
            for (index, &value) in actual.as_slice().iter().enumerate() {
                let row = index / 64;
                let column = index % 64;
                let expected = 32.0 * (row + 1) as f32 * (column % 7 + 1) as f32
                    + if has_bias { biases[column] } else { 0.0 };
                assert_eq!(value, expected);
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn u32_embedding_and_async_token_readback() -> Result<(), ModelError> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let device = Device::new(0).map_err(cuda_async::error::DeviceError::from)?;
        let policy: Arc<dyn SchedulingPolicy> =
            Arc::new(SingleStream::new(&device).map_err(cuda_async::error::DeviceError::from)?);
        let uploads = Arc::new(UploadPool::<u32>::default());
        let weights = Arc::new((0..4 * 896).map(|v| v as f32).collect::<Vec<_>>());
        let tokens = ReadbackCache::default();
        let outputs = ReadbackCache::default();
        let (actual, token) = with_context(|exec| {
            value((|| -> Result<_, ModelError> {
                // SAFETY: all resources are owned/retained through completion.
                unsafe {
                    let ids = uploads.upload_slice(&[3, 0, 2], exec)?;
                    let w = api::copy_host_vec_to_device(&weights)
                        .execute(exec)?
                        .reshape(&[4, 896])?;
                    let out = kernels::embedding(
                        Tensor::<f32>::uninitialized(3 * 896)
                            .execute(exec)?
                            .assume_init()
                            .reshape(&[3, 896])?
                            .partition([1, 256]),
                        ids,
                        w,
                    )
                    .map(|(out, ..)| out)
                    .unpartition()
                    .execute(exec)?;
                    let winner =
                        api::copy_host_vec_to_device(&Arc::new(vec![151645i32])).execute(exec)?;
                    Ok((
                        enqueue_readback(out, &outputs, exec)?,
                        enqueue_readback(winner, &tokens, exec)?,
                    ))
                }
            })())
        })
        .schedule(&policy)?
        .await??;
        // Successful outer await completed both asynchronous copies.
        let actual = actual.as_slice();
        assert_eq!(token.as_slice()[0], 151645);
        for (row, id) in [3, 0, 2].into_iter().enumerate() {
            assert_eq!(
                &actual[row * 896..(row + 1) * 896],
                &weights[id * 896..(id + 1) * 896]
            );
        }
        Ok(())
    }

    /// Real-checkpoint regression, explicitly enabled with local reference artifacts.
    #[tokio::test]
    #[ignore = "requires QWEN2_MODEL_DIR and QWEN2_REFERENCE_DIR; see engine README"]
    async fn real_checkpoint_matches_reference_and_chunked_kv() -> anyhow::Result<()> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let model_dir = std::env::var("QWEN2_MODEL_DIR").context("set QWEN2_MODEL_DIR")?;
        let reference_dir = std::path::PathBuf::from(
            std::env::var("QWEN2_REFERENCE_DIR").context("set QWEN2_REFERENCE_DIR")?,
        );
        let reference: serde_json::Value =
            serde_json::from_slice(&std::fs::read(reference_dir.join("reference.json"))?)?;
        let tokens: Vec<u32> = serde_json::from_value(reference["input_ids"].clone())?;
        let expected_ids: Vec<u32> = serde_json::from_value(reference["generated_ids"].clone())?;
        let raw = std::fs::read(reference_dir.join("logits.f32"))?;
        let expected: Vec<f32> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let device = Device::new(0)?;
        let policy: Arc<dyn SchedulingPolicy> = Arc::new(SingleStream::new(&device)?);
        let model = Qwen2Engine::load(Path::new(&model_dir), &policy).await?;
        let pool = NaiveKVPool::new(
            model.kv_pool_config(tokens.len() + expected_ids.len() + 1)?,
            &policy,
        )
        .await?;
        let initial = pool.metrics();
        let mut full = KVContext::new(pool.clone())?;
        let invalid = model
            .next_tokens(
                &mut [ModelInput {
                    token_ids: &[],
                    kv: &mut full,
                }],
                policy.clone(),
            )
            .await;
        assert!(invalid[0].is_err());
        assert!(!full.is_pending());
        let (first, actual) = model
            .step_with_logits(&tokens, &mut full, policy.clone())
            .await?;
        assert_eq!(first, expected_ids[0]);
        assert_eq!(actual.len(), expected.len());
        for (&a, &b) in actual.iter().zip(&expected) {
            assert!(
                a.is_finite() && (a - b).abs() < 1e-3,
                "logits differ: {a} vs {b}"
            );
        }
        assert_eq!(full.ready_len(), tokens.len());
        full.release();
        assert_eq!(pool.metrics(), initial);
        let mut chunked = KVContext::new(pool.clone())?;
        // Prefix logits are intentionally discarded: only its KV state is continued.
        let split = tokens.len() / 2 + 1;
        let prefix_next = model
            .next_tokens(
                &mut [ModelInput {
                    token_ids: &tokens[..split],
                    kv: &mut chunked,
                }],
                policy.clone(),
            )
            .await
            .pop()
            .unwrap()?;
        let (next, continued) = model
            .step_with_logits(&tokens, &mut chunked, policy.clone())
            .await?;
        assert_eq!(next, first);
        for (&a, &b) in continued.iter().zip(&actual) {
            assert!((a - b).abs() < 1e-3);
        }
        let mut history = tokens.clone();
        history.push(next);
        let mut generated = vec![next];
        while generated.len() < expected_ids.len() && !model.is_eos(*generated.last().unwrap()) {
            let token = model
                .next_tokens(
                    &mut [ModelInput {
                        token_ids: &history,
                        kv: &mut chunked,
                    }],
                    policy.clone(),
                )
                .await
                .pop()
                .unwrap()?;
            history.push(token);
            generated.push(token);
        }
        assert_eq!(generated, expected_ids);
        assert_eq!(chunked.ready_len() + 1, history.len());
        chunked.release();
        assert_eq!(pool.metrics(), initial);

        // One model may serve different streams concurrently. Scratch checkouts
        // must remain exclusive even when one forward is still in flight.
        let mut concurrent_config = model.kv_pool_config(tokens.len())?;
        concurrent_config.max_rows = 2;
        concurrent_config.num_slots *= 2;
        let concurrent_pool = NaiveKVPool::new(concurrent_config, &policy).await?;
        let idle = concurrent_pool.metrics();
        let other_policy: Arc<dyn SchedulingPolicy> = Arc::new(SingleStream::new(&device)?);
        let mut left = KVContext::new(concurrent_pool.clone())?;
        let mut right = KVContext::new(concurrent_pool.clone())?;
        let mut left_input = [ModelInput {
            token_ids: &tokens,
            kv: &mut left,
        }];
        let mut right_input = [ModelInput {
            token_ids: &tokens[..split],
            kv: &mut right,
        }];
        let (a, b) = tokio::join!(
            model.next_tokens(&mut left_input, policy.clone()),
            model.next_tokens(&mut right_input, other_policy),
        );
        assert_eq!(*a[0].as_ref().unwrap(), first);
        assert_eq!(*b[0].as_ref().unwrap(), prefix_next);
        left.release();
        right.release();
        assert_eq!(concurrent_pool.metrics(), idle);

        let turn_len = expected_ids.len().div_ceil(2);
        let tokenizer = model.tokenizer().clone();
        let mut decoder = tokenizer.decode_stream(true);
        let mut streamed_text = String::new();
        for &id in &expected_ids {
            if let Some(delta) = decoder.step(id).unwrap() {
                streamed_text.push_str(&delta);
            }
        }
        assert_eq!(streamed_text, reference["text"].as_str().unwrap());
        let unicode = tokenizer.encode("Hello! 你好，世界🙂", false).unwrap();
        for end in 0..=unicode.len() {
            let ids = &unicode.get_ids()[..end];
            let mut decoder = tokenizer.decode_stream(true);
            let mut output = String::new();
            for &id in ids {
                if let Some(delta) = decoder.step(id).unwrap() {
                    output.push_str(&delta);
                }
            }
            let full = tokenizer.decode(ids, true).unwrap();
            assert!(full.starts_with(&output));
            assert!(!output.ends_with('�'));
            if !full.ends_with('�') {
                assert_eq!(output, full);
            }
        }
        let mut continued = tokens.clone();
        continued.extend_from_slice(&expected_ids[..turn_len]);
        continued.extend_from_slice(tokenizer.encode("1", true).unwrap().get_ids());
        let mut fresh = KVContext::new(pool.clone())?;
        let expected_next = model
            .next_tokens(
                &mut [ModelInput {
                    token_ids: &continued,
                    kv: &mut fresh,
                }],
                policy.clone(),
            )
            .await
            .pop()
            .unwrap()?;
        fresh.release();
        let expected_text = tokenizer.decode(&[expected_next], true).unwrap();
        let expected_first = tokenizer.decode(&expected_ids[..turn_len], true).unwrap();
        let prompt = reference["prompt"]
            .as_str()
            .context("reference prompt")?
            .to_owned();
        let cancel = tokio_util::sync::CancellationToken::new();
        let (commands, command_rx) = tokio::sync::mpsc::unbounded_channel();
        let (replies, reply_rx) = tokio::sync::mpsc::unbounded_channel();
        let scheduler = crate::scheduler::Scheduler::new(cancel.child_token(), commands, reply_rx);
        let owned_pool = pool.clone();
        let runner = crate::runner::model_runner::RunnerBuilder::default()
            .with_model_factory(move || async move { Ok(model) })
            .with_pool_factory(move |_| async move { Ok(owned_pool) })
            .with_policy_factory(move || Ok(policy))
            .build(command_rx, replies)
            .await?;
        let task = tokio::spawn(runner.run(cancel.child_token()));
        let result = async {
            let mut full = crate::types::SessionRequest::new(None, prompt.clone());
            full.sampling = Some(crate::types::SamplingConfig {
                max_new_tokens: Some(expected_ids.len()),
                ..Default::default()
            });
            assert_eq!(scheduler.execute(full).await?.output, streamed_text);
            let id = scheduler.open_session().await?;
            let mut first = crate::types::SessionRequest::new(Some(id), prompt);
            first.sampling = Some(crate::types::SamplingConfig {
                max_new_tokens: Some(turn_len),
                ..Default::default()
            });
            assert_eq!(scheduler.execute(first).await?.output, expected_first);
            let mut next = crate::types::SessionRequest::new(Some(id), "1".to_owned());
            next.sampling = Some(crate::types::SamplingConfig {
                max_new_tokens: Some(1),
                ..Default::default()
            });
            assert_eq!(scheduler.execute(next).await?.output, expected_text);
            scheduler.close_session(id).await?;
            Ok::<_, crate::error::EngineError>(())
        }
        .await;
        cancel.cancel();
        scheduler.wait_stopped().await;
        task.await?;
        result?;
        assert_eq!(pool.metrics(), initial);
        Ok(())
    }
}
