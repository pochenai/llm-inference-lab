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
    metrics::{Instruments, WorkspaceMetrics},
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
use std::{ops::Range, path::Path, sync::Arc, time::Instant};
use tokenizers::Tokenizer;

const KV_TOKENS_PER_TILE: i32 = 32;

type Weight = Arc<Tensor<f32>>;

/// One invocation's outputs: GPU readbacks before outer completion, CPU values after it.
struct StepOutput<Token, Logits> {
    token: Token,
    logits: Option<Logits>,
    /// Host wall time inside submit: scratch/input preparation, JIT and CUDA calls.
    /// Excludes KV mapping, GPU-completion waiting, KV commit and host result access.
    host_submit_seconds: f64,
}

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
    instruments: Instruments,
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
            tracing::debug!(target: "engine::qwen2", event = "layer_loaded", layer = i + 1, layers = config.num_hidden_layers);
        }
        let eos = match loader.generation_config()?.and_then(|g| g.eos_token_id) {
            Some(EosTokenId::One(id)) => vec![id],
            Some(EosTokenId::Many(ids)) => ids,
            None => vec![config.eos_token_id],
        };
        let attention = GqaAttention {
            q_heads: config.num_attention_heads,
            kv_heads: config.num_key_value_heads,
            metadata_uploads: Arc::default(),
        };
        let workspaces: Arc<WorkspacePool> = Arc::default();
        let instruments = Instruments::new(workspaces.metrics().model_instance);
        Ok(Self {
            instruments,
            workspaces,
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

    /// Resource-pool snapshot shared by every request on this model instance.
    /// Current/peak ownership cannot be attributed to a particular concurrent batch.
    pub fn workspace_metrics(&self) -> WorkspaceMetrics {
        self.workspaces.metrics()
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
        let output = self.step(tokens, kv, policy, true).await?;
        Ok((output.token, output.logits.unwrap()))
    }

    async fn step(
        &self,
        tokens: &[u32],
        kv: &mut KVContext<NaiveKVPool>,
        policy: Arc<dyn SchedulingPolicy>,
        read_logits: bool,
    ) -> Result<StepOutput<u32, Vec<f32>>, ModelError> {
        let mut batch = [ModelInput {
            token_ids: tokens,
            kv,
        }];
        self.validate_step_input(&batch[0])?;
        let output = self.forward_batch(&mut batch, policy, read_logits).await?;
        Ok(StepOutput {
            token: self.checked_token(output.token.as_slice()[0])?,
            logits: output.logits.map(|values| values.as_slice().to_vec()),
            host_submit_seconds: output.host_submit_seconds,
        })
    }

    fn validate_step_input(&self, input: &ModelInput<'_, NaiveKVPool>) -> Result<(), ModelError> {
        let tokens = input.token_ids;
        let kv = &input.kv;
        if kv.is_pending()
            || tokens.len() <= kv.ready_len()
            || tokens.len() > kv.max_seq_len()
            || tokens.len() > self.config.max_position_embeddings
        {
            return Err(ModelError::InvalidInput(
                "expected valid tokens with an uncached suffix and idle context".into(),
            ));
        }
        self.validate_input(&tokens[kv.ready_len()..])
    }

    fn checked_token(&self, token: i32) -> Result<u32, ModelError> {
        if token < 0 || token as usize >= self.config.vocab_size {
            return Err(ModelError::Execution(
                "non-finite logits or invalid argmax".into(),
            ));
        }
        Ok(token as u32)
    }

    /// One submission for the packed batch; outer completion precedes all KV commits
    /// and host reads. Mixed steps are labelled prefill if any input is not decode.
    async fn forward_batch(
        &self,
        batch: &mut [ModelInput<'_, NaiveKVPool>],
        policy: Arc<dyn SchedulingPolicy>,
        read_logits: bool,
    ) -> Result<StepOutput<Readback<i32>, Readback<f32>>, ModelError> {
        let rows = batch
            .iter()
            .map(|input| input.token_ids.len() - input.kv.ready_len())
            .sum();
        let prefill = batch.iter().any(|input| {
            input.kv.ready_len() == 0 || input.token_ids.len() - input.kv.ready_len() > 1
        });
        let start = Instant::now();
        let result = Self::execute_append_batch(batch, &policy, |batch, exec| {
            // SAFETY: every GPU access, including D2H, belongs to this submission.
            unsafe { self.submit(batch, read_logits, exec) }
        })
        .await;
        let step_wall_seconds = start.elapsed().as_secs_f64();
        let host_submit_seconds = result
            .as_ref()
            .ok()
            .map(|output| output.host_submit_seconds);
        self.instruments
            .step(prefill, rows, step_wall_seconds, host_submit_seconds);
        tracing::debug!(target: "engine::qwen2", event = "step_completed",
            phase = if prefill { "prefill" } else { "decode" },
            input_tokens = rows, batch_size = batch.len(), step_wall_seconds, host_submit_seconds,
            diagnostic_logits = read_logits, success = result.is_ok());
        result
    }

    unsafe fn submit(
        &self,
        batch: &mut [ModelInput<'_, NaiveKVPool>],
        read_logits: bool,
        exec: &ExecutionContext,
    ) -> Result<StepOutput<Readback<i32>, Readback<f32>>, ModelError> {
        let submit_started = Instant::now();
        // Thread-local NVTX range covers synchronous submission only, never the await.
        let _range = crate::profiling::ScopedRange::new(
            if batch.iter().any(|input| {
                input.kv.ready_len() == 0 || input.token_ids.len() - input.kv.ready_len() > 1
            }) {
                c"qwen2.prefill.submit"
            } else {
                c"qwen2.decode.submit"
            },
        );
        let mut rows = 0usize;
        let query_ranges: Vec<_> = batch
            .iter()
            .map(|input| {
                let start = rows;
                rows += input.token_ids.len() - input.kv.ready_len();
                start..rows
            })
            .collect();
        // Check the entire packed geometry before allocating/uploading model inputs.
        let mut scratch = self
            .workspaces
            .acquire(rows, batch.len(), &self.config, exec)?;
        let metadata = {
            let views = batch
                .iter()
                .map(|input| input.kv.read_view(0))
                .collect::<Result<Vec<_>, _>>()?;
            self.attention
                .prepare_metadata(rows, &views, &query_ranges, exec)?
        };
        // Pack directly into cached staging; retain u32 IDs without an intermediate Vec.
        let ids = self.input_uploads.upload_iter(
            batch
                .iter()
                .flat_map(|input| input.token_ids[input.kv.ready_len()..].iter().copied()),
            exec,
        )?;
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
                &metadata,
                false,
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
                &metadata,
                exec,
            )?;
            self.rope(
                &mut scratch.rotated_k,
                &scratch.k,
                layer.k_bias.clone(),
                &metadata,
                exec,
            )?;
            for (input, range) in batch.iter_mut().zip(&query_ranges) {
                input.kv.store(
                    index,
                    scratch.rotated_k.slice(&[range.clone()])?,
                    scratch.v.slice(&[range.clone()])?,
                    exec,
                )?;
            }
            // Preparation retains every context; all requests share this layer's backing storage.
            let view = batch[0].kv.read_view(index)?;
            self.attention.submit_prepared(
                &mut scratch.attention,
                &scratch.rotated_q,
                &view,
                &metadata,
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
                &metadata,
                false,
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
            &metadata,
            true,
            exec,
        )?;
        self.linear(
            &mut scratch.logits,
            &scratch.final_norm,
            self.lm_head.clone(),
            None,
            exec,
        )?;
        let blocks = self.config.vocab_size.div_ceil(256);
        kernels::argmax_blocks(
            (&mut scratch.maxima).partition([1, 1]),
            (&mut scratch.indices).partition([1, 1]),
            &scratch.logits,
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
            Some(enqueue_readback(
                scratch.logits,
                &self.logit_readbacks,
                exec,
            )?)
        } else {
            None
        };
        // All work is enqueued; the caller still awaits GPU/D2H completion.
        Ok(StepOutput {
            token,
            logits: values,
            host_submit_seconds: submit_started.elapsed().as_secs_f64(),
        })
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
        metadata: &Tensor<i32>,
        gather: bool,
        exec: &ExecutionContext,
    ) -> Result<(), ModelError> {
        let width = self.config.hidden_size;
        let last_rows_offset = x.shape()[0] * kernels::TOKEN_METADATA_WIDTH;
        kernels::rms_norm(
            out.partition([1, width.next_power_of_two()]),
            x,
            w,
            metadata,
            self.config.rms_norm_eps,
            width as i32,
            last_rows_offset,
        )
        .generics(vec![
            width.next_power_of_two().to_string(),
            gather.to_string(),
        ])
        .execute(exec)?;
        Ok(())
    }

    unsafe fn rope(
        &self,
        out: &mut Tensor<f32>,
        x: &Tensor<f32>,
        bias: Weight,
        metadata: &Tensor<i32>,
        exec: &ExecutionContext,
    ) -> Result<(), ModelError> {
        kernels::rope(
            out.partition([1, 64]),
            x,
            bias,
            self.inv_freq.clone(),
            metadata,
        )
        .execute(exec)?;
        Ok(())
    }
}

/// Compatible attention backend for flat [token, kv_head * 64] Naive storage.
struct GqaAttention {
    q_heads: usize,
    kv_heads: usize,
    metadata_uploads: Arc<UploadPool<i32>>,
}

impl GqaAttention {
    /// Per packed token: context row, absolute position, visible history start.
    /// The tail stores range.end - 1 for each request's final RMSNorm/LM-head row.
    /// Built/uploaded once per forward and reused by RoPE and every attention layer.
    unsafe fn prepare_metadata(
        &self,
        rows: usize,
        views: &[NaiveReadView<'_>],
        query_ranges: &[Range<usize>],
        exec: &ExecutionContext,
    ) -> Result<Arc<Tensor<i32>>, ModelError> {
        let first = views
            .first()
            .ok_or_else(|| ModelError::InvalidInput("empty GQA batch".into()))?;
        if views.len() != query_ranges.len()
            || rows
                .checked_mul(kernels::TOKEN_METADATA_WIDTH as usize)
                .and_then(|n| n.checked_add(views.len()))
                .is_none_or(|n| n > i32::MAX as usize)
        {
            return Err(ModelError::InvalidInput(
                "incompatible GQA batch geometry".into(),
            ));
        }
        let mut end = 0;
        for (view, range) in views.iter().zip(query_ranges) {
            if view.k_ptr() != first.k_ptr()
                || view.v_ptr() != first.v_ptr()
                || view.slot_table_ptr() != first.slot_table_ptr()
                || view.head_dim() != self.kv_heads * 64
                || range.start != end
                || range.is_empty()
                || range.end > rows
                || range.len() > view.positions().len()
            {
                return Err(ModelError::InvalidInput(
                    "GQA needs packed queries and compatible views from one pool".into(),
                ));
            }
            view.retain(exec)?;
            end = range.end;
        }
        if end != rows {
            return Err(ModelError::InvalidInput(
                "GQA ranges must cover every packed row".into(),
            ));
        }
        let tokens = views.iter().zip(query_ranges).flat_map(|(view, range)| {
            let start = view.positions().end - range.len();
            (0..range.len()).flat_map(move |local| {
                [
                    view.row() as i32,
                    (start + local) as i32,
                    view.positions().start as i32,
                ]
            })
        });
        Ok(self.metadata_uploads.upload_iter(
            tokens.chain(query_ranges.iter().map(|range| (range.end - 1) as i32)),
            exec,
        )?)
    }

    /// Preparation retains all contexts on exec; view supplies shared layer storage.
    unsafe fn submit_prepared(
        &self,
        out: &mut Tensor<f32>,
        q: &Tensor<f32>,
        view: &NaiveReadView<'_>,
        metadata: &Tensor<i32>,
        exec: &ExecutionContext,
    ) -> Result<(), ModelError> {
        view.retain(exec)?;
        let (k, v, table) = view.tensors();
        kernels::attention(
            out.partition([1, 64]),
            q,
            k.clone(),
            v.clone(),
            table.clone(),
            metadata,
            self.q_heads as i32,
            self.kv_heads as i32,
        )
        .generics(vec![KV_TOKENS_PER_TILE.to_string()])
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
        if q.shape().len() != 2 || q.shape()[1] as usize != self.q_heads * 64 {
            return Err(ModelError::InvalidInput(
                "incompatible GQA query width".into(),
            ));
        }
        // SAFETY: the attention kernel writes every valid output before any read.
        unsafe {
            let mut out = Tensor::<f32>::uninitialized(q.size())
                .execute(exec)?
                .assume_init()
                .reshape(&[q.shape()[0] as usize, self.q_heads * 64])?;
            let metadata =
                self.prepare_metadata(q.shape()[0] as usize, views, query_ranges, exec)?;
            self.submit_prepared(&mut out, &q, &views[0], &metadata, exec)?;
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
        let start = Instant::now();
        let batch_size = batch.len();
        if batch.is_empty() {
            return Vec::new();
        }
        let mut outputs: Vec<_> = batch
            .iter()
            .map(|input| self.validate_step_input(input).map(|()| 0))
            .collect();
        let result = if outputs.iter().any(Result::is_err) {
            // Keep valid peers packed together, preserving their original output slots.
            let mut valid: Vec<_> = batch
                .iter_mut()
                .zip(&outputs)
                .filter_map(|(input, result)| {
                    result.is_ok().then(|| ModelInput {
                        token_ids: input.token_ids,
                        kv: &mut *input.kv,
                    })
                })
                .collect();
            if valid.is_empty() {
                return outputs;
            }
            self.forward_batch(&mut valid, policy, false).await
        } else {
            self.forward_batch(batch, policy, false).await
        };
        match result {
            Ok(result) => {
                for (output, &token) in outputs
                    .iter_mut()
                    .filter(|output| output.is_ok())
                    .zip(result.token.as_slice())
                {
                    *output = self.checked_token(token);
                }
            }
            Err(error) => {
                for output in outputs.iter_mut().filter(|output| output.is_ok()) {
                    *output = Err(error.clone());
                }
            }
        }
        self.instruments
            .batch(batch_size, start.elapsed().as_secs_f64());
        tracing::debug!(target: "engine::qwen2", event = "batch_completed", batch_size,
            batch_wall_seconds = start.elapsed().as_secs_f64(), failed = outputs.iter().filter(|v| v.is_err()).count());
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
    use std::sync::Mutex;

    /// Small deterministic weights exercise the full multi-layer/GQA batch path without HF files.
    async fn tiny_model(policy: &Arc<dyn SchedulingPolicy>) -> Result<Qwen2Engine, ModelError> {
        with_context(|exec| {
            value((|| -> Result<_, ModelError> {
                let config: Qwen2Config = serde_json::from_value(serde_json::json!({
                    "model_type":"qwen2", "vocab_size":257, "hidden_size":128,
                    "intermediate_size":64, "num_hidden_layers":2,
                    "num_attention_heads":2, "num_key_value_heads":1,
                    "rms_norm_eps":1e-6, "rope_theta":1000000.0,
                    "max_position_embeddings":64, "tie_word_embeddings":true,
                    "eos_token_id":256, "hidden_act":"silu"
                }))
                .unwrap();
                let weight = |shape: &[usize], seed: usize| -> Result<Weight, ModelError> {
                    let values = (0..shape.iter().product())
                        .map(|index: usize| {
                            if seed == 0 {
                                1.0
                            } else {
                                ((index * 13 + seed * 7) % 59) as f32 * 0.003 - 0.087
                            }
                        })
                        .collect();
                    // SAFETY: all uploads share exec; construction awaits completion.
                    Ok(Arc::new(unsafe {
                        api::copy_host_vec_to_device(&Arc::new(values))
                            .execute(exec)?
                            .reshape(shape)?
                    }))
                };
                let embedding = weight(&[257, 128], 1)?;
                let mut layers = Vec::new();
                for seed in [2, 13] {
                    layers.push(Layer {
                        input_norm: weight(&[128], 0)?,
                        post_norm: weight(&[128], 0)?,
                        q: weight(&[128, 128], seed)?,
                        k: weight(&[64, 128], seed + 1)?,
                        v: weight(&[64, 128], seed + 2)?,
                        o: weight(&[128, 128], seed + 3)?,
                        q_bias: weight(&[128], seed + 4)?,
                        k_bias: weight(&[64], seed + 5)?,
                        v_bias: weight(&[64], seed + 6)?,
                        gate: weight(&[64, 128], seed + 7)?,
                        up: weight(&[64, 128], seed + 8)?,
                        down: weight(&[128, 64], seed + 9)?,
                    });
                }
                let frequencies: Vec<_> = (0..32)
                    .map(|i| config.rope_theta.powf(-(i as f32) / 32.0))
                    .collect();
                let workspaces: Arc<WorkspacePool> = Arc::default();
                Ok(Qwen2Engine {
                    instruments: Instruments::new(workspaces.metrics().model_instance),
                    workspaces,
                    attention: GqaAttention {
                        q_heads: 2,
                        kv_heads: 1,
                        metadata_uploads: Arc::default(),
                    },
                    tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
                    embedding: embedding.clone(),
                    lm_head: embedding,
                    norm: weight(&[128], 0)?,
                    inv_freq: Arc::new(unsafe {
                        api::copy_host_vec_to_device(&Arc::new(frequencies)).execute(exec)?
                    }),
                    layers,
                    config,
                    eos: vec![256],
                    input_uploads: Arc::default(),
                    token_readbacks: Default::default(),
                    logit_readbacks: Default::default(),
                })
            })())
        })
        .schedule(policy)?
        .await?
    }

    #[tokio::test]
    async fn gqa_matches_cpu_across_kv_block_boundaries_and_fragmented_slots(
    ) -> Result<(), ModelError> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = cuda_async::device_context::global_policy(0)?;
        let pool = NaiveKVPool::new(
            NaiveKVPoolConfig {
                num_layers: 1,
                num_slots: 320,
                max_rows: 2,
                max_seq_len: 160,
                head_dim: 128,
            },
            &policy,
        )
        .await?;
        let initial = pool.metrics();
        let attention = GqaAttention {
            q_heads: 4,
            kv_heads: 2,
            metadata_uploads: Arc::default(),
        };
        let readbacks = ReadbackCache::default();
        let queries = Arc::new(
            (0..5 * 256)
                .map(|i| ((i * 13 % 89) as f32 - 44.0) * 0.01)
                .collect::<Vec<_>>(),
        );
        let query_ranges = [0..3, 3..5];
        let history_starts = [5, 3];
        for length in [31, 32, 33, 63, 64, 65, 127, 129] {
            let lengths = [length, length + 5];
            let keys: Vec<_> = lengths
                .iter()
                .enumerate()
                .map(|(r, &len)| {
                    Arc::new(
                        (0..len * 128)
                            .map(|i| ((i * 7 + r * 11) % 97) as f32 * 0.003 - 0.144)
                            .collect::<Vec<_>>(),
                    )
                })
                .collect();
            let values: Vec<_> = lengths
                .iter()
                .enumerate()
                .map(|(r, &len)| {
                    Arc::new(
                        (0..len * 128)
                            .map(|i| ((i * 17 + r * 29) % 103) as f32 * 0.01 - 0.51)
                            .collect::<Vec<_>>(),
                    )
                })
                .collect();
            let mut contexts = [KVContext::new(pool.clone())?, KVContext::new(pool.clone())?];
            // Reserve A's first seven slots before B, so its later append is fragmented.
            with_context(|exec| {
                value((|| -> Result<(), ModelError> {
                    unsafe {
                        let k = api::copy_host_vec_to_device(&keys[0])
                            .execute(exec)?
                            .reshape(&[length, 128])?;
                        let v = api::copy_host_vec_to_device(&values[0])
                            .execute(exec)?
                            .reshape(&[length, 128])?;
                        contexts[0].prepare_append(7, exec)?;
                        contexts[0].store(0, k.slice(&[0..7])?, v.slice(&[0..7])?, exec)?;
                    }
                    Ok(())
                })())
            })
            .schedule(&policy)?
            .await??;
            contexts[0].commit_append()?;
            let output = with_context(|exec| {
                value((|| -> Result<_, ModelError> {
                    // Reserve B first; A's continuation cannot be contiguous with its prefix.
                    contexts[1].prepare_append(lengths[1], exec)?;
                    contexts[0].prepare_append(lengths[0] - 7, exec)?;
                    for (r, context) in contexts.iter_mut().enumerate() {
                        unsafe {
                            let k = api::copy_host_vec_to_device(&keys[r])
                                .execute(exec)?
                                .reshape(&[lengths[r], 128])?;
                            let v = api::copy_host_vec_to_device(&values[r])
                                .execute(exec)?
                                .reshape(&[lengths[r], 128])?;
                            let positions = context.pending_range().unwrap();
                            context.store(
                                0,
                                k.slice(&[positions.clone()])?,
                                v.slice(&[positions])?,
                                exec,
                            )?;
                        }
                    }
                    let views = contexts
                        .iter()
                        .enumerate()
                        .map(|(r, context)| {
                            pool.read_view(context.state(), 0, history_starts[r]..lengths[r])
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    // SAFETY: all producers/readers and D2H share exec, with leases retained.
                    unsafe {
                        let q = Arc::new(
                            api::copy_host_vec_to_device(&queries)
                                .execute(exec)?
                                .reshape(&[5, 256])?,
                        );
                        let out = attention.forward(q, &views, &query_ranges, exec)?;
                        Ok(enqueue_readback(out, &readbacks, exec)?)
                    }
                })())
            })
            .schedule(&policy)?
            .await??;
            for context in &mut contexts {
                context.commit_append()?;
            }
            for (r, range) in query_ranges.iter().enumerate() {
                for (local, packed_row) in range.clone().enumerate() {
                    let position = lengths[r] - range.len() + local;
                    for head in 0..4 {
                        let kv_head = head / 2;
                        let mut scores: Vec<f64> = (history_starts[r]..=position)
                            .map(|pos| {
                                (0..64)
                                    .map(|d| {
                                        f64::from(queries[packed_row * 256 + head * 64 + d])
                                            * f64::from(keys[r][pos * 128 + kv_head * 64 + d])
                                    })
                                    .sum::<f64>()
                                    * 0.125
                            })
                            .collect();
                        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        for score in &mut scores {
                            *score = (*score - maximum).exp();
                        }
                        let denominator: f64 = scores.iter().sum();
                        for d in 0..64 {
                            let expected: f64 = scores
                                .iter()
                                .enumerate()
                                .map(|(i, &weight)| {
                                    weight
                                        * f64::from(
                                            values[r]
                                                [(history_starts[r] + i) * 128 + kv_head * 64 + d],
                                        )
                                })
                                .sum::<f64>()
                                / denominator;
                            let actual = output.as_slice()[packed_row * 256 + head * 64 + d];
                            assert!((f64::from(actual) - expected).abs() < 2e-5,
                                "length={length}, request={r}, row={packed_row}, head={head}, dim={d}: {actual} vs {expected}");
                        }
                    }
                }
            }
            for context in contexts {
                context.release();
            }
            assert_eq!(pool.metrics(), initial);
        }
        Ok(())
    }

    #[tokio::test]
    async fn packed_qwen_matches_serial_ragged_history_decode_and_invalid_peers(
    ) -> Result<(), ModelError> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = cuda_async::device_context::global_policy(0)?;
        let mut model = tiny_model(&policy).await?;
        let recorder = metrics_util::debugging::DebuggingRecorder::new();
        model.instruments = metrics::with_local_recorder(&recorder, || {
            Instruments::new(model.workspace_metrics().model_instance)
        });
        let step_count = || {
            recorder
                .snapshotter()
                .snapshot()
                .into_vec()
                .into_iter()
                .filter_map(|(key, _, _, value)| {
                    if key.key().name() != "qwen2_steps_total" {
                        return None;
                    }
                    match value {
                        metrics_util::debugging::DebugValue::Counter(count) => Some(count),
                        _ => None,
                    }
                })
                .sum::<u64>()
        };
        let mut config = model.kv_pool_config(16)?;
        config.max_rows = 5;
        config.num_slots *= 5;
        let pool = NaiveKVPool::new(config, &policy).await?;
        let initial = pool.metrics();
        let mut contexts = (0..3)
            .map(|_| KVContext::new(pool.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut histories = [vec![1, 2, 3, 4, 5], vec![9, 8, 7], vec![23, 24, 25, 26]];
        // Different cached prefixes, including a fresh request and a one-token suffix.
        for ((history, context), prefix) in histories.iter().zip(&mut contexts).zip([2, 0, 3]) {
            if prefix > 0 {
                model
                    .step(&history[..prefix], context, policy.clone(), false)
                    .await?;
            }
        }
        for round in 0..3 {
            let mut inputs: Vec<_> = histories
                .iter()
                .zip(&mut contexts)
                .map(|(token_ids, kv)| ModelInput { token_ids, kv })
                .collect();
            // DebuggingRecorder snapshots drain counters; discard earlier forwards.
            let _ = step_count();
            let actual = model
                .forward_batch(&mut inputs, policy.clone(), true)
                .await?;
            assert_eq!(step_count(), 1, "one packed forward, not one per request");
            let logits = actual.logits.as_ref().unwrap().as_slice();
            for (index, (history, context)) in histories.iter().zip(&contexts).enumerate() {
                let mut serial = KVContext::new(pool.clone())?;
                // Full-history reference also checks cached-prefix and nonzero RoPE positions.
                let expected = model
                    .step(history, &mut serial, policy.clone(), true)
                    .await?;
                assert_eq!(
                    actual.token.as_slice()[index] as u32,
                    expected.token,
                    "round {round}, request {index}"
                );
                for (&a, b) in logits[index * 257..(index + 1) * 257]
                    .iter()
                    .zip(expected.logits.unwrap())
                {
                    assert!(a.is_finite() && (a - b).abs() < 1e-4, "{a} vs {b}");
                }
                assert_eq!(context.ready_len(), history.len());
                assert!(!context.is_pending());
                serial.release();
            }
            for (history, &token) in histories.iter_mut().zip(actual.token.as_slice()) {
                history.push(token as u32);
            }
        }
        // Reject the middle request, but still execute both valid peers in one packed batch.
        let mut invalid = KVContext::new(pool.clone())?;
        let (left, right) = contexts.split_at_mut(1);
        let _ = step_count();
        let outputs = model
            .next_tokens(
                &mut [
                    ModelInput {
                        token_ids: &histories[0],
                        kv: &mut left[0],
                    },
                    ModelInput {
                        token_ids: &[257],
                        kv: &mut invalid,
                    },
                    ModelInput {
                        token_ids: &histories[2],
                        kv: &mut right[1],
                    },
                ],
                policy.clone(),
            )
            .await;
        assert_eq!(step_count(), 1, "valid peers must stay in one batch");
        assert!(outputs[1].is_err());
        assert_eq!(invalid.ready_len(), 0);
        assert!(!invalid.is_pending());
        invalid.release();
        for (output, index) in [(&outputs[0], 0), (&outputs[2], 2)] {
            let mut serial = KVContext::new(pool.clone())?;
            let expected = model
                .step(&histories[index], &mut serial, policy.clone(), false)
                .await?;
            assert_eq!(*output.as_ref().unwrap(), expected.token);
            assert_eq!(contexts[index].ready_len(), histories[index].len());
            serial.release();
        }
        assert!(model.next_tokens(&mut [], policy).await.is_empty());
        for context in contexts {
            context.release();
        }
        assert_eq!(pool.metrics(), initial);
        assert_eq!(model.workspace_metrics().in_use_bytes, 0);
        Ok(())
    }

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
                    let ids = uploads.upload_iter([3, 0, 2], exec)?;
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
        let step_started = Instant::now();
        let output = model.step(&tokens, &mut full, policy.clone(), true).await?;
        assert!(output.host_submit_seconds.is_finite() && output.host_submit_seconds > 0.0);
        assert!(output.host_submit_seconds <= step_started.elapsed().as_secs_f64());
        let first = output.token;
        let actual = output.logits.unwrap();
        assert_eq!(first, expected_ids[0]);
        assert_eq!(actual.len(), expected.len());
        for (&a, &b) in actual.iter().zip(&expected) {
            assert!(
                a.is_finite() && (a - b).abs() < 1e-3,
                "logits differ: {a} vs {b}"
            );
        }
        assert_eq!(full.ready_len(), tokens.len());
        assert_eq!(model.workspace_metrics().in_use_bytes, 0);
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

        // Real-checkpoint packed prefill (ragged suffixes) followed by uniform decode.
        let mut packed_config = model.kv_pool_config(tokens.len() + 2)?;
        packed_config.max_rows = 3;
        packed_config.num_slots *= 3;
        let packed_pool = NaiveKVPool::new(packed_config, &policy).await?;
        let packed_idle = packed_pool.metrics();
        let mut contexts = [
            KVContext::new(packed_pool.clone())?,
            KVContext::new(packed_pool.clone())?,
        ];
        let mut histories = [tokens.clone(), tokens[..split].to_vec()];
        model
            .step(&tokens[..1], &mut contexts[0], policy.clone(), false)
            .await?;
        for _ in 0..2 {
            let mut inputs: Vec<_> = histories
                .iter()
                .zip(&mut contexts)
                .map(|(token_ids, kv)| ModelInput { token_ids, kv })
                .collect();
            let output = model
                .forward_batch(&mut inputs, policy.clone(), true)
                .await?;
            let logits = output.logits.as_ref().unwrap().as_slice();
            for (index, history) in histories.iter().enumerate() {
                let mut serial = KVContext::new(packed_pool.clone())?;
                let expected = model
                    .step(history, &mut serial, policy.clone(), true)
                    .await?;
                assert_eq!(output.token.as_slice()[index] as u32, expected.token);
                for (&a, b) in logits
                    [index * model.config.vocab_size..(index + 1) * model.config.vocab_size]
                    .iter()
                    .zip(expected.logits.unwrap())
                {
                    assert!(
                        a.is_finite() && (a - b).abs() < 1e-3,
                        "packed logit differs: {a} vs {b}"
                    );
                }
                assert_eq!(contexts[index].ready_len(), history.len());
                serial.release();
            }
            for (history, &token) in histories.iter_mut().zip(output.token.as_slice()) {
                history.push(token as u32);
            }
        }
        for context in contexts {
            context.release();
        }
        assert_eq!(packed_pool.metrics(), packed_idle);

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
        let observed = Arc::new(Mutex::new(Vec::new()));
        let captured = observed.clone();
        let scheduler = crate::scheduler::Scheduler::new_with_observers(
            cancel.child_token(),
            commands,
            reply_rx,
            vec![Arc::new(
                move |_id: u64,
                      stats: &crate::scheduler::RequestStats,
                      _result: &Result<(), crate::error::EngineError>| {
                    captured.lock().unwrap().push(*stats);
                },
            )],
        );
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
        let observed = observed.lock().unwrap();
        assert_eq!(observed.len(), 3);
        assert_eq!(observed[0].generated_tokens, Some(expected_ids.len()));
        assert_eq!(observed[1].generated_tokens, Some(turn_len));
        assert_eq!(observed[2].generated_tokens, Some(1));
        assert!(observed[0].time_to_first_text.is_some());
        assert!(observed[1].time_to_first_text.is_some());
        assert_eq!(
            observed[2].time_to_first_text.is_some(),
            !expected_text.is_empty()
        );
        for stats in observed.iter() {
            if let Some(first) = stats.time_to_first_text {
                assert!(first <= stats.elapsed);
            }
            assert!(stats.tokens_per_second().is_some_and(|tps| tps > 0.0));
        }
        assert_eq!(pool.metrics(), initial);
        Ok(())
    }
}
