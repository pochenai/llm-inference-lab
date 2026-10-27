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

use super::super::readback::{GpuTokenReadback, Readback, ReadbackCache};
use super::{
    kernels,
    metrics::{Instruments, WorkspaceMetrics},
    workspace::WorkspacePool,
};
use crate::runner::{
    kv_pool::{KVContext, NaiveKVPool, NaiveKVPoolConfig, NaiveReadView},
    loader::{EosTokenId, WeightElement, WeightLoader},
    upload::UploadPool,
};
use crate::{
    AttentionBackend, CausalModel, ModelError, ModelInput, ModelTokens, NextToken,
    SubmittedGpuWorkWithCompletion,
};
use cuda_async::{device_operation::*, scheduling_policies::SchedulingPolicy};
use cutile::{
    api,
    core::ElementType,
    tensor::{PartitionMut, Reshape, Tensor},
    tile_kernel::TileKernel,
    DType,
};
use futures::FutureExt;
use serde::Deserialize;
use std::{ffi::CStr, ops::Range, path::Path, sync::Arc, time::Instant};
use tokenizers::Tokenizer;

const KV_TOKENS_PER_TILE: i32 = 32;
// Keep narrow/decode work on the scalar path; tiled prefill reuses K/V across queries.
const ATTENTION_PREFILL_TILE: [usize; 2] = [32, 64];
const MIN_TILED_PREFILL_QUERIES: usize = 64;
const LINEAR_PREFILL_TILE: [usize; 2] = [16, 64];
const LINEAR_DECODE_TILE: [usize; 2] = [4, 32];

/// Finite shape buckets keep JIT specializations bounded and preserve small/FP32 tiles.
#[derive(Clone, Copy, Debug)]
struct LinearTile {
    rows: usize,
    columns: usize,
    reduction: usize,
}

impl LinearTile {
    fn for_shape<T: DType>(rows: usize) -> Self {
        if T::DTYPE == cutile::cuda_core::DTypeId::BF16 && rows >= 128 {
            Self {
                rows: 32,
                columns: 128,
                reduction: 64,
            }
        } else {
            let tile = if rows >= LINEAR_PREFILL_TILE[0] {
                LINEAR_PREFILL_TILE
            } else {
                LINEAR_DECODE_TILE
            };
            Self {
                rows: tile[0],
                columns: tile[1],
                reduction: 32,
            }
        }
    }
}

type Weight<T> = Arc<Tensor<T>>;

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

struct Layer<T: DType> {
    input_norm: Weight<T>,
    post_norm: Weight<T>,
    q: Weight<T>,
    k: Weight<T>,
    v: Weight<T>,
    o: Weight<T>,
    q_bias: Weight<T>,
    k_bias: Weight<T>,
    v_bias: Weight<T>,
    gate: Weight<T>,
    up: Weight<T>,
    down: Weight<T>,
}

/// T controls weight/activation/KV storage; reductions, MMA accumulation and logits use FP32.
/// BF16 and FP32 paths are validated; token IDs and Runner scheduling are dtype-independent.
pub struct Qwen2Engine<T: WeightElement + ElementType = f32> {
    config: Qwen2Config,
    tokenizer: Tokenizer,
    embedding: Weight<T>,
    norm: Weight<T>,
    lm_head: Weight<T>,
    inv_freq: Weight<f32>,
    layers: Vec<Layer<T>>,
    eos: Vec<u32>,
    attention: GqaAttention,
    workspaces: Arc<WorkspacePool<T>>,
    instruments: Instruments,
    input_uploads: Arc<UploadPool<u32>>,
    embedding_sources: Arc<UploadPool<u64>>,
    token_readback: GpuTokenReadback,
    logit_readbacks: ReadbackCache<f32>,
}

impl<T: WeightElement + ElementType> Qwen2Engine<T> {
    /// Load the supplied local checkpoint; missing or incompatible weights are errors.
    /// Checkpoint values load directly into T; BF16 -> BF16 preserves native bits.
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
        let workspaces: Arc<WorkspacePool<T>> = Arc::default();
        let instruments = Instruments::new(workspaces.metrics().model_instance);
        Ok(Self {
            token_readback: GpuTokenReadback::new(
                policy.next_stream()?.device(),
                instruments.readback_duration.clone(),
            )?,
            instruments,
            workspaces,
            input_uploads: Arc::default(),
            embedding_sources: Arc::default(),
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
        kv: &mut KVContext<NaiveKVPool<T>>,
        policy: Arc<dyn SchedulingPolicy>,
    ) -> Result<(u32, Vec<f32>), ModelError> {
        let output = self.step(tokens, kv, policy, true).await?;
        Ok((output.token, output.logits.unwrap()))
    }

    async fn step(
        &self,
        tokens: &[u32],
        kv: &mut KVContext<NaiveKVPool<T>>,
        policy: Arc<dyn SchedulingPolicy>,
        read_logits: bool,
    ) -> Result<StepOutput<u32, Vec<f32>>, ModelError> {
        let mut batch = [ModelInput {
            token_ids: (tokens).into(),
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

    fn validate_step_input(
        &self,
        input: &ModelInput<'_, NaiveKVPool<T>>,
    ) -> Result<(), ModelError> {
        let length = input.sequence_len();
        let kv = &input.kv;
        if kv.is_pending()
            || length <= kv.ready_len()
            || length > kv.max_seq_len()
            || length > self.config.max_position_embeddings
        {
            return Err(ModelError::InvalidInput(
                "expected valid tokens with an uncached suffix and idle context".into(),
            ));
        }
        match &input.token_ids {
            ModelTokens::BorrowedHost(_) | ModelTokens::SharedHost { .. } => {
                self.validate_input(&input.token_ids.host()?[kv.allocated_len()..])
            }
            ModelTokens::Device(token)
                if token.values.device_id() == self.embedding.device_id()
                    && token.vocab_size == self.config.vocab_size
                    && token.position == kv.ready_len()
                    && token.producer_stream_id.is_none() =>
            {
                Ok(())
            }
            ModelTokens::Device(_) => Err(ModelError::InvalidInput(
                "expected completed feedback at the context cursor; queued feedback uses enqueue_forward_and_return_completion".into(),
            )),
        }
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
        batch: &mut [ModelInput<'_, NaiveKVPool<T>>],
        policy: Arc<dyn SchedulingPolicy>,
        read_logits: bool,
    ) -> Result<StepOutput<Readback<i32>, Readback<f32>>, ModelError> {
        let rows = batch.iter().map(|input| input.uncached_len()).sum();
        let prefill = batch
            .iter()
            .any(|input| input.kv.ready_len() == 0 || input.uncached_len() > 1);
        let start = Instant::now();
        let result = Self::execute_append_batch(batch, &policy, |batch, exec| {
            // SAFETY: every GPU access, including D2H, belongs to this submission.
            unsafe { self.enqueue_forward_with_host_readback(batch, read_logits, exec) }
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

    /// Enqueue a forward and token/optional-logit D2H on the same exec.
    /// Caller must await this submission before accessing its host buffers.
    unsafe fn enqueue_forward_with_host_readback(
        &self,
        batch: &mut [ModelInput<'_, NaiveKVPool<T>>],
        read_logits: bool,
        exec: &ExecutionContext,
    ) -> Result<StepOutput<Readback<i32>, Readback<f32>>, ModelError> {
        let output = self.submit_forward_kernels(batch, exec)?;
        let token = self.token_readback.enqueue(output.token, exec)?;
        let logits = if read_logits {
            Some(self.logit_readbacks.enqueue(output.logits.unwrap(), exec)?)
        } else {
            None
        };
        Ok(StepOutput {
            token,
            logits,
            host_submit_seconds: output.host_submit_seconds,
        })
    }

    /// Enqueue packed Transformer layers, KV writes and argmax on exec's stream.
    /// Returned tensors are GPU handles; their values may still be pending.
    unsafe fn submit_forward_kernels(
        &self,
        batch: &mut [ModelInput<'_, NaiveKVPool<T>>],
        exec: &ExecutionContext,
    ) -> Result<StepOutput<Tensor<i32>, Tensor<f32>>, ModelError> {
        let submit_started = Instant::now();
        // Thread-local NVTX range covers synchronous submission only, never the await.
        let _range = crate::profiling::ScopedRange::new(
            if batch.iter().any(|input| {
                input
                    .kv
                    .pending_range()
                    .is_some_and(|range| range.start == 0)
                    || input.uncached_len() > 1
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
                rows += input.uncached_len();
                start..rows
            })
            .collect();
        // Check the entire packed geometry before allocating/uploading model inputs.
        let mut scratch = self
            .workspaces
            .acquire(rows, batch.len(), &self.config, exec)?;
        let metadata = {
            let _range = crate::profiling::ScopedRange::new(c"qwen2.metadata");
            let views = batch
                .iter()
                .map(|input| input.kv.read_view(0))
                .collect::<Result<Vec<_>, _>>()?;
            self.attention
                .prepare_metadata(rows, &views, &query_ranges, exec)?
        };
        {
            let _range = crate::profiling::ScopedRange::new(c"qwen2.embedding");
            // Keep host-only batches packed into one upload/embedding launch. Pure
            // decode can likewise reuse a contiguous previous GPU batch without packing.
            let contiguous_feedback = match &batch[0].token_ids {
            ModelTokens::Device(first) if batch.iter().enumerate().all(|(index, input)| {
                matches!(&input.token_ids, ModelTokens::Device(token)
                    if Arc::ptr_eq(&token.values, &first.values) && token.index == first.index + index)
            }) => Some(first),
            _ => None,
        };
            if batch
                .iter()
                .all(|input| !matches!(input.token_ids, ModelTokens::Device(_)))
            {
                let ids = self.input_uploads.upload_iter(
                    batch.iter().flat_map(|input| {
                        input.token_ids.host().unwrap()[input.kv.pending_range().unwrap().start..]
                            .iter()
                            .copied()
                    }),
                    exec,
                )?;
                kernels::embedding(
                    (&mut scratch.x).partition([1, 256]),
                    ids,
                    self.embedding.clone(),
                )
                .generics(vec![T::DTYPE.to_string()])
                .execute(exec)?;
            } else if let Some(first) = contiguous_feedback {
                // acquire retains the workspace lease on exec; this raw write covers
                // exactly the rows subsequently read by the first layer on the same stream.
                kernels::embedding_at(
                    scratch.x.device_pointer(),
                    first.values.clone(),
                    self.embedding.clone(),
                    first.index as i32,
                    0,
                    self.config.hidden_size as i32,
                    self.config.vocab_size as i32,
                )
                .grid((
                    batch.len() as u32,
                    self.config.hidden_size.div_ceil(256) as u32,
                    1,
                ))
                .generics(vec![T::DTYPE.to_string()])
                .execute(exec)?;
            } else {
                // Source addresses bypass typed kernel arguments, so retain their owners
                // explicitly. Adjacent feedback usually shares one previous batch buffer.
                let mut retained_source = None;
                for input in batch.iter() {
                    if let ModelTokens::Device(token) = &input.token_ids {
                        if retained_source
                            .is_none_or(|previous| !Arc::ptr_eq(previous, &token.values))
                        {
                            exec.retain(token.values.clone())?;
                            retained_source = Some(&token.values);
                        }
                    }
                }
                // Upload one addressing table, including inline host IDs. No per-request
                // uploads/launches and no extra GPU token-packing pass are required.
                let sources = self.embedding_sources.upload_iter(
                    batch.iter().flat_map(|input| {
                        (0..input.uncached_len()).flat_map(move |local| match &input.token_ids {
                            ModelTokens::BorrowedHost(_) | ModelTokens::SharedHost { .. } => [
                                0,
                                input.token_ids.host().unwrap()
                                    [input.kv.pending_range().unwrap().start + local]
                                    as u64,
                            ],
                            ModelTokens::Device(token) => [
                                token.values.device_pointer().cu_deviceptr()
                                    + (token.index * std::mem::size_of::<i32>()) as u64,
                                0,
                            ],
                        })
                    }),
                    exec,
                )?;
                kernels::embedding_gather(
                    (&mut scratch.x).partition([1, 256]),
                    sources,
                    self.embedding.clone(),
                    self.config.vocab_size as i32,
                )
                .generics(vec![T::DTYPE.to_string()])
                .execute(exec)?;
            }
        }
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
            self.linear(
                c"qwen2.linear.q",
                &mut scratch.q,
                &scratch.norm,
                layer.q.clone(),
                Some(layer.q_bias.clone()),
                exec,
            )?;
            self.linear(
                c"qwen2.linear.k",
                &mut scratch.k,
                &scratch.norm,
                layer.k.clone(),
                Some(layer.k_bias.clone()),
                exec,
            )?;
            self.linear(
                c"qwen2.linear.v",
                &mut scratch.v,
                &scratch.norm,
                layer.v.clone(),
                Some(layer.v_bias.clone()),
                exec,
            )?;
            self.rope(&mut scratch.rotated_q, &scratch.q, &metadata, exec)?;
            self.rope(&mut scratch.rotated_k, &scratch.k, &metadata, exec)?;
            {
                let _range = crate::profiling::ScopedRange::new(c"qwen2.kv_store");
                for (input, range) in batch.iter_mut().zip(&query_ranges) {
                    input.kv.store(
                        index,
                        scratch.rotated_k.slice(&[range.clone()])?,
                        scratch.v.slice(&[range.clone()])?,
                        exec,
                    )?;
                }
            }
            if GqaAttention::use_tiled_prefill::<T>(&query_ranges) {
                for (input, range) in batch.iter().zip(&query_ranges) {
                    self.attention.submit_prefill(
                        &mut scratch.attention,
                        &scratch.rotated_q,
                        &input.kv.read_view(index)?,
                        range,
                        exec,
                    )?;
                }
            } else {
                // Shared scalar launch handles decode, small and mixed suffixes unchanged.
                let view = batch[0].kv.read_view(index)?;
                self.attention.submit_prepared(
                    &mut scratch.attention,
                    &scratch.rotated_q,
                    &view,
                    &metadata,
                    exec,
                )?;
            }
            // Q projection has no remaining consumers: reuse its buffer for O/down.
            self.linear(
                c"qwen2.linear.o",
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
                c"qwen2.linear.gate",
                &mut scratch.gate,
                &scratch.norm,
                layer.gate.clone(),
                None,
                exec,
            )?;
            self.linear(
                c"qwen2.linear.up",
                &mut scratch.up,
                &scratch.norm,
                layer.up.clone(),
                None,
                exec,
            )?;
            {
                let _range = crate::profiling::ScopedRange::new(c"qwen2.swiglu");
                kernels::swiglu(
                    (&mut scratch.activated).partition([1, 256]),
                    &scratch.gate,
                    &scratch.up,
                )
                .generics(vec![T::DTYPE.to_string()])
                .execute(exec)?;
            }
            self.linear(
                c"qwen2.linear.down",
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
            c"qwen2.linear.lm_head",
            &mut scratch.logits,
            &scratch.final_norm,
            self.lm_head.clone(),
            None,
            exec,
        )?;
        let _argmax = crate::profiling::ScopedRange::new(c"qwen2.argmax");
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
        // GPU handles may escape after enqueue; consumers must observe stream order or completion.
        Ok(StepOutput {
            token: scratch.winner,
            logits: Some(scratch.logits),
            host_submit_seconds: submit_started.elapsed().as_secs_f64(),
        })
    }

    unsafe fn linear<O: DType>(
        &self,
        name: &'static CStr,
        out: &mut Tensor<O>,
        x: &Tensor<T>,
        w: Weight<T>,
        bias: Option<Weight<T>>,
        exec: &ExecutionContext,
    ) -> Result<(), ModelError> {
        let _range = crate::profiling::ScopedRange::new(name);
        let k = x.shape()[1] as usize;
        let has_bias = bias.is_some();
        // Bias-free specializations never read this existing rank-one weight.
        let bias = bias.unwrap_or_else(|| self.norm.clone());
        // Independent packed rows can share a tile even across request boundaries.
        let tile = LinearTile::for_shape::<T>(x.shape()[0] as usize);
        kernels::linear(out.partition([tile.rows, tile.columns]), x, w, bias)
            .generics(vec![
                T::DTYPE.to_string(),
                O::DTYPE.to_string(),
                k.to_string(),
                has_bias.to_string(),
                tile.rows.to_string(),
                tile.columns.to_string(),
                tile.reduction.to_string(),
            ])
            .execute(exec)?;
        Ok(())
    }

    unsafe fn normalize(
        &self,
        out: &mut Tensor<T>,
        x: &Tensor<T>,
        w: Weight<T>,
        metadata: &Tensor<i32>,
        gather: bool,
        exec: &ExecutionContext,
    ) -> Result<(), ModelError> {
        let width = self.config.hidden_size;
        let _range = crate::profiling::ScopedRange::new(c"qwen2.rms_norm");
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
            T::DTYPE.to_string(),
            width.next_power_of_two().to_string(),
            gather.to_string(),
        ])
        .execute(exec)?;
        Ok(())
    }

    unsafe fn rope(
        &self,
        out: &mut Tensor<T>,
        x: &Tensor<T>,
        metadata: &Tensor<i32>,
        exec: &ExecutionContext,
    ) -> Result<(), ModelError> {
        let _range = crate::profiling::ScopedRange::new(c"qwen2.rope");
        kernels::rope(out.partition([1, 64]), x, self.inv_freq.clone(), metadata)
            .generics(vec![T::DTYPE.to_string()])
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
    /// Dispatch by per-context query count, never total packed rows across requests.
    /// Mixed/small suffixes and FP32 retain the existing scalar path.
    fn use_tiled_prefill<T: DType>(ranges: &[Range<usize>]) -> bool {
        T::DTYPE == cutile::cuda_core::DTypeId::BF16
            && !ranges.is_empty()
            && ranges
                .iter()
                .all(|range| range.len() >= MIN_TILED_PREFILL_QUERIES)
    }

    /// Output storage is retained by the caller; this launch writes only this request's rows.
    /// Slot indirection and absolute causal positions support cached/chunked/ragged prefill.
    unsafe fn submit_prefill<T: DType>(
        &self,
        out: &mut Tensor<T>,
        q: &Tensor<T>,
        view: &NaiveReadView<'_, T>,
        queries: &Range<usize>,
        exec: &ExecutionContext,
    ) -> Result<(), ModelError> {
        let _range = crate::profiling::ScopedRange::new(c"qwen2.attention.prefill");
        view.retain(exec)?;
        let (k, v, table) = view.tensors();
        kernels::attention_prefill(
            out.device_pointer(),
            q,
            k.clone(),
            v.clone(),
            table.clone(),
            view.row() as i32,
            queries.start as i32,
            queries.len() as i32,
            (view.positions().end - queries.len()) as i32,
            view.positions().start as i32,
            self.q_heads as i32,
            self.kv_heads as i32,
        )
        .generics(vec![
            T::DTYPE.to_string(),
            ATTENTION_PREFILL_TILE[0].to_string(),
            ATTENTION_PREFILL_TILE[1].to_string(),
        ])
        .grid((
            queries.len().div_ceil(ATTENTION_PREFILL_TILE[0]) as u32,
            self.q_heads as u32,
            1,
        ))
        .execute(exec)?;
        Ok(())
    }

    /// Per packed token: context row, absolute position, visible history start.
    /// The tail stores range.end - 1 for each request's final RMSNorm/LM-head row.
    /// Built/uploaded once per forward and reused by RoPE and every attention layer.
    unsafe fn prepare_metadata<T: DType>(
        &self,
        rows: usize,
        views: &[NaiveReadView<'_, T>],
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
    unsafe fn submit_prepared<T: DType>(
        &self,
        out: &mut Tensor<T>,
        q: &Tensor<T>,
        view: &NaiveReadView<'_, T>,
        metadata: &Tensor<i32>,
        exec: &ExecutionContext,
    ) -> Result<(), ModelError> {
        let _range = crate::profiling::ScopedRange::new(c"qwen2.attention");
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
        .generics(vec![T::DTYPE.to_string(), KV_TOKENS_PER_TILE.to_string()])
        .execute(exec)?;
        Ok(())
    }
}

impl<T: DType + ElementType> AttentionBackend<NaiveKVPool<T>> for GqaAttention {
    fn forward(
        &self,
        q: Weight<T>,
        views: &[NaiveReadView<'_, T>],
        query_ranges: &[std::ops::Range<usize>],
        exec: &ExecutionContext,
    ) -> Result<Weight<T>, ModelError> {
        if q.shape().len() != 2 || q.shape()[1] as usize != self.q_heads * 64 {
            return Err(ModelError::InvalidInput(
                "incompatible GQA query width".into(),
            ));
        }
        // SAFETY: the attention kernel writes every valid output before any read.
        unsafe {
            let mut out = Tensor::<T>::uninitialized(q.size())
                .execute(exec)?
                .assume_init()
                .reshape(&[q.shape()[0] as usize, self.q_heads * 64])?;
            let metadata =
                self.prepare_metadata(q.shape()[0] as usize, views, query_ranges, exec)?;
            if Self::use_tiled_prefill::<T>(query_ranges) {
                // Raw output writes must survive an error before this handle is returned.
                // The alias is retained only; all accesses remain ordered on this exec.
                exec.retain(out.into_shared_alias())?;
                for (view, queries) in views.iter().zip(query_ranges) {
                    self.submit_prefill(&mut out, &q, view, queries, exec)?;
                }
            } else {
                self.submit_prepared(&mut out, &q, &views[0], &metadata, exec)?;
            }
            Ok(Arc::new(out))
        }
    }
}

impl<T: WeightElement + ElementType> CausalModel<NaiveKVPool<T>> for Qwen2Engine<T> {
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

    async fn forward(
        &self,
        batch: &mut [ModelInput<'_, NaiveKVPool<T>>],
        policy: Arc<dyn SchedulingPolicy>,
    ) -> Vec<Result<NextToken, ModelError>> {
        let start = Instant::now();
        if batch.is_empty() {
            return Vec::new();
        }
        let mut outputs: Vec<_> = batch
            .iter()
            .map(|input| self.validate_step_input(input).map(|()| NextToken::Host(0)))
            .collect();
        let rows = batch
            .iter()
            .zip(&outputs)
            .filter(|(_, output)| output.is_ok())
            .map(|(input, _)| input.uncached_len())
            .sum();
        let prefill = batch.iter().zip(&outputs).any(|(input, output)| {
            output.is_ok() && (input.kv.ready_len() == 0 || input.uncached_len() > 1)
        });
        let submit = |batch: &mut [ModelInput<'_, NaiveKVPool<T>>], exec: &ExecutionContext| {
            // SAFETY: all layers and their resources belong to this submission.
            unsafe { self.submit_forward_kernels(batch, exec) }
        };
        let result = if outputs.iter().any(Result::is_err) {
            // only execute non-error request
            let mut valid: Vec<_> = batch
                .iter_mut()
                .zip(&outputs)
                .filter_map(|(input, result)| {
                    result.is_ok().then(|| ModelInput {
                        token_ids: input.token_ids.clone(),
                        kv: &mut *input.kv,
                    })
                })
                .collect();
            if valid.is_empty() {
                return outputs;
            }
            Self::execute_append_batch(&mut valid, &policy, submit).await
        } else {
            Self::execute_append_batch(batch, &policy, submit).await
        };
        let host_submit_seconds = result
            .as_ref()
            .ok()
            .map(|output| output.host_submit_seconds);
        self.instruments.step(
            prefill,
            rows,
            start.elapsed().as_secs_f64(),
            host_submit_seconds,
        );
        tracing::debug!(target: "engine::qwen2", event = "step_completed",
            phase = if prefill { "prefill" } else { "decode" }, input_tokens = rows,
            batch_size = batch.len(), step_wall_seconds = start.elapsed().as_secs_f64(),
            host_submit_seconds, success = result.is_ok());
        match result {
            Ok(result) => {
                let positions = batch
                    .iter()
                    .zip(&outputs)
                    .filter(|(_, output)| output.is_ok())
                    .map(|(input, _)| input.kv.ready_len())
                    .collect();
                let tokens = self.token_readback.predictions(
                    result.token,
                    positions,
                    self.config.vocab_size,
                    futures::future::ready(Ok(())).boxed().shared(),
                    None,
                );
                for (output, token) in outputs
                    .iter_mut()
                    .filter(|output| output.is_ok())
                    .zip(tokens)
                {
                    *output = Ok(NextToken::Device(token));
                }
            }
            Err(error) => {
                for output in outputs.iter_mut().filter(|output| output.is_ok()) {
                    *output = Err(error.clone());
                }
            }
        }
        self.instruments
            .batch(batch.len(), start.elapsed().as_secs_f64());
        tracing::debug!(target: "engine::qwen2", event = "batch_completed", batch_size = batch.len(),
            batch_wall_seconds = start.elapsed().as_secs_f64(),
            failed = outputs.iter().filter(|output| output.is_err()).count());
        outputs
    }

    /// Enqueue one packed forward and restore KV owners before returning GPU handles.
    /// Runner waits for gpu_completion later, then commits KV and processes predictions.
    async fn enqueue_forward_and_return_completion(
        self: Arc<Self>,
        batch: &mut [ModelInput<'_, NaiveKVPool<T>>],
        policy: Arc<dyn SchedulingPolicy>,
    ) -> SubmittedGpuWorkWithCompletion {
        let mut outputs: Vec<_> = batch
            .iter()
            .map(|input| {
                let end = input.sequence_len();
                if end <= input.kv.allocated_len()
                    || end > input.kv.max_seq_len()
                    || end > self.config.max_position_embeddings
                {
                    return Err(ModelError::InvalidInput(
                        "invalid queued model input range".into(),
                    ));
                }
                match &input.token_ids {
                    ModelTokens::Device(token)
                        if token.position != input.kv.allocated_len()
                            || token.values.device_id() != self.embedding.device_id()
                            || token.vocab_size != self.config.vocab_size =>
                    {
                        Err(ModelError::InvalidInput(
                            "stale or incompatible GPU feedback".into(),
                        ))
                    }
                    ModelTokens::Device(_) => Ok(NextToken::Host(0)),
                    _ => self
                        .validate_input(&input.token_ids.host()?[input.kv.allocated_len()..])
                        .map(|()| NextToken::Host(0)),
                }
            })
            .collect();
        if outputs.iter().all(Result::is_err) {
            return SubmittedGpuWorkWithCompletion::from_completed_output(outputs);
        }
        let positions: Vec<_> = batch
            .iter()
            .zip(&outputs)
            .filter(|(_, result)| result.is_ok())
            .map(|(input, _)| input.sequence_len())
            .collect();
        let rows: usize = batch
            .iter()
            .zip(&outputs)
            .filter(|(_, result)| result.is_ok())
            .map(|(input, _)| input.uncached_len())
            .sum();
        let prefill = batch.iter().zip(&outputs).any(|(input, result)| {
            result.is_ok()
                && (!matches!(input.token_ids, ModelTokens::Device(_)) || input.uncached_len() > 1)
        });
        let selected = outputs
            .iter()
            .enumerate()
            .filter_map(|(index, result)| result.is_ok().then_some(index));
        let model = self.clone();
        let started = Instant::now();
        // The default trait method restores KV ownership; GPU completion is awaited later.
        let submitted =
            Self::enqueue_append_batch(batch, selected, &policy, move |inputs, exec| {
                // SAFETY: the common helper prepares KV and retains submission resources.
                unsafe { model.submit_forward_kernels(inputs, exec) }
            })
            .await;
        let SubmittedGpuWorkWithCompletion {
            output: result,
            gpu_completion,
            compute_stream_id,
        } = match submitted {
            Ok(submitted) => submitted,
            Err(error) => {
                for output in outputs.iter_mut().filter(|result| result.is_ok()) {
                    *output = Err(error.clone());
                }
                return SubmittedGpuWorkWithCompletion::from_completed_output(outputs);
            }
        };
        let batch_size = batch.len();
        let host_submit_seconds = result
            .as_ref()
            .ok()
            .map(|output| output.host_submit_seconds);
        let model = self.clone();
        let gpu_completion = async move {
            let result = gpu_completion.await;
            model.instruments.step(
                prefill,
                rows,
                started.elapsed().as_secs_f64(),
                if result.is_ok() {
                    host_submit_seconds
                } else {
                    None
                },
            );
            let batch_wall_seconds = started.elapsed().as_secs_f64();
            tracing::debug!(target: "engine::qwen2", event = "step_completed",
                phase = if prefill { "prefill" } else { "decode" }, input_tokens = rows,
                batch_size, step_wall_seconds = batch_wall_seconds, host_submit_seconds,
                success = result.is_ok() && host_submit_seconds.is_some());
            model.instruments.batch(batch_size, batch_wall_seconds);
            tracing::debug!(target: "engine::qwen2", event = "batch_completed",
                batch_size, batch_wall_seconds, success = result.is_ok() && host_submit_seconds.is_some());
            result
        }
        .boxed()
        .shared();
        match result {
            Ok(result) => {
                for (output, token) in outputs.iter_mut().filter(|result| result.is_ok()).zip(
                    self.token_readback.predictions(
                        result.token,
                        positions,
                        self.config.vocab_size,
                        gpu_completion.clone(),
                        compute_stream_id,
                    ),
                ) {
                    *output = Ok(NextToken::Device(token));
                }
            }
            Err(error) => {
                for output in outputs.iter_mut().filter(|result| result.is_ok()) {
                    *output = Err(error.clone());
                }
            }
        }
        SubmittedGpuWorkWithCompletion {
            output: outputs,
            gpu_completion,
            compute_stream_id,
        }
    }

    fn is_eos(&self, token: u32) -> bool {
        self.eos.contains(&token)
    }
}

async fn weight<T: WeightElement>(
    loader: &WeightLoader,
    name: &str,
    shape: &[usize],
    policy: &Arc<dyn SchedulingPolicy>,
) -> Result<Weight<T>, ModelError> {
    let w = loader.load_device::<T>(name, policy).await?;
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

unsafe fn add<T: DType>(
    out: &mut Tensor<T>,
    x: &Tensor<T>,
    y: &Tensor<T>,
    exec: &ExecutionContext,
) -> Result<(), ModelError> {
    let _range = crate::profiling::ScopedRange::new(c"qwen2.residual_add");
    kernels::add(out.partition([1, 256]), x, y)
        .generics(vec![T::DTYPE.to_string()])
        .execute(exec)?;
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
        tiny_model_with_dtype::<f32>(policy).await
    }

    async fn tiny_model_with_dtype<T: WeightElement + ElementType>(
        policy: &Arc<dyn SchedulingPolicy>,
    ) -> Result<Qwen2Engine<T>, ModelError> {
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
                let weight = |shape: &[usize], seed: usize| -> Result<Weight<T>, ModelError> {
                    let values: Vec<f32> = (0..shape.iter().product())
                        .map(|index: usize| {
                            if seed == 0 {
                                1.0
                            } else {
                                ((index * 13 + seed * 7) % 59) as f32 * 0.003 - 0.087
                            }
                        })
                        .collect();
                    let bytes: Vec<u8> =
                        values.iter().flat_map(|v: &f32| v.to_le_bytes()).collect();
                    let values = T::decode_weights(safetensors::Dtype::F32, &bytes)?;
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
                let workspaces: Arc<WorkspacePool<T>> = Arc::default();
                let instruments = Instruments::new(workspaces.metrics().model_instance);
                Ok(Qwen2Engine {
                    token_readback: GpuTokenReadback::new(
                        exec.get_cuda_stream().device(),
                        instruments.readback_duration.clone(),
                    )?,
                    instruments,
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
                    embedding_sources: Arc::default(),
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
        let pool = NaiveKVPool::<f32>::new(
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
                        Ok(readbacks.enqueue(out, exec)?)
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
        let pool = NaiveKVPool::<f32>::new(config, &policy).await?;
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
                .map(|(token_ids, kv)| ModelInput {
                    token_ids: token_ids.into(),
                    kv,
                })
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
                        token_ids: (&histories[0]).into(),
                        kv: &mut left[0],
                    },
                    ModelInput {
                        token_ids: (&[257]).into(),
                        kv: &mut invalid,
                    },
                    ModelInput {
                        token_ids: (&histories[2]).into(),
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
        // Cover all buckets with partial M/N/K, including K smaller than the reduction tile.
        for (rows, width) in [(3, 32), (35, 64), (33, 96)] {
            let input: Arc<Vec<f32>> = Arc::new(
                (0..rows * width)
                    .map(|i| ((i * 17 % 31) as f32 - 15.0) / 32.0)
                    .collect(),
            );
            let weights: Arc<Vec<f32>> = Arc::new(
                (0..96 * width)
                    .map(|i| ((i * 7 % 29) as f32 - 14.0) / 64.0)
                    .collect(),
            );
            let biases: Arc<Vec<f32>> = Arc::new((0..96).map(|i| i as f32 / 128.0).collect());
            for tile in [
                LinearTile {
                    rows: 4,
                    columns: 32,
                    reduction: 32,
                },
                LinearTile {
                    rows: 16,
                    columns: 64,
                    reduction: 32,
                },
                LinearTile {
                    rows: 32,
                    columns: 128,
                    reduction: 64,
                },
            ] {
                for has_bias in [false, true] {
                    let actual = with_context(|exec| {
                        value((|| -> Result<_, ModelError> {
                            // SAFETY: producers and D2H share exec; host reads follow await.
                            unsafe {
                                let x = api::copy_host_vec_to_device(&input)
                                    .execute(exec)?
                                    .reshape(&[rows, width])?;
                                let w = api::copy_host_vec_to_device(&weights)
                                    .execute(exec)?
                                    .reshape(&[96, width])?;
                                let bias = api::copy_host_vec_to_device(&biases).execute(exec)?;
                                let out = kernels::linear(
                                    Tensor::<f32>::uninitialized(rows * 96)
                                        .execute(exec)?
                                        .assume_init()
                                        .reshape(&[rows, 96])?
                                        .partition([tile.rows, tile.columns]),
                                    x,
                                    w,
                                    bias,
                                )
                                .generics(vec![
                                    "f32".into(),
                                    "f32".into(),
                                    width.to_string(),
                                    has_bias.to_string(),
                                    tile.rows.to_string(),
                                    tile.columns.to_string(),
                                    tile.reduction.to_string(),
                                ])
                                .map(|(out, ..)| out)
                                .unpartition()
                                .execute(exec)?;
                                Ok(cache.enqueue(out, exec)?)
                            }
                        })())
                    })
                    .schedule(&policy)?
                    .await??;
                    for (index, &value) in actual.as_slice().iter().enumerate() {
                        let row = index / 96;
                        let column = index % 96;
                        let expected: f32 = (0..width)
                            .map(|k| input[row * width + k] * weights[column * width + k])
                            .sum::<f32>()
                            + if has_bias { biases[column] } else { 0.0 };
                        assert!((value - expected).abs() < 1e-5,
                            "tile={tile:?}, rows={rows}, bias={has_bias}, index={index}: {value} vs {expected}");
                    }
                }
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
                    Ok((outputs.enqueue(out, exec)?, tokens.enqueue(winner, exec)?))
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

    #[tokio::test]
    async fn embedding_gather_handles_multiple_buffers_inline_ids_and_masked_columns(
    ) -> Result<(), ModelError> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = cuda_async::device_context::global_policy(0)?;
        let uploads = Arc::new(UploadPool::<u64>::default());
        let readbacks = ReadbackCache::default();
        for width in [128, 896] {
            let actual = with_context(|exec| {
                value((|| -> Result<Readback<f32>, ModelError> {
                    // SAFETY: uploads precede gather on exec, addressed sources
                    // are retained, and the outer await completes the final D2H.
                    unsafe {
                        let a = Arc::new(
                            api::copy_host_vec_to_device(&Arc::new(vec![7i32, 9, -1, i32::MAX]))
                                .execute(exec)?,
                        );
                        let b = Arc::new(
                            api::copy_host_vec_to_device(&Arc::new(vec![2i32, 4, 5]))
                                .execute(exec)?,
                        );
                        let pa = a.device_pointer().cu_deviceptr();
                        let pb = b.device_pointer().cu_deviceptr();
                        let word = std::mem::size_of::<i32>() as u64;
                        exec.retain(a)?;
                        exec.retain(b)?;
                        let sources = uploads.upload_iter(
                            [
                                pa + word,
                                0,
                                0,
                                3,
                                pb,
                                0,
                                pa,
                                0,
                                0,
                                6,
                                pb + 2 * word,
                                0,
                                pa + 2 * word,
                                0,
                                pa + 3 * word,
                                0,
                            ],
                            exec,
                        )?;
                        let weights = Arc::new(
                            api::copy_host_vec_to_device(&Arc::new(
                                (0..10 * width).map(|value| value as f32).collect(),
                            ))
                            .execute(exec)?
                            .reshape(&[10, width])?,
                        );
                        let mut out = Tensor::<f32>::uninitialized(8 * width)
                            .execute(exec)?
                            .assume_init()
                            .reshape(&[8, width])?;
                        kernels::embedding_gather(
                            (&mut out).partition([1, 256]),
                            sources,
                            weights,
                            10,
                        )
                        .execute(exec)?;
                        Ok(readbacks.enqueue(out, exec)?)
                    }
                })())
            })
            .schedule(&policy)?
            .await??;
            // Negative/invalid feedback clamps only to prevent an unsafe GPU
            // load; GpuTokenPredictionHandle::read still reports invalid sampled IDs separately.
            for (row, id) in [9, 3, 2, 7, 6, 5, 0, 9].into_iter().enumerate() {
                for column in 0..width {
                    assert_eq!(
                        actual.as_slice()[row * width + column],
                        (id * width + column) as f32
                    );
                }
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn gpu_feedback_matches_serial_with_reordering_and_new_prefill() -> Result<(), ModelError>
    {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = cuda_async::device_context::global_policy(0)?;
        let model = tiny_model(&policy).await?;
        let mut config = model.kv_pool_config(32)?;
        config.max_rows = 3;
        config.num_slots *= 3;
        let pool = NaiveKVPool::<f32>::new(config, &policy).await?;
        let idle = pool.metrics();
        let prompts = [vec![1, 2, 3], vec![9], vec![17, 18]];
        let mut expected = Vec::new();
        for prompt in &prompts {
            let mut history = prompt.clone();
            let mut kv = KVContext::new(pool.clone())?;
            let mut tokens = Vec::new();
            for _ in 0..3 {
                let token = model
                    .next_tokens(
                        &mut [ModelInput {
                            token_ids: (&history).into(),
                            kv: &mut kv,
                        }],
                        policy.clone(),
                    )
                    .await
                    .pop()
                    .unwrap()?;
                history.push(token);
                tokens.push(token);
            }
            expected.push(tokens);
        }
        let mut a = KVContext::new(pool.clone())?;
        let mut b = KVContext::new(pool.clone())?;
        let mut c = KVContext::new(pool.clone())?;
        let first = model
            .forward(
                &mut [
                    ModelInput {
                        token_ids: (&prompts[0]).into(),
                        kv: &mut a,
                    },
                    ModelInput {
                        token_ids: (&prompts[1]).into(),
                        kv: &mut b,
                    },
                ],
                policy.clone(),
            )
            .await;
        let first: Vec<_> = first
            .into_iter()
            .map(|output| match output.unwrap() {
                NextToken::Device(token) => token,
                NextToken::Host(_) => panic!("Qwen must return GPU feedback"),
            })
            .collect();
        let mut incompatible = first[0].clone();
        incompatible.vocab_size += 1;
        let rejected = model
            .forward(
                &mut [ModelInput {
                    token_ids: ModelTokens::Device(incompatible),
                    kv: &mut c,
                }],
                policy.clone(),
            )
            .await;
        assert!(matches!(&rejected[0], Err(ModelError::InvalidInput(_))));
        assert_eq!(c.ready_len(), 0);
        // Launch without reading first's tokens on CPU. Reverse A/B and admit C.
        let second = model
            .forward(
                &mut [
                    ModelInput {
                        token_ids: ModelTokens::Device(first[1].clone()),
                        kv: &mut b,
                    },
                    ModelInput {
                        token_ids: ModelTokens::Device(first[0].clone()),
                        kv: &mut a,
                    },
                    ModelInput {
                        token_ids: (&prompts[2]).into(),
                        kv: &mut c,
                    },
                ],
                policy.clone(),
            )
            .await;
        let stale = model
            .forward(
                &mut [ModelInput {
                    token_ids: ModelTokens::Device(first[0].clone()),
                    kv: &mut a,
                }],
                policy.clone(),
            )
            .await;
        assert!(
            stale[0].is_err(),
            "completed entry must reject stale feedback positions"
        );
        assert_eq!(first[0].read().await?, expected[0][0]);
        assert_eq!(first[1].read().await?, expected[1][0]);
        let second: Vec<_> = second
            .into_iter()
            .map(|output| match output.unwrap() {
                NextToken::Device(token) => token,
                NextToken::Host(_) => panic!("Qwen must return GPU feedback"),
            })
            .collect();
        let third = model
            .forward(
                &mut [
                    ModelInput {
                        token_ids: ModelTokens::Device(second[2].clone()),
                        kv: &mut c,
                    },
                    ModelInput {
                        token_ids: ModelTokens::Device(second[1].clone()),
                        kv: &mut a,
                    },
                ],
                policy,
            )
            .await;
        for (token, expected) in second
            .iter()
            .zip([expected[1][1], expected[0][1], expected[2][0]])
        {
            assert_eq!(token.read().await?, expected);
        }
        for (token, expected) in third.into_iter().zip([expected[2][1], expected[0][2]]) {
            assert_eq!(token?.read().await?, expected);
        }
        assert_eq!([a.ready_len(), b.ready_len(), c.ready_len()], [5, 2, 3]);
        drop((a, b, c, first, second));
        assert_eq!(pool.metrics(), idle);
        assert_eq!(model.workspace_metrics().in_use_bytes, 0);
        Ok(())
    }

    /// Real-checkpoint regression, explicitly enabled with local reference artifacts.
    #[tokio::test]
    async fn queued_gpu_feedback_matches_serial_and_commits_fifo() -> Result<(), ModelError> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let device =
            cutile::cuda_core::Device::new(0).map_err(cuda_async::error::DeviceError::from)?;
        let policy: Arc<dyn SchedulingPolicy> = Arc::new(SingleStream::new(&device)?);
        let model = Arc::new(tiny_model(&policy).await?);
        let mut config = model.kv_pool_config(32)?;
        config.max_rows = 3;
        config.num_slots *= 3;
        let pool = NaiveKVPool::<f32>::new(config, &policy).await?;
        let initial = pool.metrics();
        let prompts = [vec![1, 2, 3], vec![9], vec![17, 18]];
        let mut expected = Vec::new();
        for prompt in &prompts {
            let mut history = prompt.clone();
            let mut kv = KVContext::new(pool.clone())?;
            let mut result = Vec::new();
            for _ in 0..2 {
                let token = model
                    .next_tokens(
                        &mut [ModelInput {
                            token_ids: (&history).into(),
                            kv: &mut kv,
                        }],
                        policy.clone(),
                    )
                    .await
                    .pop()
                    .unwrap()?;
                history.push(token);
                result.push(token);
            }
            expected.push(result);
        }
        let mut a = KVContext::new(pool.clone())?;
        let mut b = KVContext::new(pool.clone())?;
        let mut c = KVContext::new(pool.clone())?;
        let first = model
            .clone()
            .enqueue_forward_and_return_completion(
                &mut [
                    ModelInput {
                        token_ids: (&prompts[0]).into(),
                        kv: &mut a,
                    },
                    ModelInput {
                        token_ids: (&prompts[1]).into(),
                        kv: &mut b,
                    },
                ],
                policy.clone(),
            )
            .await;
        let tokens: Vec<_> = first
            .output
            .iter()
            .map(|output| match output.as_ref().unwrap() {
                NextToken::Device(token) => token.clone(),
                _ => panic!("expected GPU feedback"),
            })
            .collect();
        assert_eq!((a.ready_len(), a.allocated_len()), (0, 3));
        assert_eq!(a.pending_range(), Some(0..3));
        // Common submission rejects cross-stream feedback before reserving another append.
        let other_policy: Arc<dyn SchedulingPolicy> = Arc::new(SingleStream::new(&device)?);
        let rejected = model
            .clone()
            .enqueue_forward_and_return_completion(
                &mut [ModelInput {
                    token_ids: ModelTokens::Device(tokens[0].clone()),
                    kv: &mut a,
                }],
                other_policy,
            )
            .await;
        assert!(
            matches!(&rejected.output[0], Err(ModelError::InvalidInput(message))
            if message == "queued feedback requires the same compute stream")
        );
        rejected.gpu_completion.await?;
        assert!(a.has_owned_pool_state());
        assert_eq!((a.ready_len(), a.allocated_len()), (0, 3));
        assert_eq!(a.pending_range(), Some(0..3));
        let second = model
            .clone()
            .enqueue_forward_and_return_completion(
                &mut [
                    ModelInput {
                        token_ids: ModelTokens::Device(tokens[1].clone()),
                        kv: &mut b,
                    },
                    ModelInput {
                        token_ids: ModelTokens::Device(tokens[0].clone()),
                        kv: &mut a,
                    },
                    ModelInput {
                        token_ids: (&prompts[2]).into(),
                        kv: &mut c,
                    },
                ],
                policy.clone(),
            )
            .await;
        for output in &second.output {
            assert!(output.is_ok(), "queued output failed");
        }
        assert_eq!((a.ready_len(), a.allocated_len()), (0, 4));
        assert_eq!(a.pending_range(), Some(3..4));
        assert!(
            a.commit_completed_append(4).is_err(),
            "cannot commit the later append first"
        );
        first.gpu_completion.clone().await?;
        a.commit_completed_append(3)?;
        b.commit_completed_append(1)?;
        assert!(a.is_pending());
        assert_eq!((a.ready_len(), a.allocated_len()), (3, 4));
        assert_eq!(a.pending_range(), Some(3..4));
        for (output, index) in first.output.iter().zip([0, 1]) {
            assert_eq!(
                output.as_ref().unwrap().clone().read().await?,
                expected[index][0]
            );
        }
        second.gpu_completion.clone().await?;
        a.commit_completed_append(4)?;
        b.commit_completed_append(2)?;
        c.commit_completed_append(2)?;
        assert_eq!((a.ready_len(), a.allocated_len()), (4, 4));
        assert_eq!(a.pending_range(), None);
        assert!(!a.is_pending());
        for (output, (index, step)) in second.output.iter().zip([(1, 1), (0, 1), (2, 0)]) {
            assert_eq!(
                output.as_ref().unwrap().clone().read().await?,
                expected[index][step]
            );
        }
        // Reusing an old feedback position must not append or disturb valid peers.
        let invalid = model
            .clone()
            .enqueue_forward_and_return_completion(
                &mut [ModelInput {
                    token_ids: ModelTokens::Device(tokens[0].clone()),
                    kv: &mut a,
                }],
                policy,
            )
            .await;
        assert!(invalid.output[0].is_err());
        assert_eq!(a.ready_len(), 4);
        drop((tokens, first, second, invalid));
        a.release();
        b.release();
        c.release();
        assert_eq!(pool.metrics(), initial);
        assert_eq!(model.workspace_metrics().in_use_bytes, 0);
        Ok(())
    }

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
        let model = Qwen2Engine::<f32>::load(Path::new(&model_dir), &policy).await?;
        let mut runner_config = model.kv_pool_config(tokens.len() + expected_ids.len() + 1)?;
        runner_config.max_rows = 2;
        runner_config.num_slots *= 2;
        let pool = NaiveKVPool::<f32>::new(runner_config, &policy).await?;
        let initial = pool.metrics();
        let mut full = KVContext::new(pool.clone())?;
        let invalid = model
            .next_tokens(
                &mut [ModelInput {
                    token_ids: (&[]).into(),
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
                    token_ids: (&tokens[..split]).into(),
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
                        token_ids: (&history).into(),
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
        let concurrent_pool = NaiveKVPool::<f32>::new(concurrent_config, &policy).await?;
        let idle = concurrent_pool.metrics();
        let other_policy: Arc<dyn SchedulingPolicy> = Arc::new(SingleStream::new(&device)?);
        let mut left = KVContext::new(concurrent_pool.clone())?;
        let mut right = KVContext::new(concurrent_pool.clone())?;
        let mut left_input = [ModelInput {
            token_ids: (&tokens).into(),
            kv: &mut left,
        }];
        let mut right_input = [ModelInput {
            token_ids: (&tokens[..split]).into(),
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
        let packed_pool = NaiveKVPool::<f32>::new(packed_config, &policy).await?;
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
                .map(|(token_ids, kv)| ModelInput {
                    token_ids: token_ids.into(),
                    kv,
                })
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
                    token_ids: (&continued).into(),
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
            // Force several Runner prefill chunks, including concurrent prompts
            // and a continuation with one previously sampled uncached token.
            .with_max_batch_tokens(std::num::NonZeroUsize::new(7).unwrap())
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
            assert_eq!(scheduler.execute(full.clone()).await?.output, streamed_text);
            // Distinct requests share Runner's dynamic packed batch and retain the
            // same output as the independently validated single-request path.
            let (left, right) =
                tokio::join!(scheduler.execute(full.clone()), scheduler.execute(full),);
            assert_eq!(left?.output, streamed_text);
            assert_eq!(right?.output, streamed_text);
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
        assert_eq!(observed.len(), 5);
        for stats in &observed[..3] {
            assert_eq!(stats.generated_tokens, Some(expected_ids.len()));
            assert!(stats.time_to_first_text.is_some());
        }
        assert_eq!(observed[3].generated_tokens, Some(turn_len));
        assert_eq!(observed[4].generated_tokens, Some(1));
        assert!(observed[3].time_to_first_text.is_some());
        assert_eq!(
            observed[4].time_to_first_text.is_some(),
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

    #[tokio::test]
    async fn bf16_ragged_cached_forward_matches_full_history_and_fp32() -> Result<(), ModelError> {
        use cutile::core::bf16;
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = cuda_async::device_context::global_policy(0)?;
        let model = tiny_model_with_dtype::<bf16>(&policy).await?;
        let reference = tiny_model(&policy).await?;
        assert_eq!(
            model.embedding.num_bytes() * 2,
            reference.embedding.num_bytes()
        );
        let mut config = model.kv_pool_config(64)?;
        config.max_rows = 6;
        config.num_slots *= 6;
        let pool = NaiveKVPool::<bf16>::new(config, &policy).await?;
        let reference_pool = NaiveKVPool::<f32>::new(config, &policy).await?;
        let initial = pool.metrics();
        assert_eq!(
            initial.layers.allocated_bytes * 2,
            reference_pool.metrics().layers.allocated_bytes
        );
        let mut histories = [vec![1u32; 33], vec![4, 9, 7], vec![3, 5, 8, 13, 21]];
        let mut contexts = (0..3)
            .map(|_| KVContext::new(pool.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        for (history, kv) in histories.iter().zip(&mut contexts) {
            model.step(&history[..2], kv, policy.clone(), false).await?;
        }
        for _ in 0..3 {
            let mut inputs: Vec<_> = histories
                .iter()
                .zip(&mut contexts)
                .map(|(ids, kv)| ModelInput {
                    token_ids: ids.into(),
                    kv,
                })
                .collect();
            let output = model
                .forward_batch(&mut inputs, policy.clone(), true)
                .await?;
            for (index, history) in histories.iter_mut().enumerate() {
                let mut full = KVContext::new(pool.clone())?;
                let expected = model.step(history, &mut full, policy.clone(), true).await?;
                let mut full_fp32 = KVContext::new(reference_pool.clone())?;
                let fp32 = reference
                    .step(history, &mut full_fp32, policy.clone(), true)
                    .await?;
                let logits =
                    &output.logits.as_ref().unwrap().as_slice()[index * 257..(index + 1) * 257];
                let error = logits
                    .iter()
                    .zip(expected.logits.unwrap())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(error < 0.01, "BF16 cached/full error: {error}");
                let error = logits
                    .iter()
                    .zip(fp32.logits.unwrap())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(error < 0.03, "BF16/FP32 error: {error}");
                assert_eq!(output.token.as_slice()[index] as u32, expected.token);
                history.push(output.token.as_slice()[index] as u32);
            }
        }
        drop(contexts);
        assert_eq!(pool.metrics(), initial);
        assert_eq!(model.workspace_metrics().in_use_bytes, 0);
        Ok(())
    }
    #[test]
    fn tiled_prefill_dispatch_uses_each_suffix_not_packed_batch_size() {
        use cutile::core::bf16;
        assert!(GqaAttention::use_tiled_prefill::<bf16>(&[0..67, 67..142]));
        assert!(!GqaAttention::use_tiled_prefill::<bf16>(&[
            0..32,
            32..64,
            64..96
        ]));
        assert!(!GqaAttention::use_tiled_prefill::<bf16>(&[0..1, 1..200]));
        assert!(!GqaAttention::use_tiled_prefill::<f32>(&[0..4096]));
        assert!(!GqaAttention::use_tiled_prefill::<bf16>(&[]));
    }

    #[tokio::test]
    async fn bf16_tiled_prefill_matches_cached_ragged_and_fp32_history() -> Result<(), ModelError> {
        use cutile::core::bf16;
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = cuda_async::device_context::global_policy(0)?;
        let mut model = tiny_model_with_dtype::<bf16>(&policy).await?;
        let mut reference = tiny_model(&policy).await?;
        model.config.max_position_embeddings = 256;
        reference.config.max_position_embeddings = 256;
        let mut config = model.kv_pool_config(256)?;
        config.max_rows = 6;
        config.num_slots *= 6;
        let pool = NaiveKVPool::<bf16>::new(config, &policy).await?;
        let reference_pool = NaiveKVPool::<f32>::new(config, &policy).await?;
        let initial = pool.metrics();
        let histories: Vec<Vec<u32>> = [99, 135]
            .into_iter()
            .enumerate()
            .map(|(r, n)| (0..n).map(|i| ((i * 13 + r * 17) % 251) as u32).collect())
            .collect();
        let mut contexts = [KVContext::new(pool.clone())?, KVContext::new(pool.clone())?];
        // Fragment A's slots with B's prefix; neither query range starts on a tile boundary.
        for ((ids, kv), prefix) in histories.iter().zip(&mut contexts).zip([7, 11]) {
            model
                .step(&ids[..prefix], kv, policy.clone(), false)
                .await?;
        }
        let mut inputs: Vec<_> = histories
            .iter()
            .zip(&mut contexts)
            .map(|(ids, kv)| ModelInput {
                token_ids: ids.into(),
                kv,
            })
            .collect();
        let output = model
            .forward_batch(&mut inputs, policy.clone(), true)
            .await?;
        drop(inputs);
        for (index, ids) in histories.iter().enumerate() {
            let mut full = KVContext::new(pool.clone())?;
            let expected = model.step(ids, &mut full, policy.clone(), true).await?;
            let mut fp32 = KVContext::new(reference_pool.clone())?;
            let fp32 = reference.step(ids, &mut fp32, policy.clone(), true).await?;
            let logits =
                &output.logits.as_ref().unwrap().as_slice()[index * 257..(index + 1) * 257];
            for (name, values, tolerance) in [
                ("cached/full", expected.logits.unwrap(), 0.015),
                ("bf16/fp32", fp32.logits.unwrap(), 0.03),
            ] {
                let error = logits
                    .iter()
                    .zip(values)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(error < tolerance, "{name}: {error}");
            }
            assert_eq!(contexts[index].ready_len(), ids.len());
            assert_eq!(output.token.as_slice()[index] as u32, expected.token);
        }
        drop(contexts);
        assert_eq!(pool.metrics(), initial);
        assert_eq!(model.workspace_metrics().in_use_bytes, 0);
        Ok(())
    }

    #[tokio::test]
    async fn bf16_prefill_writes_every_query_and_matches_independent_attention(
    ) -> Result<(), ModelError> {
        use cutile::core::bf16;
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = cuda_async::device_context::global_policy(0)?;
        let pool = NaiveKVPool::<bf16>::new(
            NaiveKVPoolConfig {
                num_layers: 1,
                num_slots: 512,
                max_rows: 1,
                max_seq_len: 512,
                head_dim: 128,
            },
            &policy,
        )
        .await?;
        let initial = pool.metrics();
        let mut context = KVContext::new(pool.clone())?;
        let queries = 3..260;
        let q: Arc<Vec<bf16>> = Arc::new(
            (0..263 * 256)
                .map(|i| bf16::from_f32(((i * 17 % 97) as f32 - 48.0) / 64.0))
                .collect(),
        );
        let k: Arc<Vec<bf16>> = Arc::new(
            (0..262 * 128)
                .map(|i| bf16::from_f32(((i * 7 % 61) as f32 - 30.0) / 64.0))
                .collect(),
        );
        let v: Arc<Vec<bf16>> = Arc::new(
            (0..262 * 128)
                .map(|i| bf16::from_f32(((i * 13 % 63) as f32 - 31.0) / 32.0))
                .collect(),
        );
        let attention = GqaAttention {
            q_heads: 4,
            kv_heads: 2,
            metadata_uploads: Arc::default(),
        };
        let cache = ReadbackCache::default();
        let actual = with_context(|exec| {
            value((|| -> Result<_, ModelError> {
                // SAFETY: all producers and reads use this exec; host observation follows completion.
                unsafe {
                    context.prepare_append(262, exec)?;
                    let keys = api::copy_host_vec_to_device(&k)
                        .execute(exec)?
                        .reshape(&[262, 128])?;
                    let values = api::copy_host_vec_to_device(&v)
                        .execute(exec)?
                        .reshape(&[262, 128])?;
                    context.store(0, keys.slice(&[])?, values.slice(&[])?, exec)?;
                    let q_gpu = api::copy_host_vec_to_device(&q)
                        .execute(exec)?
                        .reshape(&[263, 256])?;
                    let sentinel = Arc::new(vec![bf16::from_f32(4.0); 263 * 256]);
                    let mut out = api::copy_host_vec_to_device(&sentinel)
                        .execute(exec)?
                        .reshape(&[263, 256])?;
                    exec.retain(out.into_shared_alias())?;
                    // A nonzero history lower bound and an odd output offset exercise both masks.
                    let view = pool.read_view(context.state(), 0, 3..262)?;
                    attention.submit_prefill(&mut out, &q_gpu, &view, &queries, exec)?;
                    Ok(cache.enqueue(out, exec)?)
                }
            })())
        })
        .schedule(&policy)?
        .await??;
        for (index, value) in actual.as_slice()[..3 * 256].iter().enumerate() {
            assert_eq!(value.to_f32(), 4.0, "guard overwritten at {index}");
        }
        for value in &actual.as_slice()[260 * 256..] {
            assert_eq!(value.to_f32(), 4.0, "trailing output guard overwritten");
        }
        for local in 0..queries.len() {
            let visible = 3..=5 + local;
            for head in 0..4 {
                let scores: Vec<f64> = visible
                    .clone()
                    .map(|pos| {
                        (0..64)
                            .map(|d| {
                                q[(queries.start + local) * 256 + head * 64 + d].to_f32() as f64
                                    * k[pos * 128 + (head / 2) * 64 + d].to_f32() as f64
                            })
                            .sum::<f64>()
                            * 0.125
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
                let denominator = weights.iter().sum::<f64>();
                for d in 0..64 {
                    let expected = visible
                        .clone()
                        .zip(&weights)
                        .map(|(pos, w)| w * v[pos * 128 + (head / 2) * 64 + d].to_f32() as f64)
                        .sum::<f64>()
                        / denominator;
                    let value =
                        actual.as_slice()[(queries.start + local) * 256 + head * 64 + d].to_f32();
                    assert!(
                        (value as f64 - expected).abs() < 0.004,
                        "query {local}, head {head}, dim {d}: {value} vs {expected}"
                    );
                }
            }
        }
        context.commit_append()?;
        context.release();
        assert_eq!(pool.metrics(), initial);
        Ok(())
    }
}
