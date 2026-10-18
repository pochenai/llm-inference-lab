use anyhow::{ensure, Context};
use cutile::tensor::ToHostVec;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};
use std::{
    collections::HashMap,
    fs,
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokenizers::{
    decoders::fuse::Fuse, models::wordlevel::WordLevel,
    pre_tokenizers::whitespace::WhitespaceSplit, Tokenizer,
};

use cuda_async::{
    device_buffer::DevicePointer, device_context::global_policy, device_operation::*,
    scheduling_policies::SchedulingPolicy,
};
use cutile::{
    api,
    tensor::{Reshape, Tensor, Unpartition},
    tile_kernel::{PartitionOp, TileKernel, ToHostVecOp},
};

use super::{AttentionBackend, Model, ModelError, ModelInput, ModelOutput};
mod workspace;
#[cfg(test)]
use crate::runner::kv_pool::KVContext;
use crate::runner::kv_pool::{NaiveKVPool, NaiveReadView};
use crate::{
    runner::loader::{GenerationConfig, WeightLoader},
    types::{BatchIn, SamplingConfig},
};
use workspace::WorkspacePool;

pub const INPUT_DIM: usize = 16;
const TILE: usize = 4;
const WEIGHT_NAMES: [&str; 6] = [
    "layers.0.weight",
    "layers.1.weight",
    "embedding.weight",
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
];

fn weight_shapes() -> [Vec<usize>; 6] {
    [
        vec![16, 16],
        vec![16],
        vec![256, 16],
        vec![16, 16],
        vec![16, 16],
        vec![16, 16],
    ]
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct SimpleMLPConfig {
    pub model_type: String,
    pub architectures: Vec<String>,
    pub torch_dtype: String,
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub input_divisor: f32,
}

impl Default for SimpleMLPConfig {
    fn default() -> Self {
        Self {
            model_type: "simple_mlp_attn".into(),
            architectures: vec!["SimpleMLPEngine".into()],
            torch_dtype: "float32".into(),
            input_size: 16,
            hidden_size: 16,
            output_size: 1,
            input_divisor: 1.0,
        }
    }
}

/*
token_ids [T]
    ↓ Embedding [vocab_size, 16]
X [T, 16]
    ↓
Q = XWq, K = XWk, V = XWv
    ↓
将本次 K/V 追加到 KV cache
    ↓
A = causal_attention(Q, K_cache, V_cache)
    ↓
H = X + A
    ↓ 复用原来的两层 MLP
output = ReLU(H × W0) × W1
 */
pub struct SimpleMLPEngine {
    tokenizer: Tokenizer,
    config: SimpleMLPConfig,
    generation_config: Option<GenerationConfig>,
    // Used only for weight initialization/export; inference receives Runner policy.
    weight_policy: Arc<dyn SchedulingPolicy>,
    // MLP weights, embedding, and Q/K/V projection matrices (see WEIGHT_NAMES).
    layers: Vec<Arc<Tensor<f32>>>,
    workspaces: Arc<WorkspacePool>,
}

impl SimpleMLPEngine {
    fn default_model_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("models/simple_mlp")
    }

    pub async fn new(device_id: usize) -> Result<Self, ModelError> {
        Self::load(&Self::default_model_dir(), device_id).await
    }

    pub async fn load(model_dir: &Path, device_id: usize) -> Result<Self, ModelError> {
        if !model_dir
            .try_exists()
            .with_context(|| format!("failed to inspect {}", model_dir.display()))?
        {
            eprintln!(
                "WARN: model directory {} does not exist; initializing random weights on GPU {device_id}",
                model_dir.display()
            );
            return Self::random(device_id).await;
        }

        let result: anyhow::Result<Self> = async {
            let loader = WeightLoader::new(model_dir)?;
            let config: SimpleMLPConfig = loader.config()?;
            ensure!(
                config.model_type == "simple_mlp_attn"
                    && config.input_size == INPUT_DIM
                    && config.hidden_size == INPUT_DIM
                    && config.output_size == 1
                    && config.torch_dtype == "float32"
                    && config.input_divisor.is_finite()
                    && config.input_divisor > 0.0,
                "expected float32 simple_mlp_attn (16 -> 16 -> 1); re-export old MLP fixtures"
            );
            let tokenizer = loader.tokenizer()?;
            let generation_config = loader.generation_config()?;
            let policy = global_policy(device_id)?;
            let mut layers = Vec::new();
            for (name, shape) in WEIGHT_NAMES.iter().zip(weight_shapes()) {
                let weight = loader.load_device_f32(name, &policy).await?;
                ensure!(
                    weight
                        .shape()
                        .iter()
                        .map(|&dim| dim as usize)
                        .eq(shape.iter().copied()),
                    "unexpected shape for {name}: {:?}",
                    weight.shape()
                );
                layers.push(weight);
            }
            Ok(Self {
                tokenizer,
                config,
                generation_config,
                weight_policy: policy,
                layers,
                workspaces: Arc::default(),
            })
        }
        .await;
        result.map_err(Into::into)
    }

    pub fn config(&self) -> &SimpleMLPConfig {
        &self.config
    }

    pub fn generation_config(&self) -> Option<&GenerationConfig> {
        self.generation_config.as_ref()
    }

    /// Initialize weights on the GPU for the explicit export step, not inference startup.
    pub async fn random(device_id: usize) -> Result<Self, ModelError> {
        // A tiny demo vocabulary: numeric words map to their integer IDs.
        // A real LLM would load its matching tokenizer.json here.
        let mut vocab: HashMap<String, u32> = (0..256).map(|id| (id.to_string(), id)).collect();
        // [UNK] means "unknown": WordLevel uses this entry for words outside
        // the vocabulary. Its spelling and ID are configured here, not built in.
        vocab.insert("[UNK]".into(), 256);
        // Demo output tokens spell the numeric result; Fuse joins them into text.
        for (offset, symbol) in ["-", ".", "e", "+"].into_iter().enumerate() {
            vocab.insert(symbol.into(), 257 + offset as u32);
        }
        let word_level = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into()) // Select the fallback entry defined above.
            .build()
            .map_err(|error| ModelError::Execution(error.to_string()))?;
        let mut tokenizer = Tokenizer::new(word_level);
        tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
        tokenizer.with_decoder(Some(Fuse::new()));
        let policy = global_policy(device_id)?;
        // Generate weights on the GPU and wait before exposing them to forward kernels.
        let mut layers = Vec::new();
        for (index, shape) in weight_shapes().into_iter().enumerate() {
            layers.push(Arc::new(
                api::randn(
                    0.0f32,
                    0.25,
                    [shape.iter().product::<usize>()],
                    Some(42 + index as u64),
                )
                .schedule(&policy)?
                .await?
                .reshape(&shape)?,
            ));
        }
        Ok(Self {
            tokenizer,
            config: SimpleMLPConfig::default(),
            generation_config: Some(GenerationConfig {
                do_sample: Some(false),
                ..Default::default()
            }),
            weight_policy: policy,
            layers,
            workspaces: Arc::default(),
        })
    }

    /// Save a local Hugging Face style directory; the Rust model owns its computation graph.
    pub async fn save_pretrained(&self, model_dir: &Path) -> Result<(), ModelError> {
        let result: anyhow::Result<()> = async {
            fs::create_dir_all(model_dir)?;
            ensure!(
                !model_dir.join("model.safetensors.index.json").exists(),
                "refusing to export single-file weights over an existing shard index"
            );
            let mut buffers = Vec::new();
            for layer in &self.layers {
                let data = layer
                    .clone()
                    .to_host_vec()
                    .schedule(&self.weight_policy)?
                    .await?;
                buffers.push(
                    data.into_iter()
                        .flat_map(f32::to_le_bytes)
                        .collect::<Vec<_>>(),
                );
            }
            let tensors = self
                .layers
                .iter()
                .zip(&buffers)
                .enumerate()
                .map(|(index, (layer, bytes))| {
                    Ok((
                        WEIGHT_NAMES[index].to_string(),
                        TensorView::new(
                            Dtype::F32,
                            layer.shape().iter().map(|&dim| dim as usize).collect(),
                            bytes,
                        )?,
                    ))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            serialize_to_file(
                tensors,
                Some(HashMap::from([("format".into(), "pt".into())])),
                &model_dir.join("model.safetensors"),
            )?;
            fs::write(
                model_dir.join("config.json"),
                serde_json::to_string_pretty(&self.config)?,
            )?;
            if let Some(config) = &self.generation_config {
                fs::write(
                    model_dir.join("generation_config.json"),
                    serde_json::to_string_pretty(config)?,
                )?;
            }
            self.tokenizer
                .save(model_dir.join("tokenizer.json"), true)
                .map_err(|error| anyhow::anyhow!("failed to save tokenizer: {error}"))?;
            Ok(())
        }
        .await;
        result
            .context("failed to save MLP model directory")
            .map_err(Into::into)
    }
}

impl SimpleMLPEngine {
    /// Test helper using the same packed batch implementation for one history.
    #[cfg(test)]
    async fn forward(
        &self,
        token_ids: &[u32],
        kv: &mut KVContext<NaiveKVPool>,
        policy: Arc<dyn SchedulingPolicy>,
    ) -> Result<f32, ModelError> {
        Ok(self
            .forward_batch(vec![ModelInput { token_ids, kv }], policy)
            .await?[0])
    }

    /// Pack all uncached suffixes into one tensor. Shared projections and MLP run once;
    /// per-query metadata selects each request's KV row and causal history boundary.
    async fn forward_batch(
        &self,
        mut batch: Vec<ModelInput<'_, NaiveKVPool>>,
        policy: Arc<dyn SchedulingPolicy>,
    ) -> Result<Vec<f32>, ModelError> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }
        // Concatenate only uncached suffixes, then pad the final packed array.
        // A=[10,11,12,13], ready=2;
        // B=[20,21], ready=1.
        // With TILE=4: packed_token_ids=[12,13,21,0(PAD)], query_ranges=[0..2, 2..3].
        // Per-request rows in packed token/Q tensors and projected K/V, excluding padding;
        // these are not history positions or physical KV slot indices.
        let (packed_token_ids, query_ranges) = Self::pack_token_ids(&batch, TILE);
        let rows = packed_token_ids.len();
        if rows > i32::MAX as usize / INPUT_DIM {
            return Err(ModelError::InvalidInput("packed batch is too large".into()));
        }
        let values = Self::execute_append_batch(&mut batch, &policy, |batch, exec| {
            // SAFETY: one awaited future owns the whole submission. All buffers and
            // logical allocation leases are retained before their GPU accesses.
            unsafe {
                self.submit_batch(batch, Arc::new(packed_token_ids), &query_ranges, rows, exec)
            }
        })
        .await?;
        Ok(query_ranges
            .iter()
            .map(|range| values[range.end - 1])
            .collect())
    }

    /// Submit one packed batch; no host wait occurs between requests or layers.
    /// execute_append_batch has already reserved KV and enqueued mapping updates;
    /// this callback submits the model's computation on that same exec.
    unsafe fn submit_batch(
        &self,
        batch: &mut [ModelInput<'_, NaiveKVPool>],
        packed_token_ids: Arc<Vec<i32>>,
        query_ranges: &[Range<usize>],
        rows: usize,
        exec: &ExecutionContext,
    ) -> Result<Vec<f32>, ModelError> {
        let _range = crate::profiling::ScopedRange::new(c"submit_batch");
        let workspace = self.workspaces.acquire(rows, exec)?;
        let packed_token_ids = api::copy_host_vec_to_device(&packed_token_ids).execute(exec)?;
        let x = kernels::embedding(
            value(workspace.x).partition([1, INPUT_DIM]),
            packed_token_ids,
            self.layers[2].clone(),
            self.config.input_divisor,
        )
        .first()
        .unpartition()
        .execute(exec)?;
        let x = Arc::new(x);
        let q = {
            // Views retain cache buffers/mapping/leases on exec before the fused launch.
            // They borrow each context exclusively until this projection is submitted.
            let destinations = batch
                .iter_mut()
                .map(|input| input.kv.write_view(0, exec))
                .collect::<Result<Vec<_>, _>>()?;
            let first = &destinations[0];
            // Per packed row: context row and absolute token position in that context.
            // A negative context row marks padding, which must never write any KV slot.
            let mut metadata = vec![-1i32; rows * 2];
            for (dst, range) in destinations.iter().zip(query_ranges) {
                if dst.k_mut_ptr().cu_deviceptr() != first.k_mut_ptr().cu_deviceptr()
                    || dst.v_mut_ptr().cu_deviceptr() != first.v_mut_ptr().cu_deviceptr()
                    || dst.slot_table_ptr() != first.slot_table_ptr()
                    || dst.head_dim() != INPUT_DIM
                    || dst.positions().len() != range.len()
                {
                    return Err(ModelError::InvalidInput(
                        "projection batch needs compatible write views from the same pool".into(),
                    ));
                }
                for (local, packed_row) in range.clone().enumerate() {
                    metadata[packed_row * 2] = dst.row() as i32;
                    metadata[packed_row * 2 + 1] = (dst.positions().start + local) as i32;
                }
            }
            let metadata = api::copy_host_vec_to_device(&Arc::new(metadata))
                .execute(exec)?
                .reshape(&[rows, 2])?;
            // SAFETY: the views retain this mapping on exec, and the kernel only reads it.
            let table = DevicePointer::<i32>::from_cu_deviceptr(first.slot_table_ptr() as u64);
            kernels::qkv_to_cache(
                value(workspace.q).partition([TILE, TILE]),
                x.clone(),
                self.layers[3].clone(),
                self.layers[4].clone(),
                self.layers[5].clone(),
                first.k_mut_ptr(),
                first.v_mut_ptr(),
                table,
                metadata,
                first.table_width() as i32,
            )
            .generics(vec!["4".into(), "4".into(), "4".into()])
            .map(|(q, ..)| q)
            .unpartition()
            .execute(first.execution_context())?
        };
        let views = batch
            .iter()
            .map(|input| input.kv.read_view(0))
            .collect::<Result<Vec<_>, _>>()?;
        let attention = NaiveAttention.forward_into(
            Arc::new(q),
            &views,
            query_ranges,
            workspace.attention,
            exec,
        )?;
        let h = Arc::new(
            kernels::residual(
                value(workspace.residual).partition([1, INPUT_DIM]),
                x,
                attention,
            )
            .first()
            .unpartition()
            .execute(exec)?,
        );
        let hidden = kernels::gemm(
            value(workspace.hidden).partition([TILE, TILE]),
            h,
            self.layers[0].clone(),
        )
        .generics(vec!["4".into(), "4".into(), "4".into(), "16".into()])
        .first();
        let hidden = kernels::relu(hidden)
            .generics(vec!["4".into(), "4".into()])
            .map(|(hidden,)| hidden);
        Ok(kernels::matvec(
            value(workspace.output).partition([TILE]),
            hidden.unpartition(),
            self.layers[1].clone(),
        )
        .generics(vec!["4".into(), "4".into(), "16".into()])
        .first()
        .unpartition()
        .to_host_vec()
        .execute(exec)?)
    }
}

impl Model<NaiveKVPool> for SimpleMLPEngine {
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn validate_input(
        &self,
        token_ids: &[u32],
        sampling: &SamplingConfig,
    ) -> Result<(), ModelError> {
        if token_ids.iter().any(|&id| id >= 256) {
            return Err(ModelError::InvalidInput(
                "expected numeric input tokens in 0..256".into(),
            ));
        }
        if sampling.do_sample == Some(true)
            || sampling.temperature.is_some()
            || sampling.top_k.is_some()
            || sampling.top_p.is_some()
            || sampling.max_new_tokens.is_some()
            || sampling.eos_token_id.is_some()
        {
            return Err(ModelError::InvalidInput(
                "the scalar demo does not implement sampling or generation limits".into(),
            ));
        }
        Ok(())
    }

    async fn generate(
        &self,
        batch: BatchIn<ModelInput<'_, NaiveKVPool>>,
        policy: Arc<dyn SchedulingPolicy>,
    ) -> Vec<ModelOutput> {
        let mut request_ids = Vec::with_capacity(batch.len());
        let mut inputs = Vec::with_capacity(batch.len());
        for item in batch {
            request_ids.push(item.assigned_request_id());
            inputs.push(item.input);
        }
        let scalars = match self.forward_batch(inputs, policy).await {
            Ok(scalars) => scalars,
            Err(error) => {
                return request_ids
                    .into_iter()
                    .map(|request_id| ModelOutput {
                        request_id,
                        result: Err(error.clone()),
                    })
                    .collect()
            }
        };
        request_ids
            .into_iter()
            .zip(scalars)
            .map(|(request_id, scalar)| {
                let result = scalar
                    .to_string()
                    .chars()
                    .map(|character| {
                        self.tokenizer
                            .token_to_id(&character.to_string())
                            .ok_or_else(|| {
                                ModelError::Execution(format!("no output token for {character:?}"))
                            })
                    })
                    .collect();
                ModelOutput { request_id, result }
            })
            .collect()
    }
}

struct NaiveAttention;

impl AttentionBackend<NaiveKVPool> for NaiveAttention {
    /// Causal queries correspond to the trailing tokens of each logical read_range.
    /// Backend addressing is separate from KVContext's ready/pending lifecycle.
    fn forward(
        &self,
        q: Arc<Tensor<f32>>,
        views: &[NaiveReadView<'_>],
        query_ranges: &[Range<usize>],
        exec: &ExecutionContext,
    ) -> Result<Arc<Tensor<f32>>, ModelError> {
        // SAFETY: the caller owns exec and establishes completion.
        let out = unsafe { api::zeros::<f32>(&[q.shape()[0] as usize, INPUT_DIM]).execute(exec)? };
        self.forward_into(q, views, query_ranges, out, exec)
    }
}

impl NaiveAttention {
    fn forward_into(
        &self,
        q: Arc<Tensor<f32>>,
        views: &[NaiveReadView<'_>],
        query_ranges: &[Range<usize>],
        out: Tensor<f32>,
        exec: &ExecutionContext,
    ) -> Result<Arc<Tensor<f32>>, ModelError> {
        let first = views
            .first()
            .ok_or_else(|| ModelError::InvalidInput("empty attention batch".into()))?;
        if q.shape().len() != 2
            || q.shape()[1] != INPUT_DIM as i32
            || views.len() != query_ranges.len()
        {
            return Err(ModelError::InvalidInput(
                "attention requires packed 16-dimensional queries".into(),
            ));
        }
        let rows = q.shape()[0] as usize;
        // Per packed query: context row, logical history start, exclusive causal end.
        let mut metadata = vec![0i32; rows * 3];
        for (view, range) in views.iter().zip(query_ranges) {
            if view.k_ptr() != first.k_ptr()
                || view.v_ptr() != first.v_ptr()
                || view.slot_table_ptr() != first.slot_table_ptr()
                || view.head_dim() != INPUT_DIM
                || range.end > rows
                || range.len() > view.positions().len()
            {
                return Err(ModelError::InvalidInput(
                    "attention batch needs compatible views from the same pool".into(),
                ));
            }
            view.retain(exec)?;
            for (local, row) in range.clone().enumerate() {
                metadata[row * 3] = view.row() as i32;
                metadata[row * 3 + 1] = view.positions().start as i32;
                metadata[row * 3 + 2] = (view.positions().end - range.len() + local + 1) as i32;
            }
        }
        // SAFETY: the same submission has already updated every mapping and written KV.
        let out = unsafe {
            let metadata = api::copy_host_vec_to_device(&Arc::new(metadata))
                .execute(exec)?
                .reshape(&[rows, 3])?;
            let (k, v, table) = first.tensors();
            kernels::attention(
                value(out).partition([1, INPUT_DIM]),
                q,
                k.clone(),
                v.clone(),
                table.clone(),
                metadata,
            )
            .map(|(out, ..)| out)
            .unpartition()
            .execute(exec)?
        };
        Ok(Arc::new(out))
    }
}

// MLP kernels adapted from NVIDIA cutile-examples/examples/async_mlp.rs.
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub(super) fn embedding(
        out: &mut Tensor<f32, { [1, 16] }>,
        packed_token_ids: &Tensor<i32, { [-1] }>,
        weights: &Tensor<f32, { [-1, 16] }>,
        divisor: f32,
    ) {
        let i = program_id(0);
        let id: i32 = tile_to_scalar(
            packed_token_ids
                .partition(shape![1])
                .load([i])
                .reshape(shape![]),
        );
        out.store(
            weights.partition(shape![1, 16]).load([id, 0]) / divisor.broadcast(shape![1, 16]),
        );
    }

    /// Compute Q/K/V together; only Q is materialized as a temporary tensor.
    /// K/V accumulators scatter directly into mapped cache slots. Invalid packed rows
    /// are masked, so padding cannot overwrite another context or an unused slot.
    #[cutile::entry()]
    pub(super) unsafe fn qkv_to_cache<const BM: i32, const BN: i32, const BK: i32>(
        q: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, 16] }>,
        wq: &Tensor<f32, { [16, 16] }>,
        wk: &Tensor<f32, { [16, 16] }>,
        wv: &Tensor<f32, { [16, 16] }>,
        cache_k_ptr: *mut f32,
        cache_v_ptr: *mut f32,
        slot_table_base: *const i32,
        metadata: &Tensor<i32, { [-1, 2] }>,
        table_width: i32,
    ) {
        let pid = get_tile_block_id();
        let part_x = x.partition(shape![BM, BK]);
        let part_q = wq.partition(shape![BK, BN]);
        let part_k = wk.partition(shape![BK, BN]);
        let part_v = wv.partition(shape![BK, BN]);
        let mut q_values: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        let mut k_values: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        let mut v_values: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        for i in 0i32..(16 / BK) {
            let inputs = part_x.load([pid.0, i]);
            q_values = mma(inputs, part_q.load([i, pid.1]), q_values);
            k_values = mma(inputs, part_k.load([i, pid.1]), k_values);
            v_values = mma(inputs, part_v.load([i, pid.1]), v_values);
        }
        q.store(q_values);

        let context_rows: Tile<i32, { [BM] }> = metadata
            .partition(shape![BM, 1])
            .load([pid.0, 0])
            .reshape(shape![BM]);
        let positions: Tile<i32, { [BM] }> = metadata
            .partition(shape![BM, 1])
            .load([pid.0, 1])
            .reshape(shape![BM]);
        let padding_marker: Tile<i32, { [BM] }> = constant(-1i32, shape![BM]);
        let valid = lt_tile(padding_marker, context_rows);
        let table_offsets = context_rows * table_width.broadcast(shape![BM]) + positions;
        let tp: PointerTile<*const i32, { [] }> = pointer_to_tile(slot_table_base);
        let tp: PointerTile<*const i32, { [1] }> = tp.reshape(shape![1]);
        let tp: PointerTile<*const i32, { [BM] }> = tp.broadcast(shape![BM]);
        let tp: PointerTile<*const i32, { [BM] }> = tp.offset_tile(table_offsets);
        let (slots, _token): (Tile<i32, { [BM] }>, Token) = unsafe {
            load_ptr_tko(
                tp,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(valid),
                Some(0i32),
                None,
                Latency::<1>,
            )
        };
        let row_offsets = slots * 16i32.broadcast(shape![BM]);
        let row_offsets: Tile<i32, { [BM, BN] }> =
            row_offsets.reshape(shape![BM, 1]).broadcast(shape![BM, BN]);
        let columns: Tile<i32, { [BN] }> = iota(shape![BN]);
        let columns = columns + (pid.1 * BN).broadcast(shape![BN]);
        let columns: Tile<i32, { [BM, BN] }> =
            columns.reshape(shape![1, BN]).broadcast(shape![BM, BN]);
        let offsets = row_offsets + columns;
        let mask: Tile<bool, { [BM, BN] }> = valid.reshape(shape![BM, 1]).broadcast(shape![BM, BN]);
        let kp: PointerTile<*mut f32, { [] }> = pointer_to_tile(cache_k_ptr);
        let kp: PointerTile<*mut f32, { [1, 1] }> = kp.reshape(shape![1, 1]);
        let kp: PointerTile<*mut f32, { [BM, BN] }> = kp.broadcast(shape![BM, BN]);
        let kp: PointerTile<*mut f32, { [BM, BN] }> = kp.offset_tile(offsets);
        let vp: PointerTile<*mut f32, { [] }> = pointer_to_tile(cache_v_ptr);
        let vp: PointerTile<*mut f32, { [1, 1] }> = vp.reshape(shape![1, 1]);
        let vp: PointerTile<*mut f32, { [BM, BN] }> = vp.broadcast(shape![BM, BN]);
        let vp: PointerTile<*mut f32, { [BM, BN] }> = vp.offset_tile(offsets);
        // tko = Token-Ordered: these stores use computed addresses and value dependencies.
        unsafe {
            store_ptr_tko(
                kp,
                k_values,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<1>,
            );
            store_ptr_tko(
                vp,
                v_values,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<1>,
            );
        }
    }

    #[cutile::entry()]
    pub(super) fn attention(
        out: &mut Tensor<f32, { [1, 16] }>,
        q: &Tensor<f32, { [-1, 16] }>,
        k: &Tensor<f32, { [-1, 16] }>,
        v: &Tensor<f32, { [-1, 16] }>,
        table: &Tensor<i32, { [-1, -1] }>,
        metadata: &Tensor<i32, { [-1, 3] }>,
    ) {
        let i = program_id(0);
        let mut acc: Tile<f32, { [1, 16] }> = constant(0.0f32, shape![1, 16]);
        let row: i32 = tile_to_scalar(
            metadata
                .partition(shape![1, 1])
                .load([i, 0])
                .reshape(shape![]),
        );
        let history_start: i32 = tile_to_scalar(
            metadata
                .partition(shape![1, 1])
                .load([i, 1])
                .reshape(shape![]),
        );
        let history_end: i32 = tile_to_scalar(
            metadata
                .partition(shape![1, 1])
                .load([i, 2])
                .reshape(shape![]),
        );
        if history_end > history_start {
            let query = q.partition(shape![1, 16]).load([i, 0]);
            let mut maximum: Tile<f32, { [1, 1] }> = constant(f32::NEG_INFINITY, shape![1, 1]);
            let mut denominator: Tile<f32, { [1, 1] }> = constant(0.0f32, shape![1, 1]);
            for position in history_start..history_end {
                let slot: i32 = tile_to_scalar(
                    table
                        .partition(shape![1, 1])
                        .load([row, position])
                        .reshape(shape![]),
                );
                let key = k.partition(shape![1, 16]).load([slot, 0]);
                let dot: Tile<f32, { [1] }> = reduce_sum(query * key, 1);
                let score = dot.reshape(shape![1, 1]) * constant(0.25f32, shape![1, 1]);
                let next_max = max_tile(maximum, score);
                let alpha = exp(maximum - next_max);
                let probability = exp(score - next_max);
                acc = acc * alpha.broadcast(shape![1, 16])
                    + v.partition(shape![1, 16]).load([slot, 0])
                        * probability.broadcast(shape![1, 16]);
                denominator = denominator * alpha + probability;
                maximum = next_max;
            }
            acc = true_div(acc, denominator.broadcast(shape![1, 16]));
        }
        out.store(acc);
    }

    #[cutile::entry()]
    pub(super) fn residual(
        out: &mut Tensor<f32, { [1, 16] }>,
        x: &Tensor<f32, { [-1, 16] }>,
        attention: &Tensor<f32, { [-1, 16] }>,
    ) {
        let i = program_id(0);
        out.store(
            x.partition(shape![1, 16]).load([i, 0])
                + attention.partition(shape![1, 16]).load([i, 0]),
        );
    }

    #[cutile::entry()]
    pub(super) fn gemm<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        y: &Tensor<f32, { [K, -1] }>,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_y = y.partition(shape![BK, BN]);
        let pid = get_tile_block_id();
        let mut tile_z: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        for i in 0i32..(K / BK) {
            tile_z = mma(part_x.load([pid.0, i]), part_y.load([i, pid.1]), tile_z);
        }
        z.store(tile_z);
    }

    #[cutile::entry()]
    pub(super) fn relu<const BM: i32, const BN: i32>(z: &mut Tensor<f32, { [BM, BN] }>) {
        let input = z.load();
        let zero = constant(0.0f32, shape![BM, BN]);
        z.store(max_tile(zero, input));
    }

    #[cutile::entry()]
    pub(super) fn matvec<const BM: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<f32, { [BM] }>,
        x: &Tensor<f32, { [-1, K] }>,
        y: &Tensor<f32, { [K] }>,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_y = y.partition(shape![BK]);
        let pid = get_tile_block_id();
        let mut tile_z: Tile<f32, { [BM, 1] }> = constant(0.0f32, shape![BM, 1]);
        for i in 0i32..(K / BK) {
            tile_z = mma(
                part_x.load([pid.0, i]),
                part_y.load([i]).reshape(shape![BK, 1]),
                tile_z,
            );
        }
        z.store(tile_z.reshape(shape![BM]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::kv_pool::{KVPool, NaiveKVPoolConfig};

    fn close(actual: f32, expected: f32) {
        assert!(actual.is_finite() && expected.is_finite());
        assert!(
            (actual - expected).abs() <= 2e-6 + 3e-3 * expected.abs(),
            "GPU={actual}, reference={expected}"
        );
    }

    async fn pool(policy: &Arc<dyn SchedulingPolicy>) -> Result<Arc<NaiveKVPool>, ModelError> {
        Ok(NaiveKVPool::new(
            NaiveKVPoolConfig {
                num_slots: 64,
                max_rows: 4,
                max_seq_len: 32,
                ..Default::default()
            },
            policy,
        )
        .await?)
    }

    fn cpu_reference(tokens: &[u32], weights: &[Vec<f32>], divisor: f32) -> f32 {
        let embeddings: Vec<Vec<f32>> = tokens
            .iter()
            .map(|&id| {
                weights[2][id as usize * 16..(id as usize + 1) * 16]
                    .iter()
                    .map(|v| v / divisor)
                    .collect()
            })
            .collect();
        let project = |x: &[f32], w: &[f32]| -> Vec<f32> {
            (0..16)
                .map(|j| (0..16).map(|i| x[i] * w[i * 16 + j]).sum())
                .collect()
        };
        let q = project(embeddings.last().unwrap(), &weights[3]);
        let k: Vec<_> = embeddings.iter().map(|x| project(x, &weights[4])).collect();
        let v: Vec<_> = embeddings.iter().map(|x| project(x, &weights[5])).collect();
        let scores: Vec<f32> = k
            .iter()
            .map(|key| q.iter().zip(key).map(|(q, k)| q * k).sum::<f32>() * 0.25)
            .collect();
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let probabilities: Vec<f32> = scores.iter().map(|s| (s - maximum).exp()).collect();
        let denominator: f32 = probabilities.iter().sum();
        let h: Vec<f32> = (0..16)
            .map(|j| {
                embeddings.last().unwrap()[j]
                    + probabilities
                        .iter()
                        .zip(&v)
                        .map(|(p, v)| p * v[j])
                        .sum::<f32>()
                        / denominator
            })
            .collect();
        project(&h, &weights[0])
            .iter()
            .zip(&weights[1])
            .map(|(h, w)| h.max(0.0) * w)
            .sum()
    }

    /// Actual cached GPU data must match the independent dense CPU attention reference.
    #[tokio::test]
    async fn attention_cache_matches_cpu_full_chunked_and_decode() -> Result<(), ModelError> {
        let _guard = crate::runner::GPU_TEST_LOCK.lock().await;
        let model = SimpleMLPEngine::random(0).await?;
        let policy = model.weight_policy.clone();
        let pool = pool(&policy).await?;
        let initial = pool.metrics();
        let mut weights = Vec::new();
        for w in &model.layers {
            weights.push(w.clone().to_host_vec().schedule(&policy)?.await?);
        }
        let tokens = [2, 9, 1, 17, 4, 8, 100];
        let expected = cpu_reference(&tokens, &weights, model.config.input_divisor);
        let mut full = KVContext::new(pool.clone())?;
        let mut chunks = KVContext::new(pool.clone())?;
        let mut decode = KVContext::new(pool.clone())?;
        let a = model.forward(&tokens, &mut full, policy.clone()).await?;
        let first = model
            .forward(&tokens[..3], &mut chunks, policy.clone())
            .await?;
        close(
            first,
            cpu_reference(&tokens[..3], &weights, model.config.input_divisor),
        );
        let b = model.forward(&tokens, &mut chunks, policy.clone()).await?;
        let mut c = 0.0;
        for end in 1..=tokens.len() {
            c = model
                .forward(&tokens[..end], &mut decode, policy.clone())
                .await?;
            close(
                c,
                cpu_reference(&tokens[..end], &weights, model.config.input_divisor),
            );
            assert_eq!(decode.ready_len(), end);
            assert!(!decode.is_pending());
        }
        close(a, expected);
        close(b, expected);
        close(c, expected);
        // Reading only the new token would produce a different result.
        let mut isolated = KVContext::new(pool.clone())?;
        let last_only = model
            .forward(&tokens[6..], &mut isolated, policy.clone())
            .await?;
        assert!(
            (a - last_only).abs() > 1e-6,
            "history must affect attention output"
        );
        assert_eq!(
            pool.metrics().layers.assigned_bytes,
            (tokens.len() * 3 + 1) * 2 * 16 * 4
        );
        full.release();
        chunks.release();
        decode.release();
        isolated.release();
        assert_eq!(pool.metrics(), initial);
        // Reuse rows and slots with a different history; stale table entries are invisible.
        let mut reused = KVContext::new(pool.clone())?;
        close(
            model.forward(&tokens[..2], &mut reused, policy).await?,
            cpu_reference(&tokens[..2], &weights, model.config.input_divisor),
        );
        drop(reused);
        assert_eq!(pool.metrics(), initial);
        Ok(())
    }

    /// Check every fused K/V element, old prefixes, mixed context rows, and masked padding.
    #[tokio::test]
    async fn fused_projection_writes_only_mapped_tokens() -> Result<(), ModelError> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let model = SimpleMLPEngine::random(0).await?;
        let policy = model.weight_policy.clone();
        let pool = pool(&policy).await?;
        let initial = pool.metrics();
        let mut weights = Vec::new();
        for weight in &model.layers {
            weights.push(weight.clone().to_host_vec().schedule(&policy)?.await?);
        }
        let mut contexts = [
            KVContext::new(pool.clone())?,
            KVContext::new(pool.clone())?,
            KVContext::new(pool.clone())?,
        ];
        model
            .forward(&[1, 2, 3], &mut contexts[0], policy.clone())
            .await?;
        model
            .forward(&[7], &mut contexts[1], policy.clone())
            .await?;
        let histories: [&[u32]; 3] = [&[1, 2, 3, 4, 5], &[7, 8], &[9, 10]];
        // Suffixes have lengths 2, 1, 2: five real packed rows and three padding rows.
        let inputs = contexts
            .iter_mut()
            .zip(histories)
            .map(|(kv, token_ids)| ModelInput { token_ids, kv })
            .collect();
        let outputs = model.forward_batch(inputs, policy.clone()).await?;
        let mut used_slots = std::collections::HashSet::new();
        let mut cache = None;
        for (index, (kv, tokens)) in contexts.iter().zip(histories).enumerate() {
            close(
                outputs[index],
                cpu_reference(tokens, &weights, model.config.input_divisor),
            );
            let view = kv.read_view(0)?;
            // SAFETY: readbacks only read completed KV, and view retains its context lease.
            let (k, v, table) = unsafe { view.tensors() };
            let indices = table.clone().to_host_vec().schedule(&policy)?.await?;
            let keys = k.clone().to_host_vec().schedule(&policy)?.await?;
            let values = v.clone().to_host_vec().schedule(&policy)?.await?;
            for (position, &token) in tokens.iter().enumerate() {
                let slot = indices[view.row() * view.table_width() + position] as usize;
                assert!(used_slots.insert(slot));
                for column in 0..INPUT_DIM {
                    let expected_k: f32 = (0..INPUT_DIM)
                        .map(|i| {
                            weights[2][token as usize * INPUT_DIM + i] / model.config.input_divisor
                                * weights[4][i * INPUT_DIM + column]
                        })
                        .sum();
                    let expected_v: f32 = (0..INPUT_DIM)
                        .map(|i| {
                            weights[2][token as usize * INPUT_DIM + i] / model.config.input_divisor
                                * weights[5][i * INPUT_DIM + column]
                        })
                        .sum();
                    close(keys[slot * INPUT_DIM + column], expected_k);
                    close(values[slot * INPUT_DIM + column], expected_v);
                }
            }
            cache = Some((keys, values));
        }
        let (keys, values) = cache.unwrap();
        for slot in 0..pool.config().num_slots {
            if !used_slots.contains(&slot) {
                assert!(keys[slot * INPUT_DIM..(slot + 1) * INPUT_DIM]
                    .iter()
                    .all(|&v| v == 0.0));
                assert!(values[slot * INPUT_DIM..(slot + 1) * INPUT_DIM]
                    .iter()
                    .all(|&v| v == 0.0));
            }
        }
        drop(contexts);
        assert_eq!(pool.metrics(), initial);
        Ok(())
    }

    #[tokio::test]
    async fn attention_reads_logical_windows_without_pending_append() -> Result<(), ModelError> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let model = SimpleMLPEngine::random(0).await?;
        let policy = model.weight_policy.clone();
        let pool = pool(&policy).await?;
        let mut kv = KVContext::new(pool.clone())?;
        model
            .forward(&[1, 2, 8, 7, 4], &mut kv, policy.clone())
            .await?;
        assert!(!kv.is_pending());
        let (indices, cached_values) = {
            let view = kv.read_view(0)?;
            // SAFETY: only reading completed KV; the view lives through both readbacks.
            let (_, v, table) = unsafe { view.tensors() };
            (
                table.clone().to_host_vec().schedule(&policy)?.await?,
                v.clone().to_host_vec().schedule(&policy)?.await?,
            )
        };
        let q = Arc::new(
            api::zeros::<f32>(&[1, INPUT_DIM])
                .schedule(&policy)?
                .await?,
        );
        for window in [0..5, 2..5] {
            let values = with_context(|exec| {
                value((|| -> Result<Vec<f32>, ModelError> {
                    let view = pool.read_view(kv.state(), 0, window.clone())?;
                    let out = NaiveAttention.forward(q.clone(), &[view], &[0..1], exec)?;
                    // SAFETY: the outer future establishes completion before host reads.
                    unsafe { Ok(out.to_host_vec().execute(exec)?) }
                })())
            })
            .schedule(&policy)?
            .await??;
            for column in 0..INPUT_DIM {
                let expected = window
                    .clone()
                    .map(|position| cached_values[indices[position] as usize * INPUT_DIM + column])
                    .sum::<f32>()
                    / window.len() as f32;
                close(values[column], expected);
            }
        }
        kv.release();
        assert_eq!(pool.metrics().layers.assigned_bytes, 0);
        Ok(())
    }

    #[tokio::test]
    async fn shared_submission_failure_is_reported_on_every_output() -> Result<(), ModelError> {
        use crate::types::SessionRequest;
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let mut model = SimpleMLPEngine::random(0).await?;
        let policy = model.weight_policy.clone();
        let pool = pool(&policy).await?;
        let initial = pool.metrics();
        let original = model.layers[3].clone();
        // Fail a launch shape check after all contexts reserved KV and mapping work
        // was submitted, without poisoning the device with an illegal access.
        model.layers[3] = original.reshape(&[8, 32])?;
        let mut a = KVContext::new(pool.clone())?;
        let mut b = KVContext::new(pool.clone())?;
        let outputs = model
            .generate(
                vec![
                    SessionRequest::new(
                        None,
                        ModelInput {
                            token_ids: &[1, 2],
                            kv: &mut a,
                        },
                    )
                    .with_request_id(90),
                    SessionRequest::new(
                        None,
                        ModelInput {
                            token_ids: &[8],
                            kv: &mut b,
                        },
                    )
                    .with_request_id(91),
                ],
                policy.clone(),
            )
            .await;
        assert_eq!(
            outputs
                .iter()
                .map(|output| output.request_id)
                .collect::<Vec<_>>(),
            vec![90, 91]
        );
        assert!(outputs
            .iter()
            .all(|output| matches!(output.result, Err(ModelError::Execution(_)))));
        assert_eq!(
            outputs[0].result.as_ref().unwrap_err().to_string(),
            outputs[1].result.as_ref().unwrap_err().to_string()
        );
        assert!(a.is_pending() && b.is_pending());
        assert_eq!(a.ready_len(), 0);
        assert_eq!(b.ready_len(), 0);
        a.release();
        b.release();
        assert_eq!(pool.metrics(), initial);
        model.layers[3] = original;
        let mut context = KVContext::new(pool.clone())?;
        assert!(model
            .forward(&[1, 2], &mut context, policy)
            .await?
            .is_finite());
        context.release();
        assert_eq!(pool.metrics(), initial);
        Ok(())
    }

    #[tokio::test]
    async fn packed_ragged_batch_matches_independent_cpu_histories() -> Result<(), ModelError> {
        use crate::types::SessionRequest;
        let _guard = crate::runner::GPU_TEST_LOCK.lock().await;
        let model = SimpleMLPEngine::random(0).await?;
        let policy = model.weight_policy.clone();
        let pool = pool(&policy).await?;
        let initial = pool.metrics();
        let mut weights = Vec::new();
        for weight in &model.layers {
            weights.push(weight.clone().to_host_vec().schedule(&policy)?.await?);
        }
        let mut contexts = [
            KVContext::new(pool.clone())?,
            KVContext::new(pool.clone())?,
            KVContext::new(pool.clone())?,
        ];
        model
            .forward(&[1, 2], &mut contexts[0], policy.clone())
            .await?;
        model
            .forward(&[8], &mut contexts[1], policy.clone())
            .await?;
        let histories: [&[u32]; 3] = [&[1, 2, 4, 5, 9], &[8, 7], &[6, 3, 2, 10]];
        let batch = contexts
            .iter_mut()
            .zip(histories)
            .enumerate()
            .map(|(i, (kv, token_ids))| {
                SessionRequest::new(Some(100 + i as u64), ModelInput { token_ids, kv })
                    .with_request_id(20 + i as u64)
            })
            .collect();
        let outputs = model.generate(batch, policy).await;
        for (i, output) in outputs.iter().enumerate() {
            assert_eq!(output.request_id, 20 + i as u64);
            let actual = model
                .tokenizer()
                .decode(output.result.as_ref().unwrap(), true)
                .unwrap()
                .parse::<f32>()
                .unwrap();
            close(
                actual,
                cpu_reference(histories[i], &weights, model.config.input_divisor),
            );
            assert_eq!(contexts[i].ready_len(), histories[i].len());
        }
        drop(contexts);
        assert_eq!(pool.metrics(), initial);
        Ok(())
    }

    #[tokio::test]
    async fn concurrent_contexts_remain_isolated() -> Result<(), ModelError> {
        let _guard = crate::runner::GPU_TEST_LOCK.lock().await;
        let model = SimpleMLPEngine::random(0).await?;
        let policy = model.weight_policy.clone();
        let pool = pool(&policy).await?;
        let initial = pool.metrics();
        let mut a = KVContext::new(pool.clone())?;
        let mut b = KVContext::new(pool.clone())?;
        let (left, right) = tokio::join!(
            model.forward(&[1, 2, 3], &mut a, policy.clone()),
            model.forward(&[8, 7, 6, 5, 4], &mut b, policy.clone())
        );
        let (left, right) = (left?, right?);
        let mut reference = KVContext::new(pool.clone())?;
        close(
            left,
            model
                .forward(&[1, 2, 3], &mut reference, policy.clone())
                .await?,
        );
        reference.release();
        let mut reference = KVContext::new(pool.clone())?;
        close(
            right,
            model
                .forward(&[8, 7, 6, 5, 4], &mut reference, policy.clone())
                .await?,
        );
        reference.release();
        assert!(model.generate(vec![], policy).await.is_empty());
        a.release();
        b.release();
        assert_eq!(pool.metrics(), initial);
        Ok(())
    }

    /// Export/load and indexed shards include every attention weight, not just the MLP.
    #[tokio::test]
    async fn pretrained_roundtrip_and_shards() -> Result<(), ModelError> {
        let _guard = crate::runner::GPU_TEST_LOCK.lock().await;
        let dir = tempfile::tempdir().map_err(anyhow::Error::from)?;
        let source = dir.path().join("single");
        let model = SimpleMLPEngine::random(0).await?;
        model.save_pretrained(&source).await?;
        let loaded = SimpleMLPEngine::load(&source, 0).await?;
        let shards = dir.path().join("sharded");
        fs::create_dir(&shards).map_err(anyhow::Error::from)?;
        let bytes = fs::read(source.join("model.safetensors")).map_err(anyhow::Error::from)?;
        let weights = safetensors::SafeTensors::deserialize(&bytes).map_err(anyhow::Error::from)?;
        let mut weight_map = HashMap::new();
        for (index, names) in WEIGHT_NAMES.chunks(3).enumerate() {
            let name = format!("model-{index}.safetensors");
            let tensors = names
                .iter()
                .map(|&key| Ok((key, weights.tensor(key)?)))
                .collect::<anyhow::Result<Vec<_>>>()?;
            serialize_to_file(tensors, None, &shards.join(&name)).map_err(anyhow::Error::from)?;
            for &key in names {
                weight_map.insert(key, name.clone());
            }
        }
        for name in ["config.json", "generation_config.json", "tokenizer.json"] {
            fs::copy(source.join(name), shards.join(name)).map_err(anyhow::Error::from)?;
        }
        fs::write(
            shards.join("model.safetensors.index.json"),
            serde_json::json!({"weight_map": weight_map}).to_string(),
        )
        .map_err(anyhow::Error::from)?;
        let sharded = SimpleMLPEngine::load(&shards, 0).await?;
        let policy = model.weight_policy.clone();
        let pool = pool(&policy).await?;
        let mut expected = None;
        for model in [&model, &loaded, &sharded] {
            let mut kv = KVContext::new(pool.clone())?;
            let actual = model
                .forward(&[2, 7, 3, 1, 8], &mut kv, policy.clone())
                .await?;
            if let Some(expected) = expected {
                close(actual, expected);
            } else {
                expected = Some(actual);
            }
        }
        let missing = dir.path().join("missing");
        assert!(SimpleMLPEngine::load(&missing, 0).await.is_ok());
        assert!(!missing.exists());
        assert!(SimpleMLPEngine::load(dir.path(), 0).await.is_err());
        Ok(())
    }
}
