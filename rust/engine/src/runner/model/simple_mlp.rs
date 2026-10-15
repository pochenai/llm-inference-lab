use anyhow::{ensure, Context};
use cutile::tensor::ToHostVec;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokenizers::{
    decoders::fuse::Fuse, models::wordlevel::WordLevel,
    pre_tokenizers::whitespace::WhitespaceSplit, Encoding, Tokenizer,
};

use cuda_async::{
    device_context::global_policy, device_operation::*, scheduling_policies::SchedulingPolicy,
};
use cutile::{
    api,
    tensor::{Reshape, Tensor, Unpartition},
    tile_kernel::{PartitionOp, TileKernel, ToHostVecOp},
};

use super::Model;
use crate::{
    error::EngineError,
    runner::{
        loader::{GenerationConfig, WeightLoader},
        types::ModelOutput,
    },
    types::BatchIn,
};

pub const INPUT_DIM: usize = 16;
const TILE: usize = 4;

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
            model_type: "simple_mlp".into(),
            architectures: vec!["SimpleMLPEngine".into()],
            torch_dtype: "float32".into(),
            input_size: 16,
            hidden_size: 16,
            output_size: 1,
            input_divisor: 100.0,
        }
    }
}

/// A bias-free 16 -> 16 -> 1 MLP: output = ReLU(input * w0) * w1.
pub struct SimpleMLPEngine {
    tokenizer: Tokenizer,
    config: SimpleMLPConfig,
    generation_config: Option<GenerationConfig>,
    policy: Arc<dyn SchedulingPolicy>,
    // Hidden-layer matrix [16, 16], followed by output-layer vector [16].
    layers: Vec<Arc<Tensor<f32>>>,
}

impl SimpleMLPEngine {
    fn default_model_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("models/simple_mlp")
    }

    pub async fn new(device_id: usize) -> Result<Self, EngineError> {
        Self::load(&Self::default_model_dir(), device_id).await
    }

    pub async fn load(model_dir: &Path, device_id: usize) -> Result<Self, EngineError> {
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
                config.model_type == "simple_mlp"
                    && config.input_size == INPUT_DIM
                    && config.hidden_size == INPUT_DIM
                    && config.output_size == 1
                    && config.torch_dtype == "float32"
                    && config.input_divisor.is_finite()
                    && config.input_divisor > 0.0,
                "this kernel requires a float32 simple_mlp with dimensions 16 -> 16 -> 1"
            );
            let tokenizer = loader.tokenizer()?;
            let generation_config = loader.generation_config()?;
            let policy = global_policy(device_id)?;
            let mut layers = Vec::new();
            for (index, shape) in [vec![INPUT_DIM, INPUT_DIM], vec![INPUT_DIM]]
                .iter()
                .enumerate()
            {
                let name = format!("layers.{index}.weight");
                let weight = loader.load_device_f32(&name, &policy).await?;
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
                policy,
                layers,
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
    pub async fn random(device_id: usize) -> Result<Self, EngineError> {
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
            .map_err(|error| EngineError::Model(error.to_string()))?;
        let mut tokenizer = Tokenizer::new(word_level);
        tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
        tokenizer.with_decoder(Some(Fuse::new()));
        let policy = global_policy(device_id)?;
        // Generate weights on the GPU and wait before exposing them to forward kernels.
        let w0 = api::randn(0.0f32, 0.25, [INPUT_DIM, INPUT_DIM], Some(42))
            .schedule(&policy)?
            .await?;
        let w1 = api::randn(0.0f32, 0.25, [INPUT_DIM], Some(43))
            .schedule(&policy)?
            .await?;
        Ok(Self {
            tokenizer,
            config: SimpleMLPConfig::default(),
            generation_config: Some(GenerationConfig {
                do_sample: Some(false),
                ..Default::default()
            }),
            policy,
            layers: vec![Arc::new(w0), Arc::new(w1)],
        })
    }

    /// Save a local Hugging Face style directory; the Rust model owns its computation graph.
    pub async fn save_pretrained(&self, model_dir: &Path) -> Result<(), EngineError> {
        let result: anyhow::Result<()> = async {
            fs::create_dir_all(model_dir)?;
            ensure!(
                !model_dir.join("model.safetensors.index.json").exists(),
                "refusing to export single-file weights over an existing shard index"
            );
            let mut buffers = Vec::new();
            for layer in &self.layers {
                let data = layer.clone().to_host_vec().schedule(&self.policy)?.await?;
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
                        format!("layers.{index}.weight"),
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

impl Model for SimpleMLPEngine {
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    async fn generate(&self, batch: BatchIn<Encoding>) -> Result<Vec<ModelOutput>, EngineError> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }
        for item in &batch {
            if item.request.get_ids().len() != INPUT_DIM
                || item.request.get_ids().iter().any(|&id| id >= 256)
            {
                return Err(EngineError::InvalidInput(format!(
                    "request {} needs {INPUT_DIM} numeric tokens in 0..256",
                    item.request_id,
                )));
            }
        }
        // Zero-pad rows for the 4-row kernel tile; discard padded outputs below.
        let rows = batch.len().div_ceil(TILE) * TILE;
        let mut inputs = vec![0.0; rows * INPUT_DIM];
        for (row, item) in batch.iter().enumerate() {
            // Scale demo token IDs into the MLP's floating-point features.
            for (column, &id) in item.request.get_ids().iter().enumerate() {
                inputs[row * INPUT_DIM + column] = id as f32 / self.config.input_divisor;
            }
        }
        let input = api::copy_host_vec_to_device(&Arc::new(inputs))
            .schedule(&self.policy)?
            .await?
            .reshape(&[rows, INPUT_DIM])?;
        let hidden = api::zeros::<f32>(&[rows, INPUT_DIM]).partition([TILE, TILE]);
        let hidden = kernels::gemm(hidden, input, self.layers[0].clone())
            .generics(vec!["4".into(), "4".into(), "4".into(), "16".into()])
            .first();
        let hidden = kernels::relu(hidden)
            .generics(vec!["4".into(), "4".into()])
            .map(|(hidden,)| hidden);
        let output = api::zeros::<f32>(&[rows]).partition([TILE]);
        let values = kernels::matvec(output, hidden.unpartition(), self.layers[1].clone())
            .generics(vec!["4".into(), "4".into(), "16".into()])
            .first()
            .unpartition()
            .to_host_vec()
            .schedule(&self.policy)?
            .await?;
        batch
            .into_iter()
            .zip(values)
            .map(|(item, output)| {
                // This numeric demo spells its scalar as tokens. An LLM would
                // return token IDs produced by its sampler directly.
                let token_ids = output
                    .to_string()
                    .chars()
                    .map(|character| {
                        self.tokenizer
                            .token_to_id(&character.to_string())
                            .ok_or_else(|| {
                                EngineError::Model(format!("no output token for {character:?}"))
                            })
                    })
                    .collect::<Result<_, _>>()?;
                Ok(ModelOutput {
                    request_id: item.request_id,
                    token_ids,
                })
            })
            .collect()
    }
}

// Kernels adapted from NVIDIA cutile-examples/examples/async_mlp.rs.
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn gemm<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        y: &Tensor<f32, { [K, -1] }>,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_y = y.partition(shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z = z.load();
        for i in 0i32..(K / BK) {
            tile_z = mma(part_x.load([pid.0, i]), part_y.load([i, pid.1]), tile_z);
        }
        z.store(tile_z);
    }

    #[cutile::entry()]
    pub fn relu<const BM: i32, const BN: i32>(z: &mut Tensor<f32, { [BM, BN] }>) {
        let zero = constant(0.0f32, shape![BM, BN]);
        let input = z.load();
        z.store(max_tile(zero, input));
    }

    #[cutile::entry()]
    pub fn matvec<const BM: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<f32, { [BM] }>,
        x: &Tensor<f32, { [-1, K] }>,
        y: &Tensor<f32, { [K] }>,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_y = y.partition(shape![BK]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z = z.load().reshape(shape![BM, 1]);
        for i in 0i32..(K / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i]).reshape(shape![BK, 1]);
            tile_z = mma(tile_x, tile_y, tile_z);
        }
        z.store(tile_z.reshape(shape![BM]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SingleRequest;
    use cutile::tensor::ToHostVec;

    // Only an absent directory triggers random initialization; existing broken models fail.
    #[tokio::test]
    async fn missing_directory_uses_random_weights() -> Result<(), EngineError> {
        let _gpu_guard = crate::runner::GPU_TEST_LOCK.lock().await;
        let dir = tempfile::tempdir().map_err(anyhow::Error::from)?;
        let missing = dir.path().join("missing-model");
        let model = SimpleMLPEngine::load(&missing, 0).await?;
        assert_eq!(model.config().input_size, INPUT_DIM);
        assert!(!missing.exists()); // Fallback does not implicitly save model files.
        let output = model
            .generate(vec![SingleRequest {
                request_id: 42,
                request: model
                    .tokenizer()
                    .encode(vec!["0"; INPUT_DIM].join(" "), true)
                    .expect("numeric fixture is tokenizable"),
            }])
            .await?;
        assert_eq!(
            model
                .tokenizer()
                .decode(&output[0].token_ids, true)
                .unwrap(),
            "0"
        );
        assert!(SimpleMLPEngine::load(dir.path(), 0).await.is_err());
        fs::write(dir.path().join("model.safetensors"), b"invalid weights")
            .map_err(anyhow::Error::from)?;
        assert!(SimpleMLPEngine::load(dir.path(), 0).await.is_err());
        Ok(())
    }

    // Re-shard the exported fixture and verify that both layouts execute identically.
    #[tokio::test]
    async fn sharded_model_matches_single_file_on_gpu() -> Result<(), EngineError> {
        let _gpu_guard = crate::runner::GPU_TEST_LOCK.lock().await;
        let source = SimpleMLPEngine::default_model_dir();
        let dir = tempfile::tempdir().map_err(anyhow::Error::from)?;
        let bytes = fs::read(source.join("model.safetensors")).map_err(anyhow::Error::from)?;
        let weights = safetensors::SafeTensors::deserialize(&bytes).map_err(anyhow::Error::from)?;
        let mut weight_map = HashMap::new();
        for (index, name) in ["layers.0.weight", "layers.1.weight"].iter().enumerate() {
            let shard = format!("model-{:05}-of-00002.safetensors", index + 1);
            serialize_to_file(
                [(*name, weights.tensor(name).map_err(anyhow::Error::from)?)],
                None,
                &dir.path().join(&shard),
            )
            .map_err(anyhow::Error::from)?;
            weight_map.insert(*name, shard);
        }
        for name in ["config.json", "generation_config.json", "tokenizer.json"] {
            fs::copy(source.join(name), dir.path().join(name)).map_err(anyhow::Error::from)?;
        }
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::json!({"metadata": {"total_size": 1088}, "weight_map": weight_map})
                .to_string(),
        )
        .map_err(anyhow::Error::from)?;
        let single = SimpleMLPEngine::load(&source, 0).await?;
        let sharded = SimpleMLPEngine::load(dir.path(), 0).await?;
        assert_eq!(sharded.config().input_size, INPUT_DIM);
        assert_eq!(sharded.generation_config().unwrap().do_sample, Some(false));
        let batch = |model: &SimpleMLPEngine| {
            vec![SingleRequest {
                request_id: 42,
                request: model
                    .tokenizer()
                    .encode(
                        (0..INPUT_DIM)
                            .map(|id| id.to_string())
                            .collect::<Vec<_>>()
                            .join(" "),
                        true,
                    )
                    .expect("numeric fixture is tokenizable"),
            }]
        };
        let (a, b) = tokio::join!(
            single.generate(batch(&single)),
            sharded.generate(batch(&sharded))
        );
        let (a, b) = (a?, b?);
        assert_eq!(a[0].request_id, b[0].request_id);
        assert_eq!(a[0].token_ids, b[0].token_ids);
        Ok(())
    }

    // Verify concurrent GPU batches against the same weights evaluated on the CPU.
    #[tokio::test]
    async fn gpu_mlp_matches_cpu_for_padded_batches() -> Result<(), EngineError> {
        let _gpu_guard = crate::runner::GPU_TEST_LOCK.lock().await;
        let model = SimpleMLPEngine::new(0).await?;
        // Copy the loaded GPU weights back solely for the CPU reference.
        let w0 = model.layers[0]
            .clone()
            .to_host_vec()
            .schedule(&model.policy)?
            .await?;
        let w1 = model.layers[1]
            .clone()
            .to_host_vec()
            .schedule(&model.policy)?
            .await?;
        let make_inputs = |count: usize| -> Vec<Vec<f32>> {
            (0..count)
                .map(|i| {
                    (0..INPUT_DIM)
                        .map(|j| (i * INPUT_DIM + j) as f32 / 100.0)
                        .collect()
                })
                .collect()
        };
        // Convert features back to numeric words; encoding and ID scaling must
        // reconstruct the original features used by the CPU reference.
        let make_batch = |inputs: &[Vec<f32>]| -> BatchIn<Encoding> {
            inputs
                .iter()
                .enumerate()
                .map(|(i, input)| SingleRequest {
                    request_id: i as u64,
                    request: model
                        .tokenizer()
                        .encode(
                            input
                                .iter()
                                .map(|value| ((value * 100.0).round() as u32).to_string())
                                .collect::<Vec<_>>()
                                .join(" "),
                            true,
                        )
                        .expect("numeric fixture is tokenizable"),
                })
                .collect()
        };
        // Five rows pad to eight, and one row pads to four. Both calls borrow
        // the same model while retaining separate intermediate/output buffers.
        let inputs_a = make_inputs(5);
        let inputs_b = make_inputs(1);
        let (output_a, output_b) = tokio::join!(
            model.generate(make_batch(&inputs_a)),
            model.generate(make_batch(&inputs_b)),
        );
        for (inputs, output) in [(inputs_a, output_a?), (inputs_b, output_b?)] {
            // Padding must not produce extra responses.
            assert_eq!(output.len(), inputs.len());
            for (i, item) in output.iter().enumerate() {
                // Reference: ReLU(X * W0) * W1, with no biases.
                let expected: f32 = (0..INPUT_DIM)
                    .map(|h| {
                        let hidden: f32 = (0..INPUT_DIM)
                            .map(|k| inputs[i][k] * w0[k * INPUT_DIM + h])
                            .sum();
                        hidden.max(0.0) * w1[h]
                    })
                    .sum();
                assert_eq!(item.request_id, i as u64);
                // Decode demo output tokens back to a scalar for numerical validation.
                let actual: f32 = model
                    .tokenizer()
                    .decode(&item.token_ids, true)
                    .map_err(|error| EngineError::Model(error.to_string()))?
                    .parse()
                    .expect("MLP output is a numeric string");
                assert!(actual.is_finite());
                // Allow small differences between GPU MMA and CPU accumulation.
                assert!(
                    (actual - expected).abs() <= 1e-3 + 3e-3 * expected.abs(),
                    "GPU={}, CPU={expected}",
                    actual
                );
            }
        }
        // Empty batches return no outputs; malformed dimensions return an error.
        assert!(model.generate(Vec::new()).await?.is_empty());
        let invalid = vec![SingleRequest {
            request_id: 99,
            request: model
                .tokenizer()
                .encode("0 1 2", true)
                .expect("numeric fixture is tokenizable"),
        }];
        assert!(matches!(
            model.generate(invalid).await,
            Err(EngineError::InvalidInput(_))
        ));
        Ok(())
    }
}
