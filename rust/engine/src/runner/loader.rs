/// Loader for model weights and config
use crate::config::SafetensorsIndex;
use anyhow::{bail, Context, Result};
use memmap2::MmapOptions;
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::core::f16;
use cutile::tensor::{Reshape, Tensor};

#[derive(Debug)]
pub struct HostTensor {
    pub data: Vec<f16>,
    pub shape: Vec<usize>,
}

pub struct WeightLoader {
    model_dir: PathBuf,
    weight_map: HashMap<String, String>,
    shards: HashMap<String, memmap2::Mmap>,
}

impl WeightLoader {
    pub fn new(model_dir: &Path) -> Result<Self> {
        let index_path = model_dir.join("model.safetensors.index.json");
        let index_text = std::fs::read_to_string(&index_path)
            .with_context(|| format!("failed to read {}", index_path.display()))?;
        let index: SafetensorsIndex = serde_json::from_str(&index_text)
            .with_context(|| format!("failed to parse {}", index_path.display()))?;

        let mut shard_files = HashSet::new();
        for shard in index.weight_map.values() {
            shard_files.insert(shard.clone());
        }

        let shard_files: Vec<String> = shard_files.into_iter().collect();
        let shard_entries: Vec<(String, memmap2::Mmap)> = shard_files
            .par_iter()
            .map(|shard| -> Result<(String, memmap2::Mmap)> {
                let shard_path = model_dir.join(shard);
                let file = std::fs::File::open(&shard_path)
                    .with_context(|| format!("failed to open {}", shard_path.display()))?;
                let mmap = unsafe { MmapOptions::new().map(&file) }
                    .with_context(|| format!("failed to mmap {}", shard_path.display()))?;
                Ok((shard.clone(), mmap))
            })
            .collect::<Result<Vec<_>>>()?;
        let shards = shard_entries.into_iter().collect();

        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            weight_map: index.weight_map,
            shards,
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn load_host_f16(&self, name: &str) -> Result<HostTensor> {
        let shard_name = self
            .weight_map
            .get(name)
            .with_context(|| format!("tensor `{name}` not found in index"))?;
        let mmap = self
            .shards
            .get(shard_name)
            .with_context(|| format!("missing mmap for shard `{shard_name}`"))?;

        let st = SafeTensors::deserialize(&mmap[..])
            .with_context(|| format!("failed to deserialize `{shard_name}`"))?;
        let view = st
            .tensor(name)
            .with_context(|| format!("tensor `{name}` not found in `{shard_name}`"))?;

        let shape = view.shape().to_vec();
        let data = cast_to_f16(view.dtype(), view.data())
            .with_context(|| format!("failed to cast `{name}` from {:?}", view.dtype()))?;
        Ok(HostTensor { data, shape })
    }

    pub fn load_device_f16(
        &self,
        name: &str,
        stream: &Arc<cuda_core::Stream>,
    ) -> Result<Arc<Tensor<f16>>> {
        let host = self.load_host_f16(name)?;
        let shape = host.shape.clone();
        let host_data = Arc::new(host.data);
        let device_tensor = api::copy_host_vec_to_device(&host_data)
            .sync_on(stream)
            .map_err(|e| anyhow::anyhow!("copy to device failed: {e:?}"))?;
        let device_tensor = device_tensor
            .reshape(&shape)
            .map_err(|e| anyhow::anyhow!("reshape failed: {e:?}"))?;
        Ok(Arc::new(device_tensor))
    }
}

use cutile::core::bf16;

fn cast_to_f16(dtype: Dtype, data: &[u8]) -> Result<Vec<f16>> {
    match dtype {
        Dtype::F16 => {
            let mut out = Vec::with_capacity(data.len() / 2);
            for bytes in data.chunks_exact(2) {
                let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                out.push(f16::from_bits(bits));
            }
            Ok(out)
        }
        Dtype::BF16 => {
            let mut out = Vec::with_capacity(data.len() / 2);
            for bytes in data.chunks_exact(2) {
                let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                out.push(f16::from_f32(bf16::from_bits(bits).to_f32()));
            }
            Ok(out)
        }
        Dtype::F32 => {
            let mut out = Vec::with_capacity(data.len() / 4);
            for bytes in data.chunks_exact(4) {
                let x = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                out.push(f16::from_f32(x));
            }
            Ok(out)
        }
        Dtype::F64 => {
            let mut out = Vec::with_capacity(data.len() / 8);
            for bytes in data.chunks_exact(8) {
                let x = f64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ]);
                out.push(f16::from_f32(x as f32));
            }
            Ok(out)
        }
        other => bail!("unsupported dtype for fp16 cast: {other:?}"),
    }
}

/// Config
use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub use_sliding_window: bool,
    pub eos_token_id: u32,
}

impl ModelConfig {
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        let cfg_path = model_dir.join("config.json");
        let cfg_text = fs::read_to_string(&cfg_path)
            .with_context(|| format!("failed to read {}", cfg_path.display()))?;
        let cfg: Self = serde_json::from_str(&cfg_text)
            .with_context(|| format!("failed to parse {}", cfg_path.display()))?;
        println!("cfg: {cfg:#?}");
        Ok(cfg)
    }

    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

#[derive(Debug, Deserialize)]
pub struct SafetensorsIndex {
    pub weight_map: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EosTokenId {
    One(u32),
    Many(Vec<u32>),
}

impl EosTokenId {
    pub fn into_vec(self) -> Vec<u32> {
        match self {
            Self::One(x) => vec![x],
            Self::Many(xs) => xs,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct GenerationConfig {
    pub do_sample: Option<bool>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub eos_token_id: Option<EosTokenId>,
}

impl GenerationConfig {
    pub fn from_model_dir(model_dir: &Path) -> Result<Option<Self>> {
        let path = model_dir.join("generation_config.json");
        if !path.exists() {
            return Ok(None);
        }
        let text = fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let cfg: Self = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        Ok(Some(cfg))
    }
}
