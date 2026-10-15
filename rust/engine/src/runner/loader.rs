//! Hugging Face directory loading: JSON metadata, tokenizer, and Safetensors weights.
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

use anyhow::{bail, ensure, Context, Result};
use cuda_async::{device_operation::DeviceOp, scheduling_policies::SchedulingPolicy};
use cutile::{
    api,
    core::{bf16, f16},
    tensor::{Reshape, Tensor},
};
use memmap2::{Mmap, MmapOptions};
use safetensors::{Dtype, SafeTensors};
use serde::{de::DeserializeOwned, Deserialize};
use tokenizers::Tokenizer;

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[serde(untagged)]
pub enum EosTokenId {
    One(u32),
    Many(Vec<u32>),
}

#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub do_sample: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_new_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eos_token_id: Option<EosTokenId>,
}

#[derive(Debug)]
pub struct HostTensor {
    /// Owned CPU values converted to f32, flattened in row-major order.
    pub data: Vec<f32>,
    /// Tensor dimensions, e.g. [16, 16]; their product equals data.len().
    pub shape: Vec<usize>,
}

#[derive(Deserialize)]
struct SafetensorsIndex {
    // Tensor name -> shard filename relative to model_dir,
    // e.g. "layers.0.weight" -> "model-00001-of-00002.safetensors".
    weight_map: HashMap<String, String>,
}

pub struct WeightLoader {
    model_dir: PathBuf,
    // Tensor name -> weight filename, for both single-file and sharded models.
    weight_map: HashMap<String, String>,
    // Weight filename -> read-only CPU mapping of that Safetensors file.
    shards: HashMap<String, Mmap>,
}

impl WeightLoader {
    pub fn new(model_dir: &Path) -> Result<Self> {
        let index_path = model_dir.join("model.safetensors.index.json");
        let (weight_map, shards) = if index_path.try_exists()? {
            // An existing index is authoritative; malformed indexes must not fall back.
            let index: SafetensorsIndex = read_json(&index_path)?;
            ensure!(
                !index.weight_map.is_empty(),
                "empty weight_map in {}",
                index_path.display()
            );
            let mut shards = HashMap::new();
            for name in index.weight_map.values().collect::<HashSet<_>>() {
                let relative = Path::new(name);
                ensure!(
                    relative
                        .components()
                        .all(|part| matches!(part, Component::Normal(_)))
                        && relative
                            .extension()
                            .is_some_and(|extension| extension == "safetensors"),
                    "invalid Safetensors shard path: {name}"
                );
                shards.insert(name.clone(), map_weights(&model_dir.join(name))?);
            }
            (index.weight_map, shards)
        } else {
            let name = "model.safetensors".to_string();
            let mmap = map_weights(&model_dir.join(&name))?;
            let tensors = SafeTensors::deserialize(&mmap).context("invalid model.safetensors")?;
            let weight_map = tensors
                .names()
                .into_iter()
                .map(|key| (key.to_owned(), name.clone()))
                .collect();
            (weight_map, HashMap::from([(name, mmap)]))
        };
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            weight_map,
            shards,
        })
    }

    // Config schemas belong to each model, rather than a shared Qwen-only struct.
    pub fn config<T: DeserializeOwned>(&self) -> Result<T> {
        read_json(&self.model_dir.join("config.json"))
    }

    pub fn generation_config(&self) -> Result<Option<GenerationConfig>> {
        let path = self.model_dir.join("generation_config.json");
        if !path.try_exists()? {
            return Ok(None);
        }
        read_json(&path).map(Some)
    }

    pub fn tokenizer(&self) -> Result<Tokenizer> {
        let path = self.model_dir.join("tokenizer.json");
        Tokenizer::from_file(&path)
            .map_err(|error| anyhow::anyhow!("failed to load {}: {error}", path.display()))
    }

    pub fn load_host_f32(&self, name: &str) -> Result<HostTensor> {
        let shard = self
            .weight_map
            .get(name)
            .with_context(|| format!("tensor {name:?} not found"))?;
        let mmap = self.shards.get(shard).context("missing mapped shard")?;
        let tensors =
            SafeTensors::deserialize(mmap).with_context(|| format!("invalid shard {shard}"))?;
        let tensor = tensors
            .tensor(name)
            .with_context(|| format!("tensor {name:?} missing from {shard}"))?;
        let bytes = tensor.data();
        let data = match tensor.dtype() {
            Dtype::F32 => bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            Dtype::F16 => bytes
                .chunks_exact(2)
                .map(|b| f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
                .collect(),
            Dtype::BF16 => bytes
                .chunks_exact(2)
                .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
                .collect(),
            Dtype::F64 => bytes
                .chunks_exact(8)
                .map(|b| f64::from_le_bytes(b.try_into().unwrap()) as f32)
                .collect(),
            dtype => bail!("unsupported weight dtype {dtype:?} for tensor {name:?}"),
        };
        Ok(HostTensor {
            data,
            shape: tensor.shape().to_vec(),
        })
    }

    pub async fn load_device_f32(
        &self,
        name: &str,
        policy: &Arc<dyn SchedulingPolicy>,
    ) -> Result<Arc<Tensor<f32>>> {
        let host = self.load_host_f32(name)?;
        let tensor = api::copy_host_vec_to_device(&Arc::new(host.data))
            .schedule(policy)?
            .await?
            .reshape(&host.shape)?;
        Ok(Arc::new(tensor))
    }
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))
}

fn map_weights(path: &Path) -> Result<Mmap> {
    let file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    // Model files must remain unchanged while this read-only mapping is alive.
    unsafe { MmapOptions::new().map(&file) }
        .with_context(|| format!("failed to map {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize_to_file, TensorView};
    use serde_json::json;
    use tempfile::TempDir;

    fn write_tensor(
        path: &Path,
        name: &str,
        dtype: Dtype,
        shape: Vec<usize>,
        bytes: &[u8],
    ) -> Result<()> {
        serialize_to_file([(name, TensorView::new(dtype, shape, bytes)?)], None, path)?;
        Ok(())
    }

    // Both file layouts must resolve names to identical values and shapes.
    #[test]
    fn loads_single_file_and_sharded_weights() -> Result<()> {
        for sharded in [false, true] {
            let dir = TempDir::new()?;
            let a: Vec<u8> = [1.0f32, -2.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect();
            let b = 3.0f32.to_le_bytes();
            if sharded {
                write_tensor(
                    &dir.path().join("model-00001-of-00002.safetensors"),
                    "a",
                    Dtype::F32,
                    vec![2],
                    &a,
                )?;
                write_tensor(
                    &dir.path().join("model-00002-of-00002.safetensors"),
                    "b",
                    Dtype::F32,
                    vec![1],
                    &b,
                )?;
                fs::write(dir.path().join("model.safetensors.index.json"), json!({
                    "metadata": {"total_size": 12},
                    "weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}
                }).to_string())?;
            } else {
                serialize_to_file(
                    [
                        ("a", TensorView::new(Dtype::F32, vec![2], &a)?),
                        ("b", TensorView::new(Dtype::F32, vec![1], &b)?),
                    ],
                    None,
                    &dir.path().join("model.safetensors"),
                )?;
            }
            let loader = WeightLoader::new(dir.path())?;
            assert_eq!(loader.load_host_f32("a")?.data, vec![1.0, -2.0]);
            assert_eq!(loader.load_host_f32("a")?.shape, vec![2]);
            assert_eq!(loader.load_host_f32("b")?.data, vec![3.0]);
            assert!(loader.load_host_f32("missing").is_err());
        }
        Ok(())
    }

    // Float conversion must preserve representable values and reject integer weights.
    #[test]
    fn converts_supported_float_dtypes() -> Result<()> {
        let dir = TempDir::new()?;
        for (dtype, bytes) in [
            (
                Dtype::F16,
                f16::from_f32(1.5).to_bits().to_le_bytes().to_vec(),
            ),
            (
                Dtype::BF16,
                bf16::from_f32(1.5).to_bits().to_le_bytes().to_vec(),
            ),
            (Dtype::F32, 1.5f32.to_le_bytes().to_vec()),
            (Dtype::F64, 1.5f64.to_le_bytes().to_vec()),
            (Dtype::I32, 1i32.to_le_bytes().to_vec()),
        ] {
            write_tensor(
                &dir.path().join("model.safetensors"),
                "w",
                dtype,
                vec![1],
                &bytes,
            )?;
            let loader = WeightLoader::new(dir.path())?;
            if dtype == Dtype::I32 {
                assert!(loader.load_host_f32("w").is_err());
            } else {
                assert_eq!(loader.load_host_f32("w")?.data, vec![1.5]);
            }
        }
        Ok(())
    }

    // An existing broken index must fail instead of silently selecting a different model.
    #[test]
    fn rejects_broken_indexes_and_missing_shards() -> Result<()> {
        let dir = TempDir::new()?;
        write_tensor(
            &dir.path().join("model.safetensors"),
            "w",
            Dtype::F32,
            vec![1],
            &1.0f32.to_le_bytes(),
        )?;
        let index = dir.path().join("model.safetensors.index.json");
        fs::write(&index, "invalid JSON")?;
        assert!(WeightLoader::new(dir.path()).is_err());
        fs::write(
            &index,
            json!({"weight_map": {"w": "missing.safetensors"}}).to_string(),
        )?;
        assert!(WeightLoader::new(dir.path()).is_err());
        fs::write(
            &index,
            json!({"weight_map": {"w": "../outside.safetensors"}}).to_string(),
        )?;
        assert!(WeightLoader::new(dir.path()).is_err());
        Ok(())
    }

    // Config is model-specific; generation_config is optional, but malformed JSON is an error.
    #[test]
    fn loads_config_and_optional_generation_settings() -> Result<()> {
        let dir = TempDir::new()?;
        write_tensor(
            &dir.path().join("model.safetensors"),
            "w",
            Dtype::F32,
            vec![1],
            &1.0f32.to_le_bytes(),
        )?;
        let loader = WeightLoader::new(dir.path())?;
        assert!(loader.config::<serde_json::Value>().is_err());
        fs::write(
            dir.path().join("config.json"),
            r#"{"model_type":"simple_mlp"}"#,
        )?;
        assert_eq!(
            loader.config::<serde_json::Value>()?["model_type"],
            "simple_mlp"
        );
        assert!(loader.generation_config()?.is_none());
        fs::write(
            dir.path().join("generation_config.json"),
            r#"{"do_sample":false,"eos_token_id":[1,2]}"#,
        )?;
        let config = loader.generation_config()?.unwrap();
        assert_eq!(config.do_sample, Some(false));
        assert!(matches!(config.eos_token_id, Some(EosTokenId::Many(ids)) if ids == vec![1,2]));
        fs::write(dir.path().join("generation_config.json"), "invalid JSON")?;
        assert!(loader.generation_config().is_err());
        Ok(())
    }
}
