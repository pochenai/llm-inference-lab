//! Hugging Face directory loading: JSON metadata, tokenizer, and Safetensors weights.
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

use anyhow::{ensure, Context, Result};
use cuda_async::{device_operation::DeviceOp, scheduling_policies::SchedulingPolicy};
use cutile::{
    api,
    core::{bf16, f16},
    tensor::{Reshape, Tensor},
    DType,
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
pub struct HostTensor<T = f32> {
    /// Owned CPU values in the requested dtype, flattened in row-major order.
    pub data: Vec<T>,
    /// Tensor dimensions, e.g. [16, 16]; their product equals data.len().
    pub shape: Vec<usize>,
}

/// A supported destination type for floating-point Safetensors weights.
///
/// This contract covers weight decoding and device storage, not kernel support.
pub trait WeightElement: DType {
    /// Decode little-endian F16, BF16, F32, or F64 data into this type.
    /// Narrowing conversions may round or overflow to infinity.
    fn decode_weights(dtype: Dtype, bytes: &[u8]) -> Result<Vec<Self>>;
}

macro_rules! impl_weight_element {
    ($ty:ty, $native:expr, $decode:expr, $convert:expr) => {
        impl WeightElement for $ty {
            fn decode_weights(dtype: Dtype, bytes: &[u8]) -> Result<Vec<Self>> {
                ensure!(
                    matches!(dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32 | Dtype::F64),
                    "unsupported weight dtype {dtype:?}"
                );
                ensure!(
                    bytes.len() % (dtype.bitsize() / 8) == 0,
                    "invalid byte length {} for weight dtype {dtype:?}",
                    bytes.len()
                );
                if dtype == $native {
                    // Preserve the exact bits when no conversion is requested.
                    return Ok(bytes
                        .chunks_exact(dtype.bitsize() / 8)
                        .map($decode)
                        .collect());
                }
                let convert: fn(f64) -> Self = $convert;
                Ok(match dtype {
                    Dtype::F16 => bytes
                        .chunks_exact(2)
                        .map(|b| {
                            convert(
                                f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f64(),
                            )
                        })
                        .collect(),
                    Dtype::BF16 => bytes
                        .chunks_exact(2)
                        .map(|b| {
                            convert(
                                bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f64(),
                            )
                        })
                        .collect(),
                    Dtype::F32 => bytes
                        .chunks_exact(4)
                        .map(|b| convert(f32::from_le_bytes(b.try_into().unwrap()) as f64))
                        .collect(),
                    Dtype::F64 => bytes
                        .chunks_exact(8)
                        .map(|b| convert(f64::from_le_bytes(b.try_into().unwrap())))
                        .collect(),
                    _ => unreachable!("weight dtype validated above"),
                })
            }
        }
    };
}

impl_weight_element!(
    f32,
    Dtype::F32,
    |b: &[u8]| f32::from_le_bytes(b.try_into().unwrap()),
    |v| v as f32
);
impl_weight_element!(
    f16,
    Dtype::F16,
    |b: &[u8]| f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())),
    f16::from_f64
);
impl_weight_element!(
    bf16,
    Dtype::BF16,
    |b: &[u8]| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())),
    bf16::from_f64
);

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

    /// Load weights into an explicit destination dtype, independent of file dtype.
    pub fn load_host<T: WeightElement>(&self, name: &str) -> Result<HostTensor<T>> {
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
        let data = T::decode_weights(tensor.dtype(), tensor.data())
            .with_context(|| format!("failed to decode tensor {name:?}"))?;
        Ok(HostTensor {
            data,
            shape: tensor.shape().to_vec(),
        })
    }

    /// Compatibility entry point for existing f32 models.
    pub fn load_host_f32(&self, name: &str) -> Result<HostTensor> {
        self.load_host::<f32>(name)
    }

    /// Upload weights using the requested destination dtype.
    pub async fn load_device<T: WeightElement>(
        &self,
        name: &str,
        policy: &Arc<dyn SchedulingPolicy>,
    ) -> Result<Arc<Tensor<T>>> {
        let host = self.load_host::<T>(name)?;
        let tensor = api::copy_host_vec_to_device(&Arc::new(host.data))
            .schedule(policy)?
            .await?
            .reshape(&host.shape)?;
        Ok(Arc::new(tensor))
    }

    /// Compatibility entry point for existing f32 models.
    pub async fn load_device_f32(
        &self,
        name: &str,
        policy: &Arc<dyn SchedulingPolicy>,
    ) -> Result<Arc<Tensor<f32>>> {
        self.load_device::<f32>(name, policy).await
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
                assert!(loader.load_host::<f32>("w").is_err());
                assert!(loader.load_host::<f16>("w").is_err());
                assert!(loader.load_host::<bf16>("w").is_err());
            } else {
                assert_eq!(loader.load_host_f32("w")?.data, vec![1.5]);
                let half: HostTensor<f16> = loader.load_host("w")?;
                assert_eq!(half.shape, vec![1]);
                assert_eq!(half.data, vec![f16::from_f32(1.5)]);
                assert_eq!(
                    loader.load_host::<bf16>("w")?.data,
                    vec![bf16::from_f32(1.5)]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn preserves_native_bits_and_validates_conversion_bytes() -> Result<()> {
        let dir = TempDir::new()?;
        // Negative zero, a subnormal, infinity, and a NaN with a payload.
        for dtype in [Dtype::F16, Dtype::BF16] {
            let bits: Vec<u16> = if dtype == Dtype::F16 {
                vec![0x8000, 0x0001, 0x7c00, 0x7e55]
            } else {
                vec![0x8000, 0x0001, 0x7f80, 0x7fc5]
            };
            let bytes: Vec<u8> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
            write_tensor(
                &dir.path().join("model.safetensors"),
                "w",
                dtype,
                vec![4],
                &bytes,
            )?;
            let loader = WeightLoader::new(dir.path())?;
            let actual: Vec<u16> = if dtype == Dtype::F16 {
                loader
                    .load_host::<f16>("w")?
                    .data
                    .into_iter()
                    .map(f16::to_bits)
                    .collect()
            } else {
                loader
                    .load_host::<bf16>("w")?
                    .data
                    .into_iter()
                    .map(bf16::to_bits)
                    .collect()
            };
            assert_eq!(actual, bits);
        }
        assert!(f16::decode_weights(Dtype::F32, &[0, 0, 0]).is_err());
        assert!(bf16::decode_weights(Dtype::F16, &[0]).is_err());
        // Narrowing rounds to a destination value and can overflow to infinity.
        let value = 1.0007f64;
        assert_eq!(
            f16::decode_weights(Dtype::F64, &value.to_le_bytes())?[0].to_bits(),
            0x3c01
        );
        assert!(f16::decode_weights(Dtype::F64, &70000.0f64.to_le_bytes())?[0].is_infinite());
        Ok(())
    }

    #[tokio::test]
    async fn uploads_requested_float_dtypes() -> Result<()> {
        use cutile::tensor::ToHostVec;
        let _gpu_guard = crate::runner::GPU_TEST_LOCK.lock().await;
        let dir = TempDir::new()?;
        let bytes: Vec<u8> = [1.5f32, -2.25]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        write_tensor(
            &dir.path().join("model.safetensors"),
            "w",
            Dtype::F32,
            vec![1, 2],
            &bytes,
        )?;
        let loader = WeightLoader::new(dir.path())?;
        let policy = cuda_async::device_context::global_policy(0)?;
        let single = loader.load_device::<f32>("w", &policy).await?;
        let half = loader.load_device::<f16>("w", &policy).await?;
        let brain = loader.load_device::<bf16>("w", &policy).await?;
        assert_eq!(single.shape(), half.shape());
        assert_eq!(single.shape(), brain.shape());
        assert_eq!(single.to_host_vec().await?, vec![1.5f32, -2.25]);
        assert_eq!(
            half.to_host_vec().await?,
            vec![f16::from_f32(1.5), f16::from_f32(-2.25)]
        );
        assert_eq!(
            brain.to_host_vec().await?,
            vec![bf16::from_f32(1.5), bf16::from_f32(-2.25)]
        );
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
