//! Offline single-request Qwen2.5-0.5B-Instruct, using cuTile for every numerical op.
//! From the repository root (MODEL_DIR is the existing HF snapshot directory):
//! cargo run --manifest-path rust/Cargo.toml --offline --release -p engine --example qwen2 -- MODEL_DIR
//! Match --max-new-tokens to the reference run; e.g. --max-new-tokens 32
//! --reference-dir rust/engine/artifacts/qwen2 validates CPU reference logits and tokens.

use anyhow::{ensure, Result};
use clap::Parser;
use cuda_async::scheduling_policies::{SchedulingPolicy, SingleStream};
use cutile::cuda_core::Device;
use engine::runner::{
    kv_pool::{KVContext, KVPool, NaiveKVPool},
    model::{CausalModel, Qwen2Engine},
};
use futures::TryStreamExt;
use std::{num::NonZeroUsize, path::PathBuf, sync::Arc, time::Instant};

/// Run offline Qwen2.5-0.5B-Instruct inference with cuTile kernels.
#[derive(Parser)]
struct Args {
    /// Local Hugging Face checkpoint directory.
    model_dir: PathBuf,
    /// User message to include in the chat template.
    #[arg(long, default_value = "Say hello.")]
    prompt: String,
    /// Maximum number of generated tokens; must be positive.
    #[arg(long, default_value = "8")]
    max_new_tokens: NonZeroUsize,
    /// Directory containing reference.json and logits.f32 for validation.
    #[arg(long)]
    reference_dir: Option<PathBuf>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let Args {
        model_dir,
        prompt,
        max_new_tokens,
        reference_dir,
    } = Args::parse();
    let max_new = max_new_tokens.get();
    let device = Device::new(0)?;
    let policy: Arc<dyn SchedulingPolicy> = Arc::new(SingleStream::new(&device)?);
    let load_start = Instant::now();
    let model = Qwen2Engine::load(&model_dir, &policy).await?;
    eprintln!("Loaded model in {:.2}s", load_start.elapsed().as_secs_f64());
    // Explicit two-message Qwen2 template; raw prompt text alone is not an instruct input.
    let formatted = format!(
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    );
    let mut history = model
        .tokenizer()
        .encode(formatted.as_str(), false)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?
        .get_ids()
        .to_vec();
    let input_ids = history.clone();
    let pool = NaiveKVPool::new(model.kv_pool_config(history.len() + max_new)?, &policy).await?;
    let initial = pool.metrics();
    let mut comparison = serde_json::Value::Null;
    if let Some(dir) = &reference_dir {
        let reference: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("reference.json"))?)?;
        ensure!(
            serde_json::to_value(&input_ids)? == reference["input_ids"],
            "tokenizer/template differs from reference"
        );
        let mut kv = KVContext::new(pool.clone())?;
        let start = Instant::now();
        let (next, actual) = model
            .step_with_logits(&history, &mut kv, policy.clone())
            .await?;
        let bytes = std::fs::read(dir.join("logits.f32"))?;
        ensure!(
            bytes.len() == actual.len() * 4,
            "reference logits shape differs"
        );
        let expected: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        ensure!(actual.iter().all(|v| v.is_finite()), "non-finite logits");
        let rms = (actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / actual.len() as f32)
            .sqrt();
        ensure!(max_error < 0.001, "logit max error too large: {max_error}");
        ensure!(
            next == reference["generated_ids"][0].as_u64().unwrap() as u32,
            "first token differs"
        );
        comparison = serde_json::json!({"max_abs_logit_error":max_error,"rms_logit_error":rms,"first_token":next,"prefill_with_jit_seconds":start.elapsed().as_secs_f64()});
        kv.release();
        ensure!(pool.metrics() == initial, "validation KV was not released");
    }
    let mut kv = KVContext::new(pool.clone())?;
    let start = Instant::now();
    let generated: Vec<u32> = model
        .generate_greedy(&mut history, &mut kv, max_new, policy.clone())
        .try_collect()
        .await?;
    let elapsed = start.elapsed().as_secs_f64();
    // One sampled CPU token is still uncached, ready for a subsequent continuation.
    ensure!(
        kv.ready_len() + 1 == history.len(),
        "unexpected token/KV progress"
    );
    if let Some(dir) = reference_dir {
        let reference: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("reference.json"))?)?;
        ensure!(
            serde_json::to_value(&generated)? == reference["generated_ids"],
            "greedy sequence differs from CPU reference: {generated:?}"
        );
    }
    let text = model
        .tokenizer()
        .decode(&generated, true)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let ready = kv.ready_len();
    kv.release();
    ensure!(pool.metrics() == initial, "generation KV was not released");
    println!(
        "{}",
        serde_json::json!({"model":"Qwen2.5-0.5B-Instruct","compute_dtype":"f32","input_ids":input_ids,
        "generated_ids":generated,"text":text,"ready_len_before_release":ready,"elapsed_seconds":elapsed,
        "validation":comparison,"kv_released":true})
    );
    Ok(())
}
