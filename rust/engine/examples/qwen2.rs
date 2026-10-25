//! Qwen2.5-0.5B-Instruct through Scheduler -> Runner -> cuTile model.
//! From the repository root (MODEL_DIR is the existing HF snapshot directory):
//! cargo run --manifest-path rust/Cargo.toml --offline --release -p engine --example qwen2 -- MODEL_DIR
//! Match --max-new-tokens to the reference run; e.g. --max-new-tokens 32
//! --reference-dir rust/engine/artifacts/qwen2 validates CPU reference logits and generated text.

use anyhow::{ensure, Result};
use clap::Parser;
use cuda_async::scheduling_policies::{SchedulingPolicy, SingleStream};
use cutile::cuda_core::Device;
use engine::runner::{
    kv_pool::{KVContext, KVPool, NaiveKVPool},
    model::Qwen2Engine,
    model_runner::RunnerBuilder,
};
use engine::{
    scheduler::Scheduler,
    types::{SamplingConfig, SessionRequest},
    CausalModel,
};
use std::{num::NonZeroUsize, path::PathBuf, sync::Arc, time::Instant};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

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
    let encoded = model
        .tokenizer()
        .encode(formatted.as_str(), false)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let input_ids = encoded.get_ids();
    let pool = NaiveKVPool::new(model.kv_pool_config(input_ids.len() + max_new)?, &policy).await?;
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
            .step_with_logits(input_ids, &mut kv, policy.clone())
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
    let cancel = CancellationToken::new();
    let (commands, command_rx) = mpsc::unbounded_channel();
    let (replies, reply_rx) = mpsc::unbounded_channel();
    let scheduler = Scheduler::new(cancel.child_token(), commands, reply_rx);
    let owned_pool = pool.clone();
    let runner = RunnerBuilder::default()
        .with_model_factory(move || async move { Ok(model) })
        .with_pool_factory(move |_| async move { Ok(owned_pool) })
        .with_policy_factory(move || Ok(policy))
        .build(command_rx, replies)
        .await?;
    let task = tokio::spawn(runner.run(cancel.child_token()));
    let mut request = SessionRequest::new(None, formatted);
    request.sampling = Some(SamplingConfig {
        max_new_tokens: Some(max_new),
        ..Default::default()
    });
    let start = Instant::now();
    let result = scheduler.execute(request).await;
    let elapsed = start.elapsed().as_secs_f64();
    cancel.cancel();
    scheduler.wait_stopped().await;
    task.await?;
    let response = result?;
    if let Some(dir) = reference_dir {
        let reference: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("reference.json"))?)?;
        ensure!(
            response.output == reference["text"].as_str().unwrap(),
            "Runner output differs from CPU reference"
        );
    }
    ensure!(pool.metrics() == initial, "generation KV was not released");
    println!(
        "{}",
        serde_json::json!({"model":"Qwen2.5-0.5B-Instruct","compute_dtype":"f32","input_ids":input_ids,
        "request_id":response.request_id,"text":response.output,"elapsed_seconds":elapsed,
        "validation":comparison,"kv_released":true})
    );
    Ok(())
}
