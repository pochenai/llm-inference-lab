//! Qwen2.5-0.5B-Instruct through Scheduler -> Runner -> cuTile model.
//! From the repository root (MODEL_DIR is the existing HF snapshot directory):
//! cargo run --manifest-path rust/Cargo.toml --offline --release -p engine --example qwen2 -- --model-dir MODEL_DIR
//! Match --max-new-tokens to the reference run; e.g. --max-new-tokens 32
//! --dtype f32 --reference-dir rust/engine/artifacts/qwen2 validates the CPU/FP32 reference.
//! Add --prefix-cache --repeat 2 --trace-steps to verify cross-request token/KV reuse.
//! Use qwen2_reference.py --dtype bf16 --device cuda --attention sdpa for BF16 fixtures.

use anyhow::{ensure, Result};
use clap::Parser;
use cuda_async::scheduling_policies::{SchedulingPolicy, SingleStream};
use cutile::{
    core::{bf16, ElementType},
    cuda_core::Device,
};
use engine::runner::loader::WeightElement;
use engine::runner::{
    kv_pool::{KVContext, KVPool, NaiveKVPool},
    model::Qwen2Engine,
    model_runner::RunnerBuilder,
    PrefixCacheConfig,
};
use engine::{
    scheduler::{MetricsObserver, RequestStats, Scheduler},
    types::{SamplingConfig, SessionRequest},
    CausalModel,
};
use std::{
    num::NonZeroUsize,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Instant,
};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Run offline Qwen2.5-0.5B-Instruct inference with cuTile kernels.
#[derive(Parser)]
struct Args {
    /// Weight, activation and KV storage; reductions and accumulation remain FP32.
    #[arg(long, default_value = "bf16", value_parser = ["bf16", "f32"])]
    dtype: String,
    /// Local Hugging Face checkpoint directory.
    #[arg(long)]
    model_dir: PathBuf,
    /// User message to include in the chat template.
    #[arg(long, default_value = "Say hello.")]
    prompt: String,
    /// Maximum number of generated tokens; must be positive.
    #[arg(long, default_value = "8")]
    max_new_tokens: NonZeroUsize,
    /// Maximum uncached input tokens per Runner forward, shared by decode and prefill.
    #[arg(long, default_value = "512")]
    max_batch_tokens: NonZeroUsize,
    /// Wait for CPU token processing before the next forward (serial baseline).
    #[arg(long)]
    disable_overlap: bool,
    /// Directory containing reference.json and logits.f32 for validation.
    #[arg(long)]
    reference_dir: Option<PathBuf>,
    /// Retain immutable token prefixes across independent requests on this Runner.
    #[arg(long)]
    prefix_cache: bool,
    /// Repeat the same stateless prompt, checking output equality (prefix-cache demo).
    #[arg(long, default_value = "1")]
    repeat: NonZeroUsize,
    /// Enable fixed-buffer decode CUDA Graph replay; prefill stays eager.
    #[arg(long)]
    cuda_graph: bool,
    /// Emit per-step/batch JSON metrics to stderr (includes cold JIT).
    #[arg(long)]
    trace_steps: bool,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();
    match args.dtype.as_str() {
        "bf16" => run::<bf16>(args).await,
        "f32" => run::<f32>(args).await,
        _ => unreachable!(),
    }
}

async fn run<T: WeightElement + ElementType>(args: Args) -> Result<()> {
    let Args {
        dtype: _,
        model_dir,
        prompt,
        max_new_tokens,
        max_batch_tokens,
        disable_overlap,
        reference_dir,
        trace_steps,
        cuda_graph,
        prefix_cache,
        repeat,
    } = args;
    tracing_subscriber::fmt()
        .json()
        .with_writer(std::io::stderr)
        .with_max_level(if trace_steps {
            tracing::Level::DEBUG
        } else {
            tracing::Level::INFO
        })
        .try_init()
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let max_new = max_new_tokens.get();
    let device = Device::new(0)?;
    let policy: Arc<dyn SchedulingPolicy> = Arc::new(SingleStream::new(&device)?);
    let load_start = Instant::now();
    let model = Qwen2Engine::<T>::load(&model_dir, &policy).await?;
    let model = if cuda_graph {
        model.with_cuda_graphs()
    } else {
        model
    };
    tracing::info!(target: "engine::qwen2", event = "model_loaded",
        seconds = load_start.elapsed().as_secs_f64());
    // Explicit two-message Qwen2 template; raw prompt text alone is not an instruct input.
    let formatted = format!(
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    );
    let encoded = model
        .tokenizer()
        .encode(formatted.as_str(), false)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let input_ids = encoded.get_ids();
    let mut pool_config = model.kv_pool_config(input_ids.len() + max_new)?;
    if prefix_cache {
        // Room for retained KV plus a cold request/suffix; cached prefixes consume no rows.
        pool_config.num_slots *= 2;
    }
    let pool = NaiveKVPool::<T>::new(pool_config, &policy).await?;
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
        tracing::info!(target: "engine::qwen2", event = "reference_error", max_error, rms);
        ensure!(
            max_error < if size_of::<T>() == 2 { 0.5 } else { 0.001 },
            "logit max error too large: {max_error}"
        );
        let expected_token = reference["generated_ids"][0].as_u64().unwrap() as u32;
        ensure!(
            next == expected_token,
            "first token differs: actual={next} {:?}, expected={expected_token} {:?}; reference logits {:?}, {:?}",
            actual.get(next as usize), actual.get(expected_token as usize),
            expected.get(next as usize), expected.get(expected_token as usize),
        );
        comparison = serde_json::json!({"max_abs_logit_error":max_error,"rms_logit_error":rms,"first_token":next,"prefill_with_jit_seconds":start.elapsed().as_secs_f64()});
        kv.release();
        ensure!(pool.metrics() == initial, "validation KV was not released");
    }
    let cancel = CancellationToken::new();
    let (commands, command_rx) = mpsc::unbounded_channel();
    let (replies, reply_rx) = mpsc::unbounded_channel();
    let request_stats = Arc::new(Mutex::new(None));
    let captured = request_stats.clone();
    let scheduler = Scheduler::new_with_observers(
        cancel.child_token(),
        commands,
        reply_rx,
        vec![
            Arc::new(MetricsObserver::default()),
            // This optional completion-only observer keeps the CLI's JSON summary.
            Arc::new(
                move |_id: u64,
                      stats: &RequestStats,
                      _result: &Result<(), engine::error::EngineError>| {
                    *captured.lock().unwrap() = Some(*stats);
                },
            ),
        ],
    );
    let owned_pool = pool.clone();
    let builder = RunnerBuilder::default()
        .with_max_batch_tokens(max_batch_tokens)
        .with_overlap(!disable_overlap)
        .with_model_factory(move || async move { Ok(model) })
        .with_pool_factory(move |_| async move { Ok(owned_pool) })
        .with_policy_factory(move || Ok(policy));
    let builder = if prefix_cache {
        builder.with_prefix_cache(PrefixCacheConfig::default())
    } else {
        builder
    };
    let runner = builder.build(command_rx, replies).await?;
    let task = tokio::spawn(runner.run(cancel.child_token()));
    let mut request = SessionRequest::new(None, formatted);
    request.sampling = Some(SamplingConfig {
        max_new_tokens: Some(max_new),
        ..Default::default()
    });
    let repeated_request = (repeat.get() > 1).then(|| request.clone());
    let result: Result<_> = async {
        let mut response = scheduler.execute(request).await?;
        for _ in 1..repeat.get() {
            let next = scheduler
                .execute(repeated_request.as_ref().unwrap().clone())
                .await?;
            ensure!(
                next.output == response.output,
                "repeated independent request output differs"
            );
            response = next;
        }
        Ok(response)
    }
    .await;
    cancel.cancel();
    scheduler.wait_stopped().await;
    task.await?;
    let response = result?;
    let stats = request_stats
        .lock()
        .unwrap()
        .take()
        .expect("completion observer ran before delivery");
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
        serde_json::json!({"model":"Qwen2.5-0.5B-Instruct","compute_dtype":T::DTYPE.to_string(),"input_ids":input_ids,
        "request_id":response.request_id,"text":response.output,"elapsed_seconds":stats.elapsed.as_secs_f64(),
        "time_to_first_text_seconds":stats.time_to_first_text.map(|elapsed| elapsed.as_secs_f64()),
        "generated_tokens":stats.generated_tokens,"tokens_per_second":stats.tokens_per_second(),"validation":comparison,"prefix_cache":prefix_cache,"requests":repeat.get(),"kv_released":true})
    );
    Ok(())
}
