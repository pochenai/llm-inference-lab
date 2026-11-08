//! B1 BF16 pooled Qwen3.5 latency, including host token/logit validation.

use std::{path::PathBuf, time::Instant};

use anyhow::{ensure, Result};
use clap::Parser;
use cutile::core::bf16;
use engine::runner::{
    kv_pool::{KVContext, KVPool},
    model::{CausalModel, ModelInput, Qwen3_5Engine, Qwen3_5StatePool},
};
use serde_json::json;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long, default_value_t = 32)]
    tokens: usize,
    #[arg(long, default_value_t = 8)]
    output_tokens: usize,
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    #[arg(long, default_value_t = 3)]
    iterations: usize,
    /// Capture only measured forwards with Nsight Systems.
    #[arg(long)]
    profile: bool,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.tokens > 0 && args.output_tokens > 1 && args.iterations > 0,
        "tokens and iterations must be positive and output-tokens must exceed one"
    );
    let samples = args
        .warmup
        .checked_add(args.iterations)
        .ok_or_else(|| anyhow::anyhow!("sample count overflow"))?;
    let capacity = args
        .tokens
        .checked_add(args.output_tokens)
        .ok_or_else(|| anyhow::anyhow!("token capacity overflow"))?;
    // SAFETY: use the installed CUDA driver's profiler API, as in the Qwen2 bench.
    let driver = unsafe { libloading::Library::new("libcuda.so.1")? };
    let profiler_start =
        unsafe { driver.get::<unsafe extern "C" fn() -> i32>(b"cuProfilerStart\0")? };
    let profiler_stop =
        unsafe { driver.get::<unsafe extern "C" fn() -> i32>(b"cuProfilerStop\0")? };
    let policy = cuda_async::device_context::global_policy(0)?;
    let model = Qwen3_5Engine::<bf16>::load(&args.model_dir, &policy).await?;
    let pool = Qwen3_5StatePool::new(&model, model.kv_pool_config(capacity)?, &policy).await?;
    let initial = pool.metrics();
    println!(
        "{}",
        json!({"event":"loaded", "dtype":"bf16", "batch_size":1, "input_tokens":args.tokens, "output_tokens":args.output_tokens, "weight_bytes":model.weight_bytes(), "gpu_feedback":false, "logits_validation":true, "cuda_graph":false, "cpu_affinity":std::fs::read_to_string("/proc/self/status")?.lines().find(|line| line.starts_with("Cpus_allowed_list:")).unwrap_or("")})
    );
    let mut expected = None;
    for iteration in 0..samples {
        if args.profile && iteration == args.warmup {
            policy.next_stream()?.context().bind_to_thread()?;
            // SAFETY: warmup forwards and readbacks have completed.
            ensure!(unsafe { profiler_start() } == 0, "cuProfilerStart failed");
        }
        let mut kv = KVContext::new(pool.clone())?;
        let mut history = vec![9707u32; args.tokens];
        let mut generated = Vec::with_capacity(args.output_tokens);
        let mut times = Vec::with_capacity(args.output_tokens);
        for _ in 0..args.output_tokens {
            let started = Instant::now();
            let mut output = model
                .forward(
                    &mut [ModelInput {
                        token_ids: history.as_slice().into(),
                        kv: &mut kv,
                    }],
                    policy.clone(),
                )
                .await;
            ensure!(output.len() == 1, "expected one prediction");
            let token = output.pop().unwrap()?.read().await?;
            times.push(started.elapsed().as_secs_f64());
            generated.push(token);
            history.push(token);
        }
        if let Some(expected) = &expected {
            ensure!(&generated == expected, "fresh-state generation differs");
        } else {
            expected = Some(generated.clone());
        }
        kv.release();
        ensure!(
            pool.metrics() == initial,
            "session resources were not released"
        );
        println!(
            "{}",
            json!({"event":"sample", "iteration":iteration, "warmup":iteration < args.warmup, "batch_size":1, "input_tokens":args.tokens, "forward_count":args.output_tokens, "prefill_seconds":times[0], "decode_step_seconds":&times[1..], "generated_ids":generated, "kv_released":true})
        );
    }
    if args.profile {
        policy.next_stream()?.context().bind_to_thread()?;
        // SAFETY: measured forwards and readbacks have completed.
        ensure!(unsafe { profiler_stop() } == 0, "cuProfilerStop failed");
    }
    Ok(())
}
