//! Fixed-length Qwen-only benchmark, excluding Scheduler queueing/text decoding.
//! Fresh KV for every warmup/sample; model weights and scratch/upload caches stay warm.
//! Run from the repository root (MODEL_DIR is a local HF snapshot):
//! ```bash
//! cargo bench --manifest-path rust/Cargo.toml --offline -p engine --bench qwen2 --no-run
//! mkdir -p rust/profiling/qwen2
//! cargo bench --manifest-path rust/Cargo.toml --offline --locked -p engine --bench qwen2 -- --model-dir MODEL_DIR --tokens 4096 --decode-steps 8 --warmup 1 --iterations 3 --validate-chunked > rust/profiling/qwen2/4096.jsonl
//! ```
//! Add --profile under nsys --capture-range=cudaProfilerApi to capture measured
//! iterations only. Use the executable path printed by the --no-run command:
//! ```bash
//! nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --event-sample=none --backtrace=none --capture-range=cudaProfilerApi --capture-range-end=stop --output=rust/profiling/qwen2/packed-b1-t32 BENCH_PATH --model-dir MODEL_DIR --tokens 32 --batch-size 1 --decode-steps 8 --warmup 3 --iterations 5 --profile
//! ```
//! Timings are host wall time through GPU/D2H completion, not CUDA event/kernel time.
//! --trace-steps adds structured tracing JSON to stderr; omit it for latency sampling.

use std::{num::NonZeroUsize, path::PathBuf, sync::Arc, time::Instant};

use anyhow::{ensure, Result};
use clap::Parser;
use cuda_async::scheduling_policies::{SchedulingPolicy, SingleStream};
use cutile::{
    core::{bf16, ElementType},
    cuda_core::Device,
};
use engine::runner::loader::WeightElement;
use engine::{
    runner::{
        kv_pool::{KVContext, KVPool, NaiveKVPool},
        model::Qwen2Engine,
    },
    CausalModel, ModelInput,
};
use serde_json::json;

#[derive(Parser)]
struct Args {
    /// Weight, activation and KV storage; reductions and accumulation remain FP32.
    #[arg(long, default_value = "bf16", value_parser = ["bf16", "f32"])]
    dtype: String,
    #[arg(long)]
    model_dir: PathBuf,
    /// Cargo appends this flag even when the standard benchmark harness is disabled.
    #[arg(long = "bench", hide = true)]
    _bench: bool,
    #[arg(long, default_value = "4096")]
    tokens: NonZeroUsize,
    /// Number of one-token steps after prefill (does not stop at EOS).
    #[arg(long, default_value = "8")]
    decode_steps: NonZeroUsize,
    /// Number of requests sharing one packed GPU forward.
    #[arg(long, default_value = "1")]
    batch_size: NonZeroUsize,
    #[arg(long, default_value = "1")]
    warmup: NonZeroUsize,
    #[arg(long, default_value = "3")]
    iterations: NonZeroUsize,
    /// Compare full vs chunked final-position logits and next token after sampling.
    #[arg(long)]
    validate_chunked: bool,
    #[arg(long, default_value = "256")]
    chunk_size: NonZeroUsize,
    #[arg(long)]
    trace_steps: bool,
    /// Start CUDA profiler capture after warmup and stop before diagnostic validation.
    #[arg(long)]
    profile: bool,
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
    tracing_subscriber::fmt()
        .json()
        .with_writer(std::io::stderr)
        .with_max_level(if args.trace_steps {
            tracing::Level::DEBUG
        } else {
            tracing::Level::INFO
        })
        .try_init()
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let device = Device::new(0)?;
    // SAFETY: reuse the installed driver's profiler API, as in simple_mlp_attn.
    let driver = unsafe { libloading::Library::new("libcuda.so.1")? };
    let profiler_start =
        unsafe { driver.get::<unsafe extern "C" fn() -> i32>(b"cuProfilerStart\0")? };
    let profiler_stop =
        unsafe { driver.get::<unsafe extern "C" fn() -> i32>(b"cuProfilerStop\0")? };
    let policy: Arc<dyn SchedulingPolicy> = Arc::new(SingleStream::new(&device)?);
    let load_start = Instant::now();
    let model = Qwen2Engine::<T>::load(&args.model_dir, &policy).await?;
    let mut config = model.kv_pool_config(
        args.tokens
            .get()
            .checked_add(args.decode_steps.get())
            .ok_or_else(|| anyhow::anyhow!("context length overflow"))?,
    )?;
    config.max_rows = args.batch_size.get();
    config.num_slots = config
        .num_slots
        .checked_mul(config.max_rows)
        .ok_or_else(|| anyhow::anyhow!("batch capacity overflow"))?;
    let pool = NaiveKVPool::<T>::new(config, &policy).await?;
    let initial_kv = pool.metrics();
    let instance = model.workspace_metrics().model_instance.to_string();
    let used_peak =
        metrics::gauge!("qwen2_cuda_pool_used_peak_bytes", "model_instance" => instance.clone());
    let reserved_peak =
        metrics::gauge!("qwen2_cuda_pool_reserved_peak_bytes", "model_instance" => instance);
    let cuda_pool = device.default_mem_pool()?;
    println!(
        "{}",
        json!({"event":"loaded", "seconds":load_start.elapsed().as_secs_f64(),
        "tokens":args.tokens.get(), "batch_size":args.batch_size.get(), "dtype":T::DTYPE.to_string()})
    );
    // Deterministic valid vocabulary IDs; this tests capacity, not natural-language quality.
    let prefix = vec![9707u32; args.tokens.get()];
    let mut expected = None;
    let mut prefills = Vec::with_capacity(args.iterations.get());
    let mut decodes = Vec::with_capacity(args.iterations.get() * args.decode_steps.get());
    for iteration in 0..args.warmup.get() + args.iterations.get() {
        let warmup = iteration < args.warmup.get();
        if args.profile && iteration == args.warmup.get() {
            device.bind_to_thread()?;
            // SAFETY: all warmup GPU work and readbacks have completed.
            ensure!(unsafe { profiler_start() } == 0, "cuProfilerStart failed");
        }
        cuda_pool.reset_used_high()?;
        cuda_pool.reset_reserved_high()?;
        let mut contexts = (0..args.batch_size.get())
            .map(|_| KVContext::new(pool.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut histories: Vec<_> = (0..args.batch_size.get())
            .map(|_| {
                let mut ids = Vec::with_capacity(prefix.len() + args.decode_steps.get() + 1);
                ids.extend_from_slice(&prefix);
                ids
            })
            .collect();
        let mut step_seconds = Vec::with_capacity(args.decode_steps.get() + 1);
        for _ in 0..=args.decode_steps.get() {
            let mut inputs: Vec<_> = histories
                .iter()
                .zip(&mut contexts)
                .map(|(ids, kv)| ModelInput {
                    token_ids: ids.into(),
                    kv,
                })
                .collect();
            let start = Instant::now();
            let outputs = model.next_tokens(&mut inputs, policy.clone()).await;
            step_seconds.push(start.elapsed().as_secs_f64());
            for (history, output) in histories.iter_mut().zip(outputs) {
                history.push(output?);
            }
        }
        let generated: Vec<_> = histories
            .iter()
            .map(|ids| ids[prefix.len()..].to_vec())
            .collect();
        if let Some(expected) = &expected {
            ensure!(
                &generated == expected,
                "output changed across fresh-KV runs"
            );
        } else {
            expected = Some(generated.clone());
        }
        for context in contexts {
            ensure!(
                context.ready_len() == prefix.len() + args.decode_steps.get(),
                "KV length differs"
            );
            context.release();
        }
        ensure!(pool.metrics() == initial_kv, "KV leaked between samples");
        let workspace_pool = model.workspace_metrics();
        let memory = cuda_pool.mem_stats()?;
        used_peak.set(memory.used_high as f64);
        reserved_peak.set(memory.reserved_high as f64);
        let decode_seconds: f64 = step_seconds[1..].iter().sum();
        println!(
            "{}",
            json!({"event":"sample", "warmup":warmup, "iteration":iteration,
            "prefill_seconds":step_seconds[0], "decode_seconds":decode_seconds,
            "decode_step_seconds":&step_seconds[1..], "generated_ids":generated,
            "prefill_tokens_per_second":(prefix.len()*args.batch_size.get()) as f64/step_seconds[0],
            "decode_tokens_per_second":(args.decode_steps.get()*args.batch_size.get()) as f64/decode_seconds,
            "cuda_pool_used_peak_bytes":memory.used_high,"cuda_pool_reserved_peak_bytes":memory.reserved_high,
            "workspace_pool":workspace_pool,"kv_released":true})
        );
        if !warmup {
            prefills.push(step_seconds[0]);
            decodes.extend_from_slice(&step_seconds[1..]);
        }
    }
    if args.profile {
        device.bind_to_thread()?;
        // SAFETY: every measured forward and readback has completed.
        ensure!(unsafe { profiler_stop() } == 0, "cuProfilerStop failed");
    }
    let prefill_mean = prefills.iter().sum::<f64>() / prefills.len() as f64;
    let decode_total: f64 = decodes.iter().sum();
    decodes.sort_by(f64::total_cmp);
    println!(
        "{}",
        json!({"event":"summary", "iterations":args.iterations.get(),
        "prefill_mean_seconds":prefill_mean,
        "prefill_tokens_per_second":(prefix.len()*args.batch_size.get()) as f64/prefill_mean,
        "decode_total_seconds":decode_total,
        "decode_mean_step_seconds":decode_total/decodes.len() as f64,
        "decode_p50_step_seconds":decodes[(decodes.len()-1)/2],
        "decode_p95_step_seconds":decodes[(decodes.len()*95).div_ceil(100)-1],
        "decode_tokens_per_second":(decodes.len()*args.batch_size.get()) as f64/decode_total,
        "workspace_pool":model.workspace_metrics()})
    );
    if args.validate_chunked {
        let mut context = KVContext::new(pool.clone())?;
        let (full_token, full_logits) = model
            .step_with_logits(&prefix, &mut context, policy.clone())
            .await?;
        context.release();
        let mut context = KVContext::new(pool.clone())?;
        for end in (args.chunk_size.get()..prefix.len()).step_by(args.chunk_size.get()) {
            model
                .next_tokens(
                    &mut [ModelInput {
                        token_ids: (&prefix[..end]).into(),
                        kv: &mut context,
                    }],
                    policy.clone(),
                )
                .await
                .pop()
                .unwrap()?;
        }
        let (chunked_token, chunked_logits) = model
            .step_with_logits(&prefix, &mut context, policy.clone())
            .await?;
        ensure!(
            context.ready_len() == prefix.len(),
            "chunked KV length differs"
        );
        context.release();
        ensure!(pool.metrics() == initial_kv, "chunked validation leaked KV");
        let max_error =
            full_logits
                .iter()
                .zip(&chunked_logits)
                .try_fold(0.0f32, |peak, (&a, &b)| {
                    ensure!(
                        a.is_finite() && b.is_finite(),
                        "non-finite validation logits"
                    );
                    Ok::<_, anyhow::Error>(peak.max((a - b).abs()))
                })?;
        // BF16 matmul rounding can vary with the full/chunked row tile geometry.
        let tolerance = if size_of::<T>() == 2 { 0.5 } else { 1e-3 };
        ensure!(
            full_token == chunked_token && max_error < tolerance,
            "full/chunked mismatch: {max_error}"
        );
        println!(
            "{}",
            json!({"event":"chunked_validation", "tokens":prefix.len(),
            "chunk_size":args.chunk_size.get(),"next_token":full_token,"max_abs_logit_error":max_error,
            "kv_released":true,"workspace_pool":model.workspace_metrics()})
        );
    }
    Ok(())
}
