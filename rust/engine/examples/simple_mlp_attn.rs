//! Uniform eager batches through Scheduler -> Runner -> attention MLP.
//! Use --profile with nsys --capture-range=cudaProfilerApi to exclude initialization/JIT.
//!
//! Run from the repository root:
//! ```bash
//! cargo build --manifest-path rust/Cargo.toml --offline --release -p engine --example simple_mlp_attn
//! rust/target/release/examples/simple_mlp_attn --warmup 10 --iterations 50
//! mkdir -p rust/profiling/simple_mlp_attn
//! nsys profile --trace=cuda,nvtx,osrt --gpu-metrics-devices=0 --capture-range=cudaProfilerApi --capture-range-end=stop --sample=none --cpuctxsw=none --event-sample=none --backtrace=none --force-overwrite=true --output=rust/profiling/simple_mlp_attn/uniform-b5-t16 rust/target/release/examples/simple_mlp_attn --warmup 10 --iterations 50 --profile
//! nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_api_sum,cuda_kern_exec_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum --format csv --output rust/profiling/simple_mlp_attn/stats rust/profiling/simple_mlp_attn/uniform-b5-t16.nsys-rep
//! ```
//! Open uniform-b5-t16.nsys-rep directly in Nsight Systems via File -> Open.

use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::{bail, ensure, Result};
use cuda_async::scheduling_policies::{SchedulingPolicy, SingleStream};
use cutile::cuda_core::Device;
use engine::{
    runner::{
        kv_pool::{KVPool, NaiveKVPool, NaiveKVPoolConfig},
        model::{Model, ModelError, ModelInput, ModelOutput, SimpleMLPEngine},
        model_runner::RunnerBuilder,
    },
    scheduler::Scheduler,
    types::{BatchIn, SamplingConfig, SessionRequest},
};
use tokio::{sync::mpsc, task::JoinSet};
use tokio_util::sync::CancellationToken;

struct Options {
    batch_size: usize,
    tokens: usize,
    warmup: usize,
    iterations: usize,
    profile: bool,
}

impl Options {
    fn parse() -> Result<Self> {
        let mut options = Self {
            batch_size: 5,
            tokens: 16,
            warmup: 3,
            iterations: 20,
            profile: false,
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            if arg == "--profile" {
                options.profile = true;
                continue;
            }
            if arg == "--help" {
                println!("simple_mlp_attn [--batch-size 1..5] [--tokens 1..512] [--warmup N] [--iterations N] [--profile]");
                println!("Batch size 5 fills the Scheduler dispatch limit; smaller batches include its 100ms collection window.");
                std::process::exit(0);
            }
            let value: usize = args
                .next()
                .ok_or_else(|| anyhow::anyhow!("missing value for {arg}"))?
                .parse()?;
            match arg.as_str() {
                "--batch-size" => options.batch_size = value,
                "--tokens" => options.tokens = value,
                "--warmup" => options.warmup = value,
                "--iterations" => options.iterations = value,
                _ => bail!("unknown option {arg}"),
            }
        }
        ensure!(
            (1..=5).contains(&options.batch_size),
            "batch size must be 1..5"
        );
        ensure!((1..=512).contains(&options.tokens), "tokens must be 1..512");
        ensure!(
            options.warmup > 0 && options.iterations > 0,
            "warmup and iterations must be positive"
        );
        Ok(options)
    }
}

/// Example-only timing excludes Scheduler queueing/tokenization and Runner decoding.
struct TimedModel {
    inner: SimpleMLPEngine,
    samples: Arc<Mutex<Vec<(usize, f64)>>>,
}

impl Model<NaiveKVPool> for TimedModel {
    fn tokenizer(&self) -> &tokenizers::Tokenizer {
        self.inner.tokenizer()
    }

    fn validate_input(&self, ids: &[u32], sampling: &SamplingConfig) -> Result<(), ModelError> {
        self.inner.validate_input(ids, sampling)
    }

    async fn generate(
        &self,
        batch: BatchIn<ModelInput<'_, NaiveKVPool>>,
        policy: Arc<dyn SchedulingPolicy>,
    ) -> Vec<ModelOutput> {
        let size = batch.len();
        let start = Instant::now();
        let output = self.inner.generate(batch, policy).await;
        self.samples
            .lock()
            .unwrap()
            .push((size, start.elapsed().as_secs_f64() * 1e6));
        output
    }
}

async fn uniform_batch(scheduler: &Scheduler, prompt: &str, size: usize) -> Result<String> {
    let mut callers = JoinSet::new();
    for _ in 0..size {
        let scheduler = scheduler.clone();
        let request = SessionRequest::new(None, prompt.to_owned());
        callers.spawn(async move { scheduler.execute(request).await });
    }
    let mut reference = None;
    while let Some(result) = callers.join_next().await {
        let output = result??.output;
        ensure!(
            output.parse::<f32>()?.is_finite(),
            "non-finite model output"
        );
        if let Some(expected) = &reference {
            ensure!(
                &output == expected,
                "uniform requests produced different outputs"
            );
        } else {
            reference = Some(output);
        }
    }
    Ok(reference.unwrap())
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let options = Options::parse()?;
    let device = Device::new(0)?;
    // SAFETY: load the installed CUDA driver and its documented no-argument profiler APIs.
    // Dynamic loading avoids requiring a link-time libcuda.so search path under WSL.
    let driver = unsafe { libloading::Library::new("libcuda.so.1")? };
    let profiler_start =
        unsafe { driver.get::<unsafe extern "C" fn() -> i32>(b"cuProfilerStart\0")? };
    let profiler_stop =
        unsafe { driver.get::<unsafe extern "C" fn() -> i32>(b"cuProfilerStop\0")? };
    let policy: Arc<dyn SchedulingPolicy> = Arc::new(SingleStream::new(&device)?);
    let pool = NaiveKVPool::new(NaiveKVPoolConfig::default(), &policy).await?;
    let initial = pool.metrics();
    let samples = Arc::new(Mutex::new(Vec::new()));
    let timings = samples.clone();
    let owned_pool = pool.clone();
    let cancel = CancellationToken::new();
    let (command_tx, command_rx) = mpsc::unbounded_channel();
    let (reply_tx, reply_rx) = mpsc::unbounded_channel();
    let scheduler = Scheduler::new(cancel.child_token(), command_tx, reply_rx);
    let runner = RunnerBuilder::default()
        .with_model_factory(move || async move {
            Ok(TimedModel {
                inner: SimpleMLPEngine::random(0).await?,
                samples: timings,
            })
        })
        .with_pool_factory(move |_| async move { Ok(owned_pool) })
        .with_policy_factory(move || Ok(policy))
        .build(command_rx, reply_tx)
        .await?;
    let runner_task = tokio::spawn(runner.run(cancel.child_token()));
    let prompt = (0..options.tokens)
        .map(|i| (i % 256).to_string())
        .collect::<Vec<_>>()
        .join(" ");
    let mut profiling = false;
    let result = async {
        let mut expected = String::new();
        for _ in 0..options.warmup {
            expected = uniform_batch(&scheduler, &prompt, options.batch_size).await?;
        }
        samples.lock().unwrap().clear();
        if options.profile {
            device.bind_to_thread()?;
            // SAFETY: the CUDA context is current and all warmup requests have completed.
            ensure!(unsafe { profiler_start() } == 0, "cuProfilerStart failed");
            profiling = true;
        }
        let mut batch_times = Vec::new();
        for _ in 0..options.iterations {
            let start = Instant::now();
            let output = uniform_batch(&scheduler, &prompt, options.batch_size).await?;
            batch_times.push(start.elapsed().as_secs_f64() * 1e6);
            ensure!(
                output == expected,
                "steady-state result differs from warmup"
            );
            ensure!(
                pool.metrics() == initial,
                "stateless batch retained KV allocations"
            );
        }
        Ok::<_, anyhow::Error>((expected, batch_times))
    }
    .await;
    let stop = if profiling {
        // SAFETY: all measured requests completed before ending the capture range.
        unsafe { profiler_stop() }
    } else {
        0
    };
    cancel.cancel();
    scheduler.wait_stopped().await;
    runner_task.await?;
    let (output, mut batch_times) = result?;
    ensure!(stop == 0, "cuProfilerStop failed");
    let samples = samples.lock().unwrap();
    ensure!(
        samples.len() == options.iterations,
        "Scheduler split the uniform batch unexpectedly"
    );
    ensure!(
        samples.iter().all(|(size, _)| *size == options.batch_size),
        "nonuniform execution batch"
    );
    let mut model_times: Vec<_> = samples.iter().map(|(_, us)| *us).collect();
    println!(
        "{}",
        serde_json::json!({
            "batch_size": options.batch_size, "prompt_tokens": options.tokens,
            "packed_rows": (options.batch_size * options.tokens).div_ceil(4) * 4,
            "warmup": options.warmup, "iterations": options.iterations, "profiled": options.profile,
            "input_dim": 16, "hidden_dim": 16, "dtype": "f32", "uniform_output": output,
            "model_generate_median_us": median(&mut model_times),
            "model_generate_mean_us": model_times.iter().sum::<f64>() / model_times.len() as f64,
            "batch_end_to_end_median_us": median(&mut batch_times),
            "pool_index_bytes": initial.index.allocated_bytes, "pool_kv_bytes": initial.layers.allocated_bytes,
            "kv_released_after_each_batch": true
        })
    );
    Ok(())
}
