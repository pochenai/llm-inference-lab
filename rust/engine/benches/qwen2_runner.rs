//! Fixed-cohort Scheduler -> Runner serial/overlap comparison (FP32, eager).
//! Run from the repository root, replacing MODEL_DIR with the local HF snapshot:
//! ```bash
//! cargo bench --manifest-path rust/Cargo.toml --offline --locked -p engine --bench qwen2_runner --no-run
//! cargo bench --manifest-path rust/Cargo.toml --offline --locked -p engine --bench qwen2_runner -- --model-dir MODEL_DIR --mode serial --tokens 32 --batch-size 4 --output-tokens 64 --warmup 2 --iterations 3
//! cargo bench --manifest-path rust/Cargo.toml --offline --locked -p engine --bench qwen2_runner -- --model-dir MODEL_DIR --mode overlap --tokens 32 --batch-size 4 --output-tokens 64 --warmup 2 --iterations 3
//! ```
//! Use the executable printed by --no-run for profiling; warmup is outside capture:
//! ```bash
//! nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --event-sample=none --backtrace=none --capture-range=cudaProfilerApi --capture-range-end=stop --output=rust/profiling/qwen2/runner-overlap/trace-overlap BENCH_PATH --model-dir MODEL_DIR --mode overlap --tokens 32 --batch-size 4 --output-tokens 64 --warmup 2 --iterations 1 --profile
//! ```
//! Every cohort has fresh KV and a full uniform prefill (no chunking), followed by
//! output_tokens-1 decode forwards. EOS stopping is disabled only by this bench's
//! model wrapper. Real tokenization/text decoding remain enabled. All requests are
//! synchronously admitted before polling responses; batch-shape assertions reject
//! accidental split cohorts. Wall time includes ingress, tokenization and text delivery;
//! time-to-first-text is not pure prefill time. Workspace peaks are model-lifetime
//! snapshots; CUDA pool high-water marks reset per cohort. No HTTP is measured.

use std::{
    num::NonZeroUsize,
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Instant,
};

use anyhow::{ensure, Result};
use clap::{Parser, ValueEnum};
use cuda_async::scheduling_policies::{SchedulingPolicy, SingleStream};
use cutile::cuda_core::Device;
use engine::{
    runner::{
        kv_pool::{KVPool, NaiveKVPool},
        model::Qwen2Engine,
        model_runner::RunnerBuilder,
    },
    scheduler::{RequestStats, Scheduler},
    types::{SamplingConfig, SessionRequest},
    CausalModel, ModelError, ModelInput, NextToken, SubmittedGpuWorkWithCompletion,
};
use futures::{future::try_join_all, TryStreamExt};
use serde::Serialize;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
enum Mode {
    Serial,
    Overlap,
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long = "bench", hide = true)]
    _bench: bool,
    #[arg(long, value_enum, default_value = "overlap")]
    mode: Mode,
    #[arg(long, default_value = "32")]
    tokens: NonZeroUsize,
    #[arg(long, default_value = "64")]
    output_tokens: NonZeroUsize,
    #[arg(long, default_value = "1")]
    batch_size: NonZeroUsize,
    #[arg(long, default_value = "2")]
    warmup: NonZeroUsize,
    #[arg(long, default_value = "3")]
    iterations: NonZeroUsize,
    #[arg(long)]
    profile: bool,
    #[arg(long)]
    trace_steps: bool,
}

/// Bench-only fixed lengths and shape checks; forwards still execute the real model.
struct FixedCohortModel {
    inner: Arc<Qwen2Engine>,
    calls: Arc<AtomicUsize>,
    batch_size: usize,
    input_tokens: usize,
    output_tokens: usize,
    mode: Mode,
}

impl FixedCohortModel {
    fn begin_forward(
        &self,
        batch: &[ModelInput<'_, NaiveKVPool>],
        mode: Mode,
    ) -> nvtx_sys::RangeId {
        assert_eq!(self.mode, mode);
        assert_eq!(
            batch.len(),
            self.batch_size,
            "cohort split into different GPU batches"
        );
        let step = self.calls.fetch_add(1, Ordering::Relaxed);
        let expected = if step == 0 { self.input_tokens } else { 1 };
        assert!(step < self.output_tokens, "unexpected extra forward");
        assert!(
            batch.iter().all(|input| input.uncached_len() == expected),
            "unexpected prefill chunk or decode shape"
        );
        // Start/end ranges permit awaits; thread-local push/pop ranges do not.
        nvtx_sys::range_start_ascii(match mode {
            Mode::Serial => c"runner_bench.serial.forward",
            Mode::Overlap => c"runner_bench.overlap.enqueue",
        })
    }
}

impl CausalModel<NaiveKVPool> for FixedCohortModel {
    fn tokenizer(&self) -> &tokenizers::Tokenizer {
        self.inner.tokenizer()
    }

    fn validate_input(&self, ids: &[u32]) -> Result<(), ModelError> {
        self.inner.validate_input(ids)
    }

    /// Fixed-length experiment: always generate the requested budget, including EOS IDs.
    fn is_eos(&self, _token: u32) -> bool {
        false
    }

    async fn forward(
        &self,
        batch: &mut [ModelInput<'_, NaiveKVPool>],
        policy: Arc<dyn SchedulingPolicy>,
    ) -> Vec<Result<NextToken, ModelError>> {
        let range = self.begin_forward(batch, Mode::Serial);
        let output = self.inner.forward(batch, policy).await;
        nvtx_sys::range_end(range);
        output
    }

    async fn enqueue_forward_and_return_completion(
        self: Arc<Self>,
        batch: &mut [ModelInput<'_, NaiveKVPool>],
        policy: Arc<dyn SchedulingPolicy>,
    ) -> SubmittedGpuWorkWithCompletion {
        let range = self.begin_forward(batch, Mode::Overlap);
        let output = self
            .inner
            .clone()
            .enqueue_forward_and_return_completion(batch, policy)
            .await;
        nvtx_sys::range_end(range);
        output
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .json()
        .with_writer(std::io::stderr)
        .with_max_level(if args.trace_steps {
            tracing::Level::DEBUG
        } else {
            tracing::Level::INFO
        })
        .try_init()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let device = Device::new(0)?;
    let policy: Arc<dyn SchedulingPolicy> = Arc::new(SingleStream::new(&device)?);
    let model = Arc::new(Qwen2Engine::load(&args.model_dir, &policy).await?);
    let prefix = vec![9707u32; args.tokens.get()];
    let prompt = model
        .tokenizer()
        .decode(&prefix, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let encoded = model
        .tokenizer()
        .encode(prompt.as_str(), true)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    ensure!(
        encoded.get_ids() == prefix,
        "synthetic prompt must round-trip to identical token IDs"
    );
    let max_seq_len = args
        .tokens
        .get()
        .checked_add(args.output_tokens.get() - 1)
        .ok_or_else(|| anyhow::anyhow!("context length overflow"))?;
    let mut config = model.kv_pool_config(max_seq_len)?;
    config.max_rows = args.batch_size.get();
    config.num_slots = config
        .num_slots
        .checked_mul(config.max_rows)
        .ok_or_else(|| anyhow::anyhow!("batch capacity overflow"))?;
    let pool = NaiveKVPool::new(config, &policy).await?;
    let initial_kv = pool.metrics();
    let cuda_pool = device.default_mem_pool()?;
    let calls = Arc::new(AtomicUsize::new(0));
    let stats = Arc::new(Mutex::new(Vec::<RequestStats>::new()));
    let captured = stats.clone();
    let cancel = CancellationToken::new();
    let (commands, command_rx) = mpsc::unbounded_channel();
    let (replies, reply_rx) = mpsc::unbounded_channel();
    let scheduler = Scheduler::new_with_observers(
        cancel.child_token(),
        commands,
        reply_rx,
        vec![Arc::new(
            move |_id, stat: &RequestStats, _result: &Result<(), engine::error::EngineError>| {
                captured.lock().unwrap().push(*stat);
            },
        )],
    );
    let wrapper = FixedCohortModel {
        inner: model.clone(),
        calls: calls.clone(),
        batch_size: args.batch_size.get(),
        input_tokens: args.tokens.get(),
        output_tokens: args.output_tokens.get(),
        mode: args.mode,
    };
    let owned_pool = pool.clone();
    let budget = args
        .tokens
        .get()
        .checked_mul(args.batch_size.get())
        .and_then(NonZeroUsize::new)
        .ok_or_else(|| anyhow::anyhow!("token budget overflow"))?;
    let runner = RunnerBuilder::default()
        .with_overlap(args.mode == Mode::Overlap)
        .with_max_batch_tokens(budget)
        .with_model_factory(move || async move { Ok(wrapper) })
        .with_pool_factory(move |_| async move { Ok(owned_pool) })
        .with_policy_factory(move || Ok(policy))
        .build(command_rx, replies)
        .await?;
    let task = tokio::spawn(runner.run(cancel.child_token()));
    // The driver handle remains alive while profiler symbols are used.
    let driver = unsafe { libloading::Library::new("libcuda.so.1")? };
    let profiler_start =
        unsafe { driver.get::<unsafe extern "C" fn() -> i32>(b"cuProfilerStart\0")? };
    let profiler_stop =
        unsafe { driver.get::<unsafe extern "C" fn() -> i32>(b"cuProfilerStop\0")? };
    let mut profiling = false;
    let result: Result<()> = async {
        let mut expected = None;
        let mut elapsed_total = 0.0;
        let mut first_text = Vec::new();
        println!("{}", json!({"event":"configuration", "mode":args.mode, "dtype":"f32", "cuda_graph":false,
            "input_tokens":args.tokens.get(), "batch_size":args.batch_size.get(), "output_tokens":args.output_tokens.get(),
            "warmup":args.warmup.get(), "iterations":args.iterations.get(), "eos_stopping":false, "profile":args.profile}));
        for iteration in 0..args.warmup.get() + args.iterations.get() {
            let warmup = iteration < args.warmup.get();
            if args.profile && iteration == args.warmup.get() {
                device.bind_to_thread()?;
                // SAFETY: every prior cohort has completed and released its KV.
                ensure!(unsafe { profiler_start() } == 0, "cuProfilerStart failed");
                profiling = true;
            }
            stats.lock().unwrap().clear();
            calls.store(0, Ordering::Relaxed);
            cuda_pool.reset_used_high()?;
            cuda_pool.reset_reserved_high()?;
            let requests: Vec<_> = (0..args.batch_size.get()).map(|_| {
                let mut request = SessionRequest::new(None, prompt.clone());
                request.sampling = Some(SamplingConfig { max_new_tokens: Some(args.output_tokens.get()), ..Default::default() });
                request
            }).collect();
            let range = nvtx_sys::range_start_ascii(c"runner_bench.sample");
            let started = Instant::now();
            // execute_stream admits synchronously: all cohort commands exist before yielding.
            let streams = requests.into_iter().map(|request| scheduler.execute_stream(request)).collect::<Result<Vec<_>, _>>()?;
            let output = try_join_all(streams.into_iter().map(|stream| async move {
                futures::pin_mut!(stream);
                let mut output = String::new();
                while let Some(delta) = stream.try_next().await? { output.push_str(&delta.output); }
                Ok::<_, engine::error::EngineError>(output)
            })).await;
            let seconds = started.elapsed().as_secs_f64();
            nvtx_sys::range_end(range);
            let output = output?;
            ensure!(calls.load(Ordering::Relaxed) == args.output_tokens.get(), "wrong forward count");
            let records = stats.lock().unwrap().clone();
            ensure!(records.len() == args.batch_size.get(), "missing request stats");
            ensure!(records.iter().all(|stat| stat.generated_tokens == Some(args.output_tokens.get())), "request stopped early");
            ensure!(pool.metrics() == initial_kv, "cohort leaked KV");
            let workspace = model.workspace_metrics();
            ensure!(workspace.in_use_bytes == 0, "cohort retained scratch leases");
            if let Some(expected) = &expected { ensure!(&output == expected, "fresh-KV output changed"); }
            else { expected = Some(output.clone()); }
            let memory = cuda_pool.mem_stats()?;
            let first: Vec<_> = records.iter().map(|stat| stat.time_to_first_text.map(|d| d.as_secs_f64())).collect();
            if !warmup { elapsed_total += seconds; first_text.extend(first.iter().flatten().copied()); }
            println!("{}", json!({"event":"sample", "mode":args.mode, "warmup":warmup, "iteration":iteration,
                "cohort_wall_seconds":seconds, "output_tokens_per_second":(args.batch_size.get()*args.output_tokens.get()) as f64/seconds,
                "time_to_first_text_seconds":first, "request_elapsed_seconds":records.iter().map(|s| s.elapsed.as_secs_f64()).collect::<Vec<_>>(),
                "generated_tokens":records.iter().map(|s| s.generated_tokens).collect::<Vec<_>>(), "output_text":output,
                "forward_count":calls.load(Ordering::Relaxed), "cuda_pool_used_peak_bytes":memory.used_high,
                "cuda_pool_reserved_peak_bytes":memory.reserved_high, "workspace_pool":workspace, "kv_released":true}));
        }
        println!("{}", json!({"event":"summary", "mode":args.mode, "iterations":args.iterations.get(),
            "cohort_mean_seconds":elapsed_total/args.iterations.get() as f64,
            "output_tokens_per_second":(args.iterations.get()*args.batch_size.get()*args.output_tokens.get()) as f64/elapsed_total,
            "time_to_first_text_mean_seconds":if first_text.is_empty() { None } else { Some(first_text.iter().sum::<f64>()/first_text.len() as f64) },
            "workspace_pool":model.workspace_metrics()}));
        Ok(())
    }.await;
    cancel.cancel();
    scheduler.wait_stopped().await;
    let stopped = task.await;
    if profiling {
        device.bind_to_thread()?;
        // SAFETY: Runner shutdown drained in-flight model work.
        ensure!(unsafe { profiler_stop() } == 0, "cuProfilerStop failed");
    }
    stopped?;
    result
}
