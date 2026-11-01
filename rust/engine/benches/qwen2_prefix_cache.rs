//! Fixed-cohort Scheduler -> Runner prefix-cache comparison; optimized builds only.
//! Run from the repository root, replacing MODEL_DIR with the local HF snapshot:
//! ```bash
//! cargo bench --manifest-path rust/Cargo.toml --offline --locked -p engine --bench qwen2_prefix_cache -- --model-dir MODEL_DIR --tokens 4096 --prefix-tokens 2048 --batch-size 4 --output-tokens 1 --warmup 2 --iterations 3
//! ```
//! Repeat with --prefix-tokens 0/3584, --output-tokens 64, --mode serial or --cuda-graph.
//! Each sample creates a fresh Runner/cache, retaining model/JIT/workspace/Graph caches.
//! Cold samples start with empty KV; warm samples seed ONLY the common prefix outside
//! timing. All independent requests are admitted before yielding; suffix token IDs
//! differ at the first suffix position. Model counters verify saved work and batching.
//! Wall time includes ingress, tokenization, generation and streamed text delivery,
//! excluding Runner setup, prefix seeding and shutdown. No HTTP, sustained arrivals
//! or deliberate eviction-pressure workload is measured. First text is not GPU TTFT.
//! JSONL includes raw warmup samples; summaries exclude them. The local debugging
//! recorder is enabled in both states and drained outside the measured interval.

use std::{
    num::NonZeroUsize,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::{ensure, Result};
use clap::{Parser, ValueEnum};
use cuda_async::scheduling_policies::{SchedulingPolicy, SingleStream};
use cutile::{
    core::{bf16, ElementType},
    cuda_core::Device,
};
use engine::{
    runner::{
        kv_pool::{KVPool, NaiveKVPool},
        loader::WeightElement,
        model::Qwen2Engine,
        model_runner::RunnerBuilder,
        PrefixCacheConfig,
    },
    scheduler::{RequestStats, Scheduler},
    types::{SamplingConfig, SessionRequest},
    CausalModel, ModelError, ModelInput, NextToken, SubmittedGpuWorkWithCompletion,
};
use futures::{future::try_join_all, TryStreamExt};
use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
enum CacheState {
    Cold,
    Warm,
    Both,
}

#[derive(Parser, Serialize)]
struct Args {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long, default_value = "bf16", value_parser = ["bf16", "f32"])]
    dtype: String,
    #[arg(long, default_value = "4096")]
    tokens: NonZeroUsize,
    #[arg(long, default_value_t = 2048)]
    prefix_tokens: usize,
    #[arg(long, default_value = "1")]
    output_tokens: NonZeroUsize,
    #[arg(long, default_value = "1")]
    batch_size: NonZeroUsize,
    #[arg(long, default_value = "4096")]
    cache_tokens: NonZeroUsize,
    /// Override physical pool capacity for matched cross-engine experiments.
    #[arg(long)]
    num_slots: Option<NonZeroUsize>,
    #[arg(long, value_enum, default_value = "both")]
    cache_state: CacheState,
    #[arg(long, value_enum, default_value = "overlap")]
    mode: Mode,
    #[arg(long)]
    cuda_graph: bool,
    #[arg(long, default_value = "2")]
    warmup: NonZeroUsize,
    #[arg(long, default_value = "3")]
    iterations: NonZeroUsize,
    #[arg(long)]
    trace_steps: bool,
    #[arg(long = "bench", hide = true)]
    _bench: bool,
}

/// Bench-only fixed output length; all computation and KV handling use the real model.
struct FixedLengthModel<T: WeightElement + ElementType>(Arc<Qwen2Engine<T>>);

impl<T: WeightElement + ElementType> CausalModel<NaiveKVPool<T>> for FixedLengthModel<T> {
    fn tokenizer(&self) -> &tokenizers::Tokenizer {
        self.0.tokenizer()
    }
    fn validate_input(&self, ids: &[u32]) -> Result<(), ModelError> {
        self.0.validate_input(ids)
    }
    fn is_eos(&self, _token: u32) -> bool {
        false
    }

    async fn forward(
        &self,
        batch: &mut [ModelInput<'_, NaiveKVPool<T>>],
        policy: Arc<dyn SchedulingPolicy>,
    ) -> Vec<Result<NextToken, ModelError>> {
        self.0.forward(batch, policy).await
    }

    async fn enqueue_forward_and_return_completion(
        self: Arc<Self>,
        batch: &mut [ModelInput<'_, NaiveKVPool<T>>],
        policy: Arc<dyn SchedulingPolicy>,
    ) -> SubmittedGpuWorkWithCompletion {
        self.0
            .clone()
            .enqueue_forward_and_return_completion(batch, policy)
            .await
    }
}

struct Workload {
    shared_prefix: String,
    prompts: Vec<String>,
    suffix_token_ids: Vec<u32>,
}

impl Workload {
    const SHARED_TOKEN: u32 = 9707;

    /// Validate text round-trips because Scheduler accepts text, not arbitrary token IDs.
    fn new(tokenizer: &tokenizers::Tokenizer, args: &Args) -> Result<Self> {
        let prefix_ids = vec![Self::SHARED_TOKEN; args.prefix_tokens];
        let round_trip = |ids: &[u32]| -> Result<Option<String>> {
            let text = tokenizer
                .decode(ids, false)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let encoded = tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            Ok((encoded.get_ids() == ids).then_some(text))
        };
        let shared_prefix = round_trip(&prefix_ids)?
            .ok_or_else(|| anyhow::anyhow!("shared prefix does not round-trip"))?;
        let mut prompts = Vec::with_capacity(args.batch_size.get());
        let mut suffix_token_ids = Vec::with_capacity(args.batch_size.get());
        for token in 1000..tokenizer.get_vocab_size(false) as u32 {
            let word = tokenizer
                .decode(&[token], false)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            if !word.starts_with(' ') || !word.trim().chars().all(|c| c.is_ascii_alphabetic()) {
                continue;
            }
            let mut ids = prefix_ids.clone();
            ids.resize(args.tokens.get(), token);
            if let Some(text) = round_trip(&ids)? {
                prompts.push(text);
                suffix_token_ids.push(token);
                if prompts.len() == args.batch_size.get() {
                    break;
                }
            }
        }
        ensure!(
            prompts.len() == args.batch_size.get(),
            "not enough distinct round-trip suffix tokens"
        );
        Ok(Self {
            shared_prefix,
            prompts,
            suffix_token_ids,
        })
    }
}

/// Counts since the last drain, not model-lifetime totals or inferred hit percentages.
#[derive(Default, Serialize)]
struct Counters {
    computed_input_tokens: u64,
    forward_count: u64,
    cache_hits: u64,
    reused_tokens: u64,
    published_tokens: u64,
    evicted_tokens: u64,
}

impl Counters {
    fn drain(snapshotter: &Snapshotter) -> Self {
        let mut result = Self::default();
        // DebuggingRecorder snapshots reset counters; do not subtract two snapshots.
        for (key, _, _, value) in snapshotter.snapshot().into_vec() {
            let DebugValue::Counter(value) = value else {
                continue;
            };
            let field = match key.key().name() {
                "qwen2_input_tokens_total" => &mut result.computed_input_tokens,
                "qwen2_steps_total" => &mut result.forward_count,
                "engine_prefix_cache_hits_total" => &mut result.cache_hits,
                "engine_prefix_cache_reused_tokens_total" => &mut result.reused_tokens,
                "engine_prefix_cache_published_tokens_total" => &mut result.published_tokens,
                "engine_prefix_cache_evicted_tokens_total" => &mut result.evicted_tokens,
                _ => continue,
            };
            *field += value;
        }
        result
    }
}

#[derive(Serialize)]
struct Sample {
    cohort_wall_seconds: f64,
    time_to_first_text_seconds: Vec<Option<f64>>,
    request_elapsed_seconds: Vec<f64>,
    output_text: Vec<String>,
    counters: Counters,
    computed_prefill_tokens: u64,
    kv_retained_bytes_before_shutdown: usize,
}

async fn sample<T: WeightElement + ElementType>(
    args: &Args,
    state: CacheState,
    workload: &Workload,
    model: Arc<Qwen2Engine<T>>,
    pool: Arc<NaiveKVPool<T>>,
    policy: Arc<dyn SchedulingPolicy>,
    snapshotter: &Snapshotter,
) -> Result<Sample> {
    let initial = pool.metrics();
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
    let owned_pool = pool.clone();
    let runner = RunnerBuilder::default()
        .with_overlap(args.mode == Mode::Overlap)
        .with_max_batch_tokens(
            NonZeroUsize::new(
                args.tokens
                    .get()
                    .checked_mul(args.batch_size.get())
                    .ok_or_else(|| anyhow::anyhow!("token budget overflow"))?,
            )
            .unwrap(),
        )
        .with_prefix_cache(PrefixCacheConfig {
            max_prefix_tokens: args.tokens,
            max_cached_tokens: args.cache_tokens,
        })
        .with_model_factory(move || async move { Ok(FixedLengthModel(model)) })
        .with_pool_factory(move |_| async move { Ok(owned_pool) })
        .with_policy_factory(move || Ok(policy))
        .build(command_rx, replies)
        .await?;
    let task = tokio::spawn(runner.run(cancel.child_token()));
    let request = |text: String, count| {
        let mut request = SessionRequest::new(None, text);
        request.sampling = Some(SamplingConfig {
            max_new_tokens: Some(count),
            ..Default::default()
        });
        request
    };
    let result = async {
        if state == CacheState::Warm && args.prefix_tokens > 0 {
            scheduler
                .execute(request(workload.shared_prefix.clone(), 1))
                .await?;
        }
        stats.lock().unwrap().clear();
        Counters::drain(snapshotter);
        let requests: Vec<_> = workload
            .prompts
            .iter()
            .map(|prompt| request(prompt.clone(), args.output_tokens.get()))
            .collect();
        let range = nvtx_sys::range_start_ascii(if state == CacheState::Cold {
            c"prefix_cache.cold"
        } else {
            c"prefix_cache.warm"
        });
        let start = Instant::now();
        // Admission is synchronous: the full cohort is queued before Runner executes it.
        let output = async {
            let streams = requests
                .into_iter()
                .map(|request| scheduler.execute_stream(request))
                .collect::<Result<Vec<_>, _>>()?;
            try_join_all(streams.into_iter().map(|stream| async move {
                futures::pin_mut!(stream);
                let mut text = String::new();
                while let Some(delta) = stream.try_next().await? {
                    text.push_str(&delta.output);
                }
                Ok::<_, engine::error::EngineError>(text)
            }))
            .await
        }
        .await;
        let cohort_wall_seconds = start.elapsed().as_secs_f64();
        nvtx_sys::range_end(range);
        let output_text = output?;
        let counters = Counters::drain(snapshotter);
        let reused = if state == CacheState::Warm {
            args.prefix_tokens * args.batch_size.get()
        } else {
            0
        } as u64;
        ensure!(
            counters.reused_tokens == reused,
            "actual prefix reuse differs: {} != {reused}",
            counters.reused_tokens
        );
        ensure!(
            counters.cache_hits
                == if reused == 0 {
                    0
                } else {
                    args.batch_size.get() as u64
                },
            "wrong cache hit count"
        );
        let expected_input =
            (args.batch_size.get() * (args.tokens.get() + args.output_tokens.get() - 1)) as u64
                - reused;
        ensure!(
            counters.computed_input_tokens == expected_input,
            "computed token count differs: {} != {expected_input}",
            counters.computed_input_tokens
        );
        ensure!(
            counters.forward_count == args.output_tokens.get() as u64,
            "cohort split into unexpected GPU batches"
        );
        let records = stats.lock().unwrap();
        ensure!(
            records.len() == args.batch_size.get()
                && records
                    .iter()
                    .all(|r| r.generated_tokens == Some(args.output_tokens.get())),
            "missing or truncated request completion"
        );
        Ok(Sample {
            cohort_wall_seconds,
            time_to_first_text_seconds: records
                .iter()
                .map(|r| r.time_to_first_text.map(|d| d.as_secs_f64()))
                .collect(),
            request_elapsed_seconds: records.iter().map(|r| r.elapsed.as_secs_f64()).collect(),
            computed_prefill_tokens: counters.computed_input_tokens
                - (args.batch_size.get() * (args.output_tokens.get() - 1)) as u64,
            output_text,
            counters,
            kv_retained_bytes_before_shutdown: pool.metrics().layers.assigned_bytes,
        })
    }
    .await;
    // Drain submissions/cache owners on every error path before reusing this pool.
    cancel.cancel();
    scheduler.wait_stopped().await;
    task.await?;
    ensure!(pool.metrics() == initial, "sample leaked KV rows or slots");
    result
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!cfg!(debug_assertions), "use cargo bench (optimized build)");
    ensure!(
        args.prefix_tokens < args.tokens.get(),
        "leave at least one unique suffix token"
    );
    ensure!(
        args.prefix_tokens <= args.cache_tokens.get(),
        "cache must hold the shared prefix"
    );
    tracing_subscriber::fmt()
        .json()
        .with_writer(std::io::stderr)
        .with_max_level(if args.trace_steps {
            tracing::Level::DEBUG
        } else {
            tracing::Level::WARN
        })
        .init();
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::set_global_recorder(recorder)?;
    match args.dtype.as_str() {
        "bf16" => run::<bf16>(args, snapshotter).await,
        "f32" => run::<f32>(args, snapshotter).await,
        _ => unreachable!(),
    }
}

async fn run<T: WeightElement + ElementType>(args: Args, snapshotter: Snapshotter) -> Result<()> {
    let device = Device::new(0)?;
    let policy: Arc<dyn SchedulingPolicy> = Arc::new(SingleStream::new(&device)?);
    let model = Qwen2Engine::<T>::load(&args.model_dir, &policy).await?;
    let model = Arc::new(if args.cuda_graph {
        model.with_cuda_graphs()
    } else {
        model
    });
    let context_len = args
        .tokens
        .get()
        .checked_add(args.output_tokens.get() - 1)
        .ok_or_else(|| anyhow::anyhow!("context length overflow"))?;
    let mut config = model.kv_pool_config(context_len)?;
    config.max_rows = args.batch_size.get();
    config.num_slots = match args.num_slots {
        Some(slots) => slots.get(),
        None => context_len
            .checked_mul(config.max_rows)
            .and_then(|n| n.checked_add(args.cache_tokens.get()))
            .ok_or_else(|| anyhow::anyhow!("pool capacity overflow"))?,
    };
    let pool = NaiveKVPool::<T>::new(config, &policy).await?;
    let workload = Workload::new(model.tokenizer(), &args)?;
    println!(
        "{}",
        json!({"event":"configuration", "args":args, "shared_token_id":Workload::SHARED_TOKEN,
        "suffix_token_ids":workload.suffix_token_ids, "num_slots":config.num_slots, "max_rows":config.max_rows,
        "timing_boundary":"fixed_cohort_scheduler_to_stream_end_no_http", "eos_stopping":false})
    );
    let mut expected = None;
    let mut measured = Vec::new();
    for iteration in 0..args.warmup.get() + args.iterations.get() {
        let warmup = iteration < args.warmup.get();
        let states = if args.cache_state == CacheState::Both {
            if iteration % 2 == 0 {
                vec![CacheState::Cold, CacheState::Warm]
            } else {
                vec![CacheState::Warm, CacheState::Cold]
            }
        } else {
            vec![args.cache_state]
        };
        for state in states {
            let sample = sample(
                &args,
                state,
                &workload,
                model.clone(),
                pool.clone(),
                policy.clone(),
                &snapshotter,
            )
            .await?;
            if let Some(expected) = &expected {
                ensure!(
                    sample.output_text == *expected,
                    "cold/warm generated text differs"
                );
            } else {
                expected = Some(sample.output_text.clone());
            }
            println!(
                "{}",
                json!({"event":"sample", "state":state, "iteration":iteration, "warmup":warmup, "sample":sample, "kv_released":true})
            );
            if !warmup {
                measured.push((state, sample));
            }
        }
    }
    for state in [CacheState::Cold, CacheState::Warm] {
        let samples: Vec<_> = measured
            .iter()
            .filter(|(s, _)| *s == state)
            .map(|(_, sample)| sample)
            .collect();
        if samples.is_empty() {
            continue;
        }
        let seconds: f64 = samples.iter().map(|s| s.cohort_wall_seconds).sum();
        let first: Vec<_> = samples
            .iter()
            .flat_map(|s| s.time_to_first_text_seconds.iter().flatten())
            .collect();
        println!(
            "{}",
            json!({"event":"summary", "state":state, "samples":samples.len(),
            "cohort_mean_seconds":seconds / samples.len() as f64,
            "requests_per_second":(samples.len() * args.batch_size.get()) as f64 / seconds,
            "output_tokens_per_second":(samples.len() * args.batch_size.get() * args.output_tokens.get()) as f64 / seconds,
            "time_to_first_text_mean_seconds":if first.is_empty() { None } else { Some(first.iter().map(|v| **v).sum::<f64>() / first.len() as f64) },
            "reused_tokens":samples.iter().map(|s| s.counters.reused_tokens).sum::<u64>(),
            "computed_prefill_tokens":samples.iter().map(|s| s.computed_prefill_tokens).sum::<u64>(),
            "workspace_pool":model.workspace_metrics(), "cuda_graph_pool":model.cuda_graph_metrics()})
        );
    }
    Ok(())
}
