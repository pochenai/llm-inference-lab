//! Qwen3.5 BF16 text generation through Scheduler -> ModelRunner -> cuTile.

use anyhow::{ensure, Result};
use clap::Parser;
use engine::{
    runner::{
        kv_pool::KVPool,
        model::{Qwen3_5Engine, Qwen3_5StatePool},
        model_runner::RunnerBuilder,
    },
    scheduler::Scheduler,
    types::{SamplingConfig, SessionRequest},
};
use std::{num::NonZeroUsize, path::PathBuf, time::Instant};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(
        long,
        default_value = "Explain what a language model is in one sentence."
    )]
    prompt: String,
    #[arg(long, default_value = "16")]
    max_new_tokens: NonZeroUsize,
    /// Maximum uncached input tokens per Runner forward.
    #[arg(long, alias = "chunk-size", default_value = "32")]
    max_batch_tokens: NonZeroUsize,
    /// Repeat independent requests to check state release and output equality.
    #[arg(long, default_value = "1")]
    repeat: NonZeroUsize,
    /// Write standalone first-step FP32 logits before running the Runner example.
    #[arg(long)]
    logits: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let policy = cuda_async::device_context::global_policy(0)?;
    let started = Instant::now();
    let model = Qwen3_5Engine::<cutile::core::bf16>::load(&args.model_dir, &policy).await?;
    eprintln!(
        "Loaded {} weight bytes in {:.2}s",
        model.weight_bytes(),
        started.elapsed().as_secs_f64()
    );
    let text = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        args.prompt
    );
    let encoded = model
        .tokenizer()
        .encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let max_seq_len = encoded
        .get_ids()
        .len()
        .checked_add(args.max_new_tokens.get())
        .ok_or_else(|| anyhow::anyhow!("sequence length overflow"))?;
    let pool = Qwen3_5StatePool::new(&model, model.kv_pool_config(max_seq_len)?, &policy).await?;
    eprintln!(
        "State budget: {} bytes/session",
        model
            .config()
            .text_config
            .state_bytes::<cutile::core::bf16>(max_seq_len)
            .unwrap()
    );
    if let Some(path) = args.logits {
        let mut context = model.new_context(max_seq_len).await?;
        let (_, logits) = model
            .step_with_logits(encoded.get_ids(), &mut context)
            .await?;
        std::fs::write(
            path,
            logits
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )?;
    }
    let initial = pool.metrics();
    let cancel = CancellationToken::new();
    let (commands, command_rx) = mpsc::unbounded_channel();
    let (replies, reply_rx) = mpsc::unbounded_channel();
    let scheduler = Scheduler::new(cancel.child_token(), commands, reply_rx);
    let owned_pool = pool.clone();
    let runner = RunnerBuilder::default()
        .with_max_batch_tokens(args.max_batch_tokens)
        .with_overlap(false)
        .with_model_factory(move || async move { Ok(model) })
        .with_pool_factory(move |_| async move { Ok(owned_pool) })
        .with_policy_factory(move || Ok(policy))
        .build(command_rx, replies)
        .await?;
    let task = tokio::spawn(runner.run(cancel.child_token()));
    let mut request = SessionRequest::new(None, text);
    request.sampling = Some(SamplingConfig {
        max_new_tokens: Some(args.max_new_tokens.get()),
        ..Default::default()
    });
    let started = Instant::now();
    let result: Result<_> = async {
        let response = scheduler.execute(request.clone()).await?;
        for _ in 1..args.repeat.get() {
            let next = scheduler.execute(request.clone()).await?;
            ensure!(
                next.output == response.output,
                "repeated request output differs"
            );
        }
        Ok(response)
    }
    .await;
    cancel.cancel();
    scheduler.wait_stopped().await;
    task.await?;
    let response = result?;
    ensure!(
        pool.metrics() == initial && pool.can_append(0, 1),
        "session state was not released"
    );
    println!("{}", response.output);
    eprintln!(
        "Runner: {:.3}s, {} requests; session state released",
        started.elapsed().as_secs_f64(),
        args.repeat
    );
    Ok(())
}
