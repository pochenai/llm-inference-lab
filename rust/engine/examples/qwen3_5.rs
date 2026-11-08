use anyhow::{ensure, Result};
use clap::Parser;
use engine::runner::model::Qwen3_5Engine;
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(
        long,
        default_value = "Explain what a language model is in one sentence."
    )]
    prompt: String,
    #[arg(long, default_value_t = 16)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 32)]
    chunk_size: usize,
    /// Write first-step FP32 logits for an independent numerical comparison.
    #[arg(long)]
    logits: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.max_new_tokens > 0 && args.chunk_size > 0,
        "token budget and chunk size must be positive"
    );
    let policy = cuda_async::device_context::global_policy(0)?;
    let started = Instant::now();
    let engine = Qwen3_5Engine::load(&args.model_dir, &policy).await?;
    eprintln!(
        "Loaded {} weight bytes in {:.2}s",
        engine.weight_bytes(),
        started.elapsed().as_secs_f64()
    );
    let text = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        args.prompt
    );
    let encoded = engine
        .tokenizer()
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let prompt_len = encoded.get_ids().len();
    let max_seq_len = prompt_len.saturating_add(args.max_new_tokens);
    let mut state = engine.new_context(max_seq_len).await?;
    let mut ids = Vec::with_capacity(max_seq_len);
    ids.extend_from_slice(encoded.get_ids());
    eprintln!(
        "State budget: {} bytes",
        engine
            .config()
            .text_config
            .state_bytes(max_seq_len)
            .unwrap()
    );
    let mut result = None;
    let started = Instant::now();
    for end in (args.chunk_size..prompt_len)
        .step_by(args.chunk_size)
        .chain(std::iter::once(prompt_len))
    {
        result = Some(engine.step_with_logits(&ids[..end], &mut state).await?);
    }
    let (mut next, logits) = result.unwrap();
    eprintln!(
        "Prefill: {:.3}s, {} tokens",
        started.elapsed().as_secs_f64(),
        prompt_len
    );
    if let Some(path) = args.logits {
        std::fs::write(
            path,
            logits
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )?;
    }
    let started = Instant::now();
    for step in 0..args.max_new_tokens {
        if engine.is_eos(next) {
            break;
        }
        ids.push(next);
        if step + 1 < args.max_new_tokens {
            next = engine.step_with_logits(&ids, &mut state).await?.0;
        }
    }
    println!(
        "{}",
        engine
            .tokenizer()
            .decode(&ids[prompt_len..], true)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
    );
    eprintln!(
        "Decode: {:.3}s, {} output tokens",
        started.elapsed().as_secs_f64(),
        ids.len() - prompt_len
    );
    Ok(())
}
