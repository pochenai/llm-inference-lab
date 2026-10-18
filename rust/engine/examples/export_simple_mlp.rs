use engine::runner::model::SimpleMLPEngine;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/simple_mlp"));
    let model = SimpleMLPEngine::random(0).await?;
    model.save_pretrained(&model_dir).await?;
    println!("Saved GPU-initialized MLP to {}", model_dir.display());
    Ok(())
}
