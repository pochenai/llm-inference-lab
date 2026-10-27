mod api;
mod error;
mod output;
mod qwen2;
mod simple_mlp;
mod submission;

pub use api::{AttentionBackend, CausalModel, ModelInput};
pub use error::ModelError;
pub use output::{GpuTokenPredictionHandle, ModelTokens, NextToken};
pub use qwen2::{Qwen2Config, Qwen2Engine, WorkspaceMetrics};
pub use simple_mlp::{SimpleMLPConfig, SimpleMLPEngine, INPUT_DIM};
pub use submission::SubmittedGpuWorkWithCompletion;

pub(in crate::runner) use submission::GpuWorkCompletionFuture;
