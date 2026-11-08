mod api;
mod error;
mod output;
mod qwen2;
mod qwen3_5;
mod readback;
mod simple_mlp;
mod submission;

pub use api::{AttentionBackend, CausalModel, ModelInput};
pub use error::ModelError;
pub use output::{GpuTokenPredictionHandle, ModelTokens, NextToken};
pub use qwen2::{CudaGraphMetrics, Qwen2Config, Qwen2Engine, WorkspaceMetrics};
pub use qwen3_5::{
    Qwen3_5Config, Qwen3_5Context, Qwen3_5Dtype, Qwen3_5Engine, Qwen3_5ExecutionView,
    Qwen3_5HiddenAct, Qwen3_5LayerType, Qwen3_5ModelType, Qwen3_5PrefixState, Qwen3_5RopeType,
    Qwen3_5SessionState, Qwen3_5StatePool, Qwen3_5TextConfig, Qwen3_5WorkspaceMetrics,
};
pub use simple_mlp::{SimpleMLPConfig, SimpleMLPEngine, INPUT_DIM};
pub use submission::SubmittedGpuWorkWithCompletion;

pub(in crate::runner) use submission::GpuWorkCompletionFuture;
