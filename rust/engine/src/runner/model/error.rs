use std::fmt;

use crate::runner::kv_pool::{KVContextError, KVPoolError};

/// Errors from model validation, execution, or KV access.
#[derive(Clone, Debug)]
pub enum ModelError {
    InvalidInput(String),
    Execution(String),
    KVContext(KVContextError),
    KVPool(KVPoolError),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid model input: {message}"),
            Self::Execution(message) => write!(f, "model execution failed: {message}"),
            Self::KVContext(error) => error.fmt(f),
            Self::KVPool(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::KVContext(error) => Some(error),
            Self::KVPool(error) => Some(error),
            _ => None,
        }
    }
}

impl From<KVContextError> for ModelError {
    fn from(error: KVContextError) -> Self {
        Self::KVContext(error)
    }
}

impl From<KVPoolError> for ModelError {
    fn from(error: KVPoolError) -> Self {
        Self::KVPool(error)
    }
}

impl From<anyhow::Error> for ModelError {
    fn from(error: anyhow::Error) -> Self {
        Self::Execution(format!("{error:#}"))
    }
}

impl From<cuda_async::error::DeviceError> for ModelError {
    fn from(error: cuda_async::error::DeviceError) -> Self {
        Self::Execution(error.to_string())
    }
}

impl From<cutile::error::Error> for ModelError {
    fn from(error: cutile::error::Error) -> Self {
        Self::Execution(error.to_string())
    }
}
