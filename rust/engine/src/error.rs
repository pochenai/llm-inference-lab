use std::fmt;

use crate::runner::{
    kv_pool::{KVContextError, KVPoolError},
    model::ModelError,
};

#[derive(Clone, Debug)]
pub enum EngineError {
    ChannelClosed,
    InvalidInput(String),
    Internal(String),
    Model(ModelError),
    KVContext(KVContextError),
    KVPool(KVPoolError),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChannelClosed => write!(f, "engine channel closed"),
            Self::InvalidInput(message) => write!(f, "invalid input: {message}"),
            Self::Internal(message) => write!(f, "engine internal error: {message}"),
            Self::Model(error) => error.fmt(f),
            Self::KVContext(error) => error.fmt(f),
            Self::KVPool(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for EngineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Model(error) => Some(error),
            Self::KVContext(error) => Some(error),
            Self::KVPool(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ModelError> for EngineError {
    fn from(error: ModelError) -> Self {
        Self::Model(error)
    }
}

impl From<KVContextError> for EngineError {
    fn from(error: KVContextError) -> Self {
        Self::KVContext(error)
    }
}

impl From<KVPoolError> for EngineError {
    fn from(error: KVPoolError) -> Self {
        Self::KVPool(error)
    }
}

impl From<anyhow::Error> for EngineError {
    fn from(error: anyhow::Error) -> Self {
        Self::Internal(format!("{error:#}"))
    }
}

impl From<cuda_async::error::DeviceError> for EngineError {
    fn from(error: cuda_async::error::DeviceError) -> Self {
        Self::Internal(error.to_string())
    }
}

impl From<cutile::error::Error> for EngineError {
    fn from(error: cutile::error::Error) -> Self {
        Self::Internal(error.to_string())
    }
}
