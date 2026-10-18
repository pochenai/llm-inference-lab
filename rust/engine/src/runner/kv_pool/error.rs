use std::fmt;

/// Errors from KV storage, allocation, and backend state operations.
#[derive(Clone, Debug)]
pub enum KVPoolError {
    InvalidInput(String),
    CapacityExhausted(String),
    Device(String),
}

impl fmt::Display for KVPoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid KV input: {message}"),
            Self::CapacityExhausted(message) => write!(f, "KV capacity exhausted: {message}"),
            Self::Device(message) => write!(f, "KV device operation failed: {message}"),
        }
    }
}

impl std::error::Error for KVPoolError {}

impl From<cuda_async::error::DeviceError> for KVPoolError {
    fn from(error: cuda_async::error::DeviceError) -> Self {
        Self::Device(error.to_string())
    }
}

impl From<cutile::error::Error> for KVPoolError {
    fn from(error: cutile::error::Error) -> Self {
        Self::Device(error.to_string())
    }
}

/// Errors from logical KV progress or the underlying pool.
#[derive(Clone, Debug)]
pub enum KVContextError {
    InvalidState(String),
    CapacityExceeded,
    Pool(KVPoolError),
}

impl fmt::Display for KVContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidState(message) => write!(f, "invalid KV context state: {message}"),
            Self::CapacityExceeded => write!(f, "KV context capacity exceeded"),
            Self::Pool(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for KVContextError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Pool(error) => Some(error),
            _ => None,
        }
    }
}

impl From<KVPoolError> for KVContextError {
    fn from(error: KVPoolError) -> Self {
        Self::Pool(error)
    }
}
