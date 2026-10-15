use std::fmt;

#[derive(Clone, Debug)]
pub enum EngineError {
    ChannelClosed,
    InvalidInput(String),
    Model(String),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChannelClosed => write!(f, "engine channel closed"),
            Self::InvalidInput(message) => write!(f, "invalid input: {message}"),
            Self::Model(message) => write!(f, "model execution failed: {message}"),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<cuda_async::error::DeviceError> for EngineError {
    fn from(error: cuda_async::error::DeviceError) -> Self {
        Self::Model(error.to_string())
    }
}

impl From<cutile::error::Error> for EngineError {
    fn from(error: cutile::error::Error) -> Self {
        Self::Model(error.to_string())
    }
}
