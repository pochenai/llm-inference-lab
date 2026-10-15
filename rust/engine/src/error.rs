use std::fmt;

#[derive(Debug)]
pub enum EngineError {
    ChannelClosed,
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChannelClosed => write!(f, "engine channel closed"),
        }
    }
}

impl std::error::Error for EngineError {}
