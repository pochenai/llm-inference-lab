use crate::error::EngineError;
use tokio::sync::oneshot;

pub type RequestId = u64;

pub struct Promp {
    pub prompt: String,
}

#[derive(Debug)]
pub struct LLMResponse {
    pub request_id: RequestId,
    pub value: u32,
}

pub(crate) struct Submit {
    pub request_id: RequestId,
    pub request: Promp,
    pub reply_tx: oneshot::Sender<Result<LLMResponse, EngineError>>,
}

pub(crate) struct PendingRequest {
    pub reply_tx: oneshot::Sender<Result<LLMResponse, EngineError>>,
}

pub struct SingleRequest {
    pub request_id: RequestId,
    pub request: Promp,
}

pub type BatchIn = Vec<SingleRequest>;
pub type BatchOut = Vec<LLMResponse>;
