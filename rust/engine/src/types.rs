use crate::error::EngineError;
use tokio::sync::oneshot;

pub type RequestId = u64;

#[derive(Debug)]
pub struct LLMResponse {
    pub request_id: RequestId,
    pub output: String,
}

pub(crate) struct Submit {
    pub request_id: RequestId,
    pub request: String,
    pub reply_tx: oneshot::Sender<Result<LLMResponse, EngineError>>,
}

pub(crate) struct PendingRequest {
    pub reply_tx: oneshot::Sender<Result<LLMResponse, EngineError>>,
}

pub struct SingleRequest<T = String> {
    pub request_id: RequestId,
    pub request: T,
}

pub type BatchIn<T = String> = Vec<SingleRequest<T>>;
pub type BatchOut = Vec<LLMResponse>;

pub struct BatchReply {
    pub request_ids: Vec<RequestId>,
    pub result: Result<BatchOut, EngineError>,
}
