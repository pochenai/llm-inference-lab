use crate::error::EngineError;
use tokio::sync::oneshot;

pub type RequestId = u64;

#[derive(Debug)]
pub struct LLMResponse {
    pub request_id: RequestId,
    /// A text delta from execute_stream, or the complete text from execute.
    pub output: String,
}

pub type SessionId = u64;

pub use crate::runner::loader::GenerationConfig as SamplingConfig;

#[derive(Clone, Debug)]
pub struct SessionRequest<T = String> {
    request_id: Option<RequestId>,
    pub session_id: Option<SessionId>,
    pub input: T,
    pub sampling: Option<SamplingConfig>,
}

impl<T> SessionRequest<T> {
    pub fn new(session_id: Option<SessionId>, input: T) -> Self {
        Self {
            request_id: None,
            session_id,
            input,
            sampling: None,
        }
    }

    /// None until Scheduler admission; callers may inspect but cannot choose the ID.
    pub fn request_id(&self) -> Option<RequestId> {
        self.request_id
    }

    /// Used for Scheduler admission and preserving identity through internal transforms.
    pub(crate) fn with_request_id(mut self, id: RequestId) -> Self {
        self.request_id = Some(id);
        self
    }

    /// Internal invariant: Scheduler assigns the ID before dispatching this envelope.
    pub(crate) fn assigned_request_id(&self) -> RequestId {
        self.request_id
            .expect("Scheduler assigns request IDs before dispatch")
    }
}

pub type BatchIn<T = String> = Vec<SessionRequest<T>>;

/// Ordered transport between Scheduler and Runner; applications use Scheduler APIs.
#[derive(Debug)]
pub enum RunnerCommand {
    Execute(BatchIn),
    OpenSession {
        id: SessionId,
        reply: oneshot::Sender<Result<(), EngineError>>,
    },
    CloseSession {
        id: SessionId,
        reply: oneshot::Sender<Result<(), EngineError>>,
    },
}

/// Incremental text and terminal results use the same ordered Runner transport.
pub enum RunnerReply {
    Delta(LLMResponse),
    Finished {
        request_id: RequestId,
        /// Actual accepted model tokens, including EOS and buffered/non-visible text.
        generated_tokens: usize,
        result: Result<(), EngineError>,
    },
}
