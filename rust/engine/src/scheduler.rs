use crate::{error::EngineError, types::*};
use dashmap::DashMap;
use std::{
    collections::HashSet,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::{
    sync::{mpsc, oneshot},
    task::AbortHandle,
    time::{sleep_until, Instant},
};

use tokio_util::{sync::CancellationToken, task::TaskTracker};

// Candidate dispatch bound; Runner chooses execution batches using local capacity.
const MAX_DISPATCH_SIZE: usize = 5;
const BATCH_WAIT: Duration = Duration::from_millis(100);

struct PendingRequest {
    reply_tx: oneshot::Sender<Result<LLMResponse, EngineError>>,
}

type Requests = Arc<DashMap<RequestId, PendingRequest>>;

#[derive(Clone)]
pub struct Scheduler {
    command_tx: mpsc::UnboundedSender<RunnerCommand>,
    requests: Requests,
    next_id: Arc<AtomicU64>,
    next_session_id: Arc<AtomicU64>,
    cancel_token: CancellationToken,
    tasks: Vec<AbortHandle>,
    task_tracker: TaskTracker,
}

impl Scheduler {
    // Starts background tasks; call from within a Tokio runtime.
    pub fn new(
        cancel_token: CancellationToken,
        runner_command_tx: mpsc::UnboundedSender<RunnerCommand>,
        batch_reply_rx: mpsc::UnboundedReceiver<BatchReply>,
    ) -> Self {
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let requests = Arc::new(DashMap::new());
        let task_tracker = TaskTracker::new();
        let tasks = vec![
            task_tracker
                .spawn(Self::run_dispatch_batch(
                    command_rx,
                    runner_command_tx,
                    requests.clone(),
                ))
                .abort_handle(),
            task_tracker
                .spawn(Self::run_recv_resp(batch_reply_rx, requests.clone()))
                .abort_handle(),
        ];
        task_tracker.close();
        Self {
            command_tx,
            requests,
            next_id: Arc::new(AtomicU64::new(0)),
            next_session_id: Arc::new(AtomicU64::new(0)),
            cancel_token,
            tasks,
            task_tracker,
        }
    }

    // Must be polled to stop tasks when the caller signals cancellation.
    pub async fn wait_stopped(&self) {
        self.cancel_token.cancelled().await;
        for task in &self.tasks {
            task.abort();
        }
        self.task_tracker.wait().await;
        // Clear only after all tasks stop inserting pending requests.
        self.requests.clear();
    }

    /// Allocate a fresh invocation ID at admission, shared across Scheduler clones.
    pub async fn execute(&self, request: SessionRequest) -> Result<LLMResponse, EngineError> {
        if self.cancel_token.is_cancelled() {
            return Err(EngineError::ChannelClosed);
        }
        let request_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request = request.with_request_id(request_id);
        let (reply_tx, response_rx) = oneshot::channel();
        self.requests
            .insert(request_id, PendingRequest { reply_tx });
        if self
            .command_tx
            .send(RunnerCommand::Execute(vec![request]))
            .is_err()
        {
            self.requests.remove(&request_id);
            return Err(EngineError::ChannelClosed);
        }
        response_rx.await.map_err(|_| EngineError::ChannelClosed)?
    }

    /// Allocate an ID shared across clones and wait for Runner to create the session.
    pub async fn open_session(&self) -> Result<SessionId, EngineError> {
        if self.cancel_token.is_cancelled() {
            return Err(EngineError::ChannelClosed);
        }
        let id = self.next_session_id.fetch_add(1, Ordering::Relaxed);
        let (reply, response) = oneshot::channel();
        self.command_tx
            .send(RunnerCommand::OpenSession { id, reply })
            .map_err(|_| EngineError::ChannelClosed)?;
        tokio::select! {
            _ = self.cancel_token.cancelled() => Err(EngineError::ChannelClosed),
            result = response => {
                result.map_err(|_| EngineError::ChannelClosed)??;
                Ok(id)
            }
        }
    }

    /// Queue close after earlier requests and wait for Runner to release the session.
    pub async fn close_session(&self, id: SessionId) -> Result<(), EngineError> {
        if self.cancel_token.is_cancelled() {
            return Err(EngineError::ChannelClosed);
        }
        let (reply, response) = oneshot::channel();
        self.command_tx
            .send(RunnerCommand::CloseSession { id, reply })
            .map_err(|_| EngineError::ChannelClosed)?;
        tokio::select! {
            _ = self.cancel_token.cancelled() => Err(EngineError::ChannelClosed),
            result = response => result.map_err(|_| EngineError::ChannelClosed)?,
        }
    }

    async fn run_dispatch_batch(
        mut command_rx: mpsc::UnboundedReceiver<RunnerCommand>,
        runner_command_tx: mpsc::UnboundedSender<RunnerCommand>,
        requests: Requests,
    ) {
        let mut submissions = Vec::with_capacity(MAX_DISPATCH_SIZE);
        let mut sessions = HashSet::new();
        let mut deadline = Instant::now();
        let flush = |batch: &mut BatchIn, sessions: &mut HashSet<SessionId>| {
            if !batch.is_empty() {
                runner_command_tx
                    .send(RunnerCommand::Execute(std::mem::take(batch)))
                    .map_err(|_| EngineError::ChannelClosed)?;
                sessions.clear();
            }
            Ok::<(), EngineError>(())
        };
        let result = async {
            loop {
                let command = tokio::select! {
                    biased;
                    _ = sleep_until(deadline), if !submissions.is_empty() => {
                        flush(&mut submissions, &mut sessions)?;
                        continue;
                    }
                    command = command_rx.recv() => command,
                };
                match command {
                    Some(RunnerCommand::Execute(batch)) => {
                        for request in batch {
                            if request.session_id.is_some_and(|id| sessions.contains(&id)) {
                                flush(&mut submissions, &mut sessions)?;
                            }
                            if submissions.is_empty() {
                                deadline = Instant::now() + BATCH_WAIT;
                            }
                            if let Some(id) = request.session_id {
                                sessions.insert(id);
                            }
                            submissions.push(request);
                            if submissions.len() == MAX_DISPATCH_SIZE {
                                flush(&mut submissions, &mut sessions)?;
                            }
                        }
                    }
                    Some(command) => {
                        // Session management is a FIFO barrier: flush earlier requests first.
                        flush(&mut submissions, &mut sessions)?;
                        runner_command_tx
                            .send(command)
                            .map_err(|_| EngineError::ChannelClosed)?;
                    }
                    None => {
                        flush(&mut submissions, &mut sessions)?;
                        break;
                    }
                }
            }
            Ok::<(), EngineError>(())
        }
        .await;
        if result.is_err() {
            requests.clear();
        }
    }

    async fn run_recv_resp(
        mut batch_reply_rx: mpsc::UnboundedReceiver<BatchReply>,
        requests: Requests,
    ) {
        while let Some(output) = batch_reply_rx.recv().await {
            for (request_id, result) in output.results {
                if let Some((_, request)) = requests.remove(&request_id) {
                    let _ = request.reply_tx.send(result);
                }
            }
        }
        requests.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn per_request_errors_do_not_affect_other_callers() {
        let cancel = CancellationToken::new();
        let (request_tx, mut request_rx) = mpsc::unbounded_channel::<RunnerCommand>();
        let (reply_tx, reply_rx) = mpsc::unbounded_channel();
        let scheduler = Scheduler::new(cancel.child_token(), request_tx, reply_rx);
        let worker = tokio::spawn(async move {
            while let Some(RunnerCommand::Execute(batch)) = request_rx.recv().await {
                let results = batch
                    .into_iter()
                    .map(|request| {
                        let id = request.assigned_request_id();
                        let result = if request.input == "bad" {
                            Err(EngineError::InvalidInput("bad request".into()))
                        } else {
                            Ok(LLMResponse {
                                request_id: id,
                                output: request.input,
                            })
                        };
                        (id, result)
                    })
                    .collect();
                reply_tx.send(BatchReply { results }).unwrap();
            }
        });
        let (bad, good) = tokio::join!(
            scheduler.execute(SessionRequest::new(None, "bad".into())),
            scheduler.execute(SessionRequest::new(None, "good".into()))
        );
        assert!(matches!(bad, Err(EngineError::InvalidInput(_))));
        assert_eq!(good.unwrap().output, "good");
        cancel.cancel();
        scheduler.wait_stopped().await;
        worker.await.unwrap();
    }

    #[tokio::test]
    async fn batches_defer_repeated_sessions_without_dropping_or_reordering() {
        let (tx, rx) = mpsc::unbounded_channel();
        let (batch_tx, mut batch_rx) = mpsc::unbounded_channel();
        let inputs = [
            (Some(1), "a1"),
            (Some(2), "b1"),
            (None, "x"),
            (None, "y"),
            (Some(1), "a2"),
            (Some(2), "b2"),
            (None, "z"),
            (Some(1), "a3"),
            (None, "tail1"),
            (None, "tail2"),
            (None, "tail3"),
            (None, "tail4"),
            (None, "tail5"),
        ];
        for (id, &(session, input)) in inputs.iter().enumerate() {
            tx.send(RunnerCommand::Execute(vec![SessionRequest::new(
                session,
                input.to_owned(),
            )
            .with_request_id(id as u64)]))
                .unwrap();
        }
        drop(tx);
        Scheduler::run_dispatch_batch(rx, batch_tx, Arc::new(DashMap::new())).await;
        let mut sizes = Vec::new();
        let mut observed = Vec::new();
        while let Some(RunnerCommand::Execute(batch)) = batch_rx.recv().await {
            assert!(batch.len() <= MAX_DISPATCH_SIZE);
            sizes.push(batch.len());
            let mut sessions = HashSet::new();
            for request in batch {
                if let Some(id) = request.session_id {
                    assert!(sessions.insert(id));
                }
                observed.push((
                    request.assigned_request_id(),
                    request.session_id,
                    request.input,
                ));
            }
        }
        assert_eq!(sizes, vec![4, 3, 5, 1]);
        let expected: Vec<_> = inputs
            .iter()
            .enumerate()
            .map(|(id, &(session, input))| (id as u64, session, input.to_owned()))
            .collect();
        assert_eq!(observed, expected);
    }

    #[tokio::test]
    async fn admission_assigns_fresh_ids_across_clones_and_resubmissions() -> Result<(), EngineError>
    {
        let cancel = CancellationToken::new();
        let (request_tx, mut request_rx) = mpsc::unbounded_channel::<RunnerCommand>();
        let (reply_tx, reply_rx) = mpsc::unbounded_channel();
        let scheduler = Scheduler::new(cancel.child_token(), request_tx, reply_rx);
        let worker = tokio::spawn(async move {
            while let Some(RunnerCommand::Execute(batch)) = request_rx.recv().await {
                let mut outputs = Vec::new();
                for request in batch {
                    let id = request.assigned_request_id();
                    assert_eq!(request.session_id, Some(7));
                    outputs.push((
                        id,
                        Ok(LLMResponse {
                            request_id: id,
                            output: request.input,
                        }),
                    ));
                }
                if reply_tx.send(BatchReply { results: outputs }).is_err() {
                    break;
                }
            }
        });
        let request = SessionRequest::new(Some(7), "same payload".to_owned());
        assert_eq!(request.request_id(), None);
        let clone = scheduler.clone();
        let (left, right) = tokio::join!(
            scheduler.execute(request.clone()),
            clone.execute(request.clone())
        );
        let (left, right) = (left?, right?);
        assert_ne!(left.request_id, right.request_id);
        assert_eq!(left.output, "same payload");
        assert_eq!(right.output, "same payload");
        // Even re-admitting an internal envelope cannot reuse its previous invocation ID.
        let next = clone
            .execute(request.with_request_id(left.request_id))
            .await?;
        assert_ne!(next.request_id, left.request_id);
        assert_ne!(next.request_id, right.request_id);
        assert!(scheduler.requests.is_empty());
        cancel.cancel();
        scheduler.wait_stopped().await;
        worker.await.unwrap();
        Ok(())
    }

    #[tokio::test]
    async fn session_commands_flush_earlier_repeated_requests_in_fifo_order() {
        let (tx, rx) = mpsc::unbounded_channel();
        let (runner_tx, mut runner_rx) = mpsc::unbounded_channel();
        for (id, text) in [(1, "first"), (2, "second")] {
            tx.send(RunnerCommand::Execute(vec![SessionRequest::new(
                Some(7),
                text.into(),
            )
            .with_request_id(id)]))
                .unwrap();
        }
        let (reply, close_response) = oneshot::channel();
        tx.send(RunnerCommand::CloseSession { id: 7, reply })
            .unwrap();
        let (reply, open_response) = oneshot::channel();
        tx.send(RunnerCommand::OpenSession { id: 8, reply })
            .unwrap();
        tx.send(RunnerCommand::Execute(vec![SessionRequest::new(
            Some(8),
            "third".into(),
        )
        .with_request_id(3)]))
            .unwrap();
        drop(tx);
        Scheduler::run_dispatch_batch(rx, runner_tx, Arc::new(DashMap::new())).await;
        let mut observed = Vec::new();
        while let Some(command) = runner_rx.recv().await {
            match command {
                RunnerCommand::Execute(batch) => {
                    assert_eq!(batch.len(), 1);
                    observed.push(batch[0].input.clone());
                }
                RunnerCommand::CloseSession { id, reply } => {
                    assert_eq!(id, 7);
                    observed.push("close".into());
                    reply.send(Ok(())).unwrap();
                }
                RunnerCommand::OpenSession { id, reply } => {
                    assert_eq!(id, 8);
                    observed.push("open".into());
                    reply.send(Ok(())).unwrap();
                }
            }
        }
        assert_eq!(observed, ["first", "second", "close", "open", "third"]);
        close_response.await.unwrap().unwrap();
        open_response.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn session_management_unblocks_on_cancellation_or_runner_disconnect() {
        for opening in [true, false] {
            let cancel = CancellationToken::new();
            let (runner_tx, mut runner_rx) = mpsc::unbounded_channel();
            let (_reply_tx, reply_rx) = mpsc::unbounded_channel();
            let scheduler = Scheduler::new(cancel.child_token(), runner_tx, reply_rx);
            let operation = async {
                if opening {
                    scheduler.open_session().await.map(|_| ())
                } else {
                    scheduler.close_session(0).await
                }
            };
            tokio::pin!(operation);
            let command = tokio::select! {
                result = &mut operation => panic!("management returned before Runner acknowledgement: {result:?}"),
                command = runner_rx.recv() => command.unwrap(),
            };
            assert!(matches!(
                (&command, opening),
                (RunnerCommand::OpenSession { .. }, true)
                    | (RunnerCommand::CloseSession { .. }, false)
            ));
            // Keep the acknowledgement alive: only cancellation can wake this caller.
            cancel.cancel();
            let result = tokio::time::timeout(Duration::from_secs(1), &mut operation)
                .await
                .expect("cancelled session management must not hang");
            assert!(matches!(result, Err(EngineError::ChannelClosed)));
            drop(command);
            scheduler.wait_stopped().await;
        }
        let cancel = CancellationToken::new();
        let (runner_tx, runner_rx) = mpsc::unbounded_channel();
        let (_reply_tx, reply_rx) = mpsc::unbounded_channel();
        drop(runner_rx);
        let scheduler = Scheduler::new(cancel.child_token(), runner_tx, reply_rx);
        assert!(matches!(
            scheduler.open_session().await,
            Err(EngineError::ChannelClosed)
        ));
        assert!(matches!(
            scheduler.close_session(0).await,
            Err(EngineError::ChannelClosed)
        ));
        cancel.cancel();
        scheduler.wait_stopped().await;
    }

    // Verify cancellation releases pending callers and rejects later submissions.
    #[tokio::test]
    async fn cancellation_wakes_pending_callers_and_closes_ingress() {
        let cancel_token = CancellationToken::new();
        let (batch_req_tx, _batch_req_rx) = mpsc::unbounded_channel();
        let (_batch_reply_tx, batch_reply_rx) = mpsc::unbounded_channel();
        let scheduler = Scheduler::new(cancel_token.child_token(), batch_req_tx, batch_reply_rx);
        // No runner is attached, so this request can finish only through cancellation.
        let (result, ()) = tokio::join!(
            scheduler.execute(SessionRequest::new(None, "hello".into())),
            async {
                cancel_token.cancel();
                scheduler.wait_stopped().await;
            },
        );
        assert!(matches!(result, Err(EngineError::ChannelClosed)));
        // wait_stopped must finish both tasks before clearing all request state.
        assert!(scheduler.requests.is_empty());
        assert!(scheduler.task_tracker.is_empty());
        assert!(scheduler.tasks.iter().all(AbortHandle::is_finished));
        assert!(matches!(
            scheduler
                .execute(SessionRequest::new(None, "after cancel".into()))
                .await,
            Err(EngineError::ChannelClosed)
        ));
        assert!(matches!(
            scheduler.open_session().await,
            Err(EngineError::ChannelClosed)
        ));
        assert!(matches!(
            scheduler.close_session(0).await,
            Err(EngineError::ChannelClosed)
        ));
    }
}
