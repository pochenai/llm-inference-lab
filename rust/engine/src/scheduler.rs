use crate::{common::*, error::EngineError};
use dashmap::DashMap;
use std::{
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

const MAX_BATCH_SIZE: usize = 5;
const BATCH_WAIT: Duration = Duration::from_millis(100);

type Requests = Arc<DashMap<RequestId, PendingRequest>>;

#[derive(Clone)]
pub struct Scheduler {
    command_tx: mpsc::UnboundedSender<Submit>,
    next_id: Arc<AtomicU64>,
    requests: Requests,
    cancel_token: CancellationToken,
    tasks: Vec<AbortHandle>,
    task_tracker: TaskTracker,
}

impl Scheduler {
    // Starts background tasks; call from within a Tokio runtime.
    pub fn new(
        cancel_token: CancellationToken,
        batch_req_tx: mpsc::UnboundedSender<BatchIn>,
        batch_reply_rx: mpsc::UnboundedReceiver<BatchOut>,
    ) -> Self {
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let requests = Arc::new(DashMap::new());
        let task_tracker = TaskTracker::new();
        let tasks = vec![
            task_tracker
                .spawn(Self::run_dispatch_batch(
                    command_rx,
                    batch_req_tx,
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
            next_id: Arc::new(AtomicU64::new(0)),
            requests,
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

    pub async fn execute(&self, request: Promp) -> Result<LLMResponse, EngineError> {
        if self.cancel_token.is_cancelled() {
            return Err(EngineError::ChannelClosed);
        }
        let request_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (reply_tx, response_rx) = oneshot::channel();
        self.command_tx
            .send(Submit {
                request_id,
                request,
                reply_tx,
            })
            .map_err(|_| EngineError::ChannelClosed)?;
        response_rx.await.map_err(|_| EngineError::ChannelClosed)?
    }

    async fn run_dispatch_batch(
        mut command_rx: mpsc::UnboundedReceiver<Submit>,
        batch_req_tx: mpsc::UnboundedSender<BatchIn>,
        requests: Requests,
    ) {
        loop {
            let deadline = Instant::now() + BATCH_WAIT;
            let mut submissions = Vec::with_capacity(MAX_BATCH_SIZE);
            while submissions.len() < MAX_BATCH_SIZE {
                tokio::select! {
                    biased;
                    _ = sleep_until(deadline) => break,
                    request = command_rx.recv() => match request {
                        Some(submit) => submissions.push(submit),
                        None => break,
                    },
                }
            }
            if submissions.is_empty() {
                if command_rx.is_closed() {
                    break;
                }
                continue;
            }
            let mut batch = Vec::with_capacity(submissions.len());
            for submit in submissions {
                let request_id = submit.request_id;
                requests.insert(
                    request_id,
                    PendingRequest {
                        reply_tx: submit.reply_tx,
                    },
                );
                batch.push(SingleRequest {
                    request_id,
                    request: submit.request,
                });
            }
            if batch_req_tx.send(batch).is_err() {
                requests.clear();
                break;
            }
        }
    }

    async fn run_recv_resp(
        mut batch_reply_rx: mpsc::UnboundedReceiver<BatchOut>,
        requests: Requests,
    ) {
        while let Some(output) = batch_reply_rx.recv().await {
            for item in output {
                if let Some((_, request)) = requests.remove(&item.request_id) {
                    let _ = request.reply_tx.send(Ok(item));
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
    async fn cancellation_wakes_pending_callers_and_closes_ingress() {
        let cancel_token = CancellationToken::new();
        let (batch_req_tx, _batch_req_rx) = mpsc::unbounded_channel();
        let (_batch_reply_tx, batch_reply_rx) = mpsc::unbounded_channel();
        let scheduler = Scheduler::new(cancel_token.child_token(), batch_req_tx, batch_reply_rx);
        let (result, ()) = tokio::join!(
            scheduler.execute(Promp {
                prompt: "hello".into()
            }),
            async {
                cancel_token.cancel();
                scheduler.wait_stopped().await;
            },
        );
        assert!(matches!(result, Err(EngineError::ChannelClosed)));
        assert!(scheduler.requests.is_empty());
        assert!(scheduler.task_tracker.is_empty());
        assert!(scheduler.tasks.iter().all(AbortHandle::is_finished));
        assert!(matches!(
            scheduler
                .execute(Promp {
                    prompt: "after cancel".into()
                })
                .await,
            Err(EngineError::ChannelClosed)
        ));
    }
}
