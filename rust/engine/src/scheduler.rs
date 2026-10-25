mod observer;

pub use observer::{MetricsObserver, RequestStats, SchedulerObserver};

use crate::{error::EngineError, types::*};
use dashmap::DashMap;
use futures::{stream, Stream, TryStreamExt};
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
    reply_tx: mpsc::UnboundedSender<Result<Option<LLMResponse>, EngineError>>,
    request_id: RequestId,
    started: Instant,
    first_text: Option<Duration>,
}

type Requests = Arc<DashMap<RequestId, PendingRequest>>;

#[derive(Clone)]
pub struct Scheduler {
    command_tx: mpsc::UnboundedSender<RunnerCommand>,
    requests: Requests,
    observers: Arc<[Arc<dyn SchedulerObserver>]>,
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
        batch_reply_rx: mpsc::UnboundedReceiver<RunnerReply>,
    ) -> Self {
        Self::new_with_observers(cancel_token, runner_command_tx, batch_reply_rx, Vec::new())
    }

    /// Attach observers once at construction; clones share the same plugins.
    pub fn new_with_observers(
        cancel_token: CancellationToken,
        runner_command_tx: mpsc::UnboundedSender<RunnerCommand>,
        batch_reply_rx: mpsc::UnboundedReceiver<RunnerReply>,
        observers: Vec<Arc<dyn SchedulerObserver>>,
    ) -> Self {
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let requests = Arc::new(DashMap::new());
        let observers: Arc<[Arc<dyn SchedulerObserver>]> = observers.into();
        let task_tracker = TaskTracker::new();
        let tasks = vec![
            task_tracker
                .spawn(Self::run_dispatch_batch(
                    command_rx,
                    runner_command_tx,
                    cancel_token.clone(),
                ))
                .abort_handle(),
            task_tracker
                .spawn(Self::run_recv_resp(
                    batch_reply_rx,
                    requests.clone(),
                    cancel_token.clone(),
                    observers.clone(),
                ))
                .abort_handle(),
        ];
        task_tracker.close();
        Self {
            command_tx,
            requests,
            observers,
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
        Self::fail_pending(&self.requests, &self.observers);
    }

    /// Collect the same incremental stream exposed by execute_stream.
    pub async fn execute(&self, request: SessionRequest) -> Result<LLMResponse, EngineError> {
        let responses = self.execute_stream(request)?;
        futures::pin_mut!(responses);
        let mut output: Option<LLMResponse> = None;
        while let Some(response) = responses.try_next().await? {
            match &mut output {
                Some(output) => output.output.push_str(&response.output),
                None => output = Some(response),
            }
        }
        output.ok_or(EngineError::ChannelClosed)
    }

    /// Admit immediately and stream UTF-8 text deltas until completion or one error.
    /// Dropping the stream stops delivery; it does not cancel the admitted request.
    pub fn execute_stream(
        &self,
        request: SessionRequest,
    ) -> Result<impl Stream<Item = Result<LLMResponse, EngineError>> + Send + 'static, EngineError>
    {
        if self.cancel_token.is_cancelled() {
            return Err(EngineError::ChannelClosed);
        }
        let request_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request = request.with_request_id(request_id);
        let (reply_tx, response_rx) = mpsc::unbounded_channel();
        let pending = PendingRequest {
            reply_tx,
            request_id,
            started: Instant::now(),
            first_text: None,
        };
        self.observers.request_received(request_id);
        self.requests.insert(request_id, pending);
        if self.cancel_token.is_cancelled()
            || self
                .command_tx
                .send(RunnerCommand::Execute(vec![request]))
                .is_err()
        {
            if let Some((_, request)) = self.requests.remove(&request_id) {
                Self::finish_request(
                    request,
                    None,
                    Err(EngineError::ChannelClosed),
                    &self.observers,
                );
            }
            return Err(EngineError::ChannelClosed);
        }
        let cancel = self.cancel_token.clone();
        Ok(stream::try_unfold(
            (response_rx, cancel),
            |(mut rx, cancel)| async move {
                let response = tokio::select! {
                    biased;
                    // Preserve already-routed terminal results when Runner disconnects.
                    response = rx.recv() => response.ok_or(EngineError::ChannelClosed)??,
                    _ = cancel.cancelled() => return Err(EngineError::ChannelClosed),
                };
                Ok(response.map(|response| (response, (rx, cancel))))
            },
        ))
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

    fn finish_request(
        request: PendingRequest,
        generated_tokens: Option<usize>,
        result: Result<(), EngineError>,
        observers: &impl SchedulerObserver,
    ) {
        let stats = RequestStats {
            elapsed: request.started.elapsed(),
            time_to_first_text: request.first_text,
            generated_tokens,
        };
        observers.request_finished(request.request_id, &stats, &result);
        let _ = request.reply_tx.send(result.map(|()| None));
    }

    /// Remove ownership before callbacks; plugins may re-enter admission without
    /// holding a DashMap shard lock. Only used after routing stops or admission fails.
    fn fail_pending(requests: &Requests, observers: &impl SchedulerObserver) {
        let ids: Vec<_> = requests.iter().map(|entry| *entry.key()).collect();
        for id in ids {
            if let Some((_, request)) = requests.remove(&id) {
                Self::finish_request(request, None, Err(EngineError::ChannelClosed), observers);
            }
        }
    }

    async fn run_dispatch_batch(
        mut command_rx: mpsc::UnboundedReceiver<RunnerCommand>,
        runner_command_tx: mpsc::UnboundedSender<RunnerCommand>,
        cancel_token: CancellationToken,
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
            // Routing owns completion callbacks, preserving per-request event order.
            cancel_token.cancel();
        }
    }

    async fn run_recv_resp(
        mut batch_reply_rx: mpsc::UnboundedReceiver<RunnerReply>,
        requests: Requests,
        cancel_token: CancellationToken,
        observers: Arc<[Arc<dyn SchedulerObserver>]>,
    ) {
        loop {
            let output = tokio::select! {
                biased;
                _ = cancel_token.cancelled() => break,
                output = batch_reply_rx.recv() => match output { Some(output) => output, None => break },
            };
            match output {
                RunnerReply::Delta(response) => {
                    if let Some((id, mut request)) = requests.remove(&response.request_id) {
                        let first_text =
                            if !response.output.is_empty() && request.first_text.is_none() {
                                let elapsed = request.started.elapsed();
                                request.first_text = Some(elapsed);
                                Some(elapsed)
                            } else {
                                None
                            };
                        // Own the entry while notifying; no routing-map lock spans callbacks.
                        observers.text_delta(&response, first_text);
                        let _ = request.reply_tx.send(Ok(Some(response)));
                        requests.insert(id, request);
                    }
                }
                RunnerReply::Finished {
                    request_id,
                    generated_tokens,
                    result,
                } => {
                    if let Some((_, request)) = requests.remove(&request_id) {
                        Self::finish_request(request, Some(generated_tokens), result, &observers);
                    }
                }
            }
        }
        // Reject admissions racing with transport closure, then finish every pending request.
        cancel_token.cancel();
        Self::fail_pending(&requests, &observers);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    enum Observation {
        Received {
            request_id: RequestId,
        },
        Delta {
            request_id: RequestId,
            output: String,
            first_text: Option<Duration>,
        },
        Finished {
            request_id: RequestId,
            stats: RequestStats,
            success: bool,
        },
    }

    impl Observation {
        fn request_id(&self) -> RequestId {
            match self {
                Self::Received { request_id }
                | Self::Delta { request_id, .. }
                | Self::Finished { request_id, .. } => *request_id,
            }
        }
    }

    #[derive(Default)]
    struct ObserverLog(std::sync::Mutex<Vec<Observation>>);

    impl SchedulerObserver for ObserverLog {
        fn request_received(&self, request_id: RequestId) {
            self.0
                .lock()
                .unwrap()
                .push(Observation::Received { request_id });
        }
        fn text_delta(&self, response: &LLMResponse, first_text: Option<Duration>) {
            self.0.lock().unwrap().push(Observation::Delta {
                request_id: response.request_id,
                output: response.output.clone(),
                first_text,
            });
        }
        fn request_finished(
            &self,
            request_id: RequestId,
            stats: &RequestStats,
            result: &Result<(), EngineError>,
        ) {
            self.0.lock().unwrap().push(Observation::Finished {
                request_id,
                stats: *stats,
                success: result.is_ok(),
            });
        }
    }

    #[tokio::test]
    async fn observers_measure_tokens_and_first_text_even_when_consumer_drops(
    ) -> Result<(), EngineError> {
        use metrics_util::debugging::{DebugValue, DebuggingRecorder};
        let recorder = DebuggingRecorder::new();
        let metrics = metrics::with_local_recorder(&recorder, MetricsObserver::default);
        let log = Arc::new(ObserverLog::default());
        let cancel = CancellationToken::new();
        let (commands, mut rx) = mpsc::unbounded_channel();
        let (replies, reply_rx) = mpsc::unbounded_channel();
        let scheduler = Scheduler::new_with_observers(
            cancel.child_token(),
            commands,
            reply_rx,
            vec![Arc::new(metrics), log.clone()],
        );
        let worker = tokio::spawn(async move {
            while let Some(RunnerCommand::Execute(batch)) = rx.recv().await {
                for request in batch {
                    let request_id = request.assigned_request_id();
                    let chunks: &[&str] = match request.input.as_str() {
                        "good" => &["", "你", "好"],
                        "empty" => &[""],
                        "bad" => &["partial"],
                        _ => &["ignored"],
                    };
                    for &chunk in chunks {
                        replies
                            .send(RunnerReply::Delta(LLMResponse {
                                request_id,
                                output: chunk.into(),
                            }))
                            .unwrap();
                    }
                    // These counts deliberately differ from text-delta counts.
                    let generated_tokens = match request.input.as_str() {
                        "good" => 5,
                        "empty" => 0,
                        "bad" => 2,
                        _ => 4,
                    };
                    let result = if request.input == "bad" {
                        Err(EngineError::InvalidInput("partial failure".into()))
                    } else {
                        Ok(())
                    };
                    replies
                        .send(RunnerReply::Finished {
                            request_id,
                            generated_tokens,
                            result,
                        })
                        .unwrap();
                }
            }
        });
        drop(scheduler.execute_stream(SessionRequest::new(None, "dropped".into()))?);
        let clone = scheduler.clone();
        let (good, empty, bad) = tokio::join!(
            clone.execute(SessionRequest::new(None, "good".into())),
            scheduler.execute(SessionRequest::new(None, "empty".into())),
            scheduler.execute(SessionRequest::new(None, "bad".into())),
        );
        assert_eq!(good?.output, "你好");
        assert_eq!(empty?.output, "");
        assert!(matches!(bad, Err(EngineError::InvalidInput(_))));
        cancel.cancel();
        scheduler.wait_stopped().await;
        worker.await.unwrap();
        let events = log.0.lock().unwrap();
        for id in 0..4 {
            let request_events: Vec<_> = events
                .iter()
                .filter(|event| event.request_id() == id)
                .collect();
            assert!(matches!(
                request_events.first(),
                Some(Observation::Received { .. })
            ));
            let Some(Observation::Finished { stats, success, .. }) = request_events.last() else {
                panic!("missing finish")
            };
            assert_eq!(
                request_events
                    .iter()
                    .filter(|event| matches!(event, Observation::Finished { .. }))
                    .count(),
                1
            );
            assert_eq!(*success, stats.generated_tokens != Some(2));
            let first: Vec<_> = request_events
                .iter()
                .filter_map(|event| match event {
                    Observation::Delta {
                        first_text: Some(elapsed),
                        output,
                        ..
                    } => {
                        assert!(!output.is_empty());
                        Some(*elapsed)
                    }
                    _ => None,
                })
                .collect();
            assert_eq!(first.len(), usize::from(stats.generated_tokens != Some(0)));
            assert_eq!(stats.time_to_first_text, first.first().copied());
            if let Some(elapsed) = stats.time_to_first_text {
                assert!(elapsed <= stats.elapsed);
            }
            assert!(stats.tokens_per_second().is_some());
        }
        let exported = recorder.snapshotter().snapshot().into_vec();
        let get = |name: &str| {
            &exported
                .iter()
                .find(|(key, ..)| key.key().name() == name)
                .unwrap()
                .3
        };
        assert_eq!(get("engine_requests_total"), &DebugValue::Counter(4));
        assert_eq!(
            get("engine_request_failures_total"),
            &DebugValue::Counter(1)
        );
        assert_eq!(
            get("engine_generated_tokens_total"),
            &DebugValue::Counter(11)
        );
        assert_eq!(
            get("engine_output_text_bytes_total"),
            &DebugValue::Counter(20)
        );
        assert_eq!(
            get("engine_requests_in_flight"),
            &DebugValue::Gauge(0.0.into())
        );
        for name in [
            "engine_time_to_first_text_seconds",
            "engine_request_output_tokens_per_second",
        ] {
            let DebugValue::Histogram(values) = get(name) else {
                panic!("histogram")
            };
            assert_eq!(values.len(), 3);
        }
        Ok(())
    }

    #[tokio::test]
    async fn observers_finish_once_with_unknown_tokens_on_shutdown_and_disconnect(
    ) -> Result<(), EngineError> {
        for cause in ["cancel", "reply_closed", "dispatch_closed"] {
            let cancel = CancellationToken::new();
            let (commands, rx) = mpsc::unbounded_channel();
            let (replies, reply_rx) = mpsc::unbounded_channel();
            let log = Arc::new(ObserverLog::default());
            let scheduler = Scheduler::new_with_observers(
                cancel.child_token(),
                commands,
                reply_rx,
                vec![log.clone()],
            );
            let stream = scheduler.execute_stream(SessionRequest::new(None, "pending".into()))?;
            futures::pin_mut!(stream);
            replies
                .send(RunnerReply::Delta(LLMResponse {
                    request_id: 0,
                    output: "partial".into(),
                }))
                .unwrap();
            assert_eq!(stream.try_next().await?.unwrap().output, "partial");
            match cause {
                "cancel" => cancel.cancel(),
                "reply_closed" => drop(replies),
                _ => drop(rx),
            }
            assert!(matches!(
                tokio::time::timeout(Duration::from_secs(5), stream.try_next())
                    .await
                    .unwrap(),
                Err(EngineError::ChannelClosed)
            ));
            tokio::time::timeout(Duration::from_secs(5), scheduler.wait_stopped())
                .await
                .unwrap();
            assert!(scheduler
                .execute_stream(SessionRequest::new(None, "late".into()))
                .is_err());
            let events = log.0.lock().unwrap();
            assert_eq!(events.len(), 3);
            let Observation::Finished { stats, success, .. } = events.last().unwrap() else {
                panic!("finish")
            };
            assert!(!success);
            assert_eq!(stats.generated_tokens, None);
            assert!(stats.time_to_first_text.is_some());
            assert_eq!(stats.tokens_per_second(), None);
        }
        Ok(())
    }

    #[tokio::test]
    async fn streams_deliver_before_completion_and_route_terminal_errors() -> Result<(), EngineError>
    {
        let cancel = CancellationToken::new();
        let (commands, mut rx) = mpsc::unbounded_channel();
        let (replies, reply_rx) = mpsc::unbounded_channel();
        let scheduler = Scheduler::new(cancel.child_token(), commands, reply_rx);
        let finish = Arc::new(tokio::sync::Notify::new());
        let permit = finish.clone();
        let worker = tokio::spawn(async move {
            let Some(RunnerCommand::Execute(batch)) = rx.recv().await else {
                panic!("batch")
            };
            for request in &batch {
                if request.input != "empty" {
                    replies
                        .send(RunnerReply::Delta(LLMResponse {
                            request_id: request.assigned_request_id(),
                            output: request.input.clone(),
                        }))
                        .unwrap();
                }
            }
            permit.notified().await;
            for request in batch {
                let request_id = request.assigned_request_id();
                let result = if request.input == "bad" {
                    Err(EngineError::InvalidInput("after partial output".into()))
                } else {
                    replies
                        .send(RunnerReply::Delta(LLMResponse {
                            request_id,
                            output: if request.input == "empty" {
                                String::new()
                            } else {
                                " tail".into()
                            },
                        }))
                        .unwrap();
                    Ok(())
                };
                replies
                    .send(RunnerReply::Finished {
                        request_id,
                        generated_tokens: 0,
                        result,
                    })
                    .unwrap();
            }
        });
        let good = scheduler.execute_stream(SessionRequest::new(None, "good".into()))?;
        let bad = scheduler.execute_stream(SessionRequest::new(None, "bad".into()))?;
        let empty = scheduler.execute_stream(SessionRequest::new(None, "empty".into()))?;
        futures::pin_mut!(good, bad, empty);
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(5), good.try_next())
                .await
                .unwrap()?
                .unwrap()
                .output,
            "good"
        );
        assert_eq!(bad.try_next().await?.unwrap().output, "bad");
        assert!(
            tokio::time::timeout(Duration::from_millis(10), good.try_next())
                .await
                .is_err()
        );
        finish.notify_one();
        assert_eq!(good.try_next().await?.unwrap().output, " tail");
        assert!(good.try_next().await?.is_none());
        assert!(matches!(
            bad.try_next().await,
            Err(EngineError::InvalidInput(_))
        ));
        assert!(bad.try_next().await?.is_none());
        assert!(empty.try_next().await?.unwrap().output.is_empty());
        assert!(empty.try_next().await?.is_none());
        worker.await.unwrap();
        cancel.cancel();
        scheduler.wait_stopped().await;
        Ok(())
    }

    #[tokio::test]
    async fn per_request_errors_do_not_affect_other_callers() {
        let cancel = CancellationToken::new();
        let (request_tx, mut request_rx) = mpsc::unbounded_channel::<RunnerCommand>();
        let (reply_tx, reply_rx) = mpsc::unbounded_channel();
        let scheduler = Scheduler::new(cancel.child_token(), request_tx, reply_rx);
        let worker = tokio::spawn(async move {
            while let Some(RunnerCommand::Execute(batch)) = request_rx.recv().await {
                for request in batch {
                    let request_id = request.assigned_request_id();
                    let result = if request.input == "bad" {
                        Err(EngineError::InvalidInput("bad request".into()))
                    } else {
                        reply_tx
                            .send(RunnerReply::Delta(LLMResponse {
                                request_id,
                                output: request.input,
                            }))
                            .unwrap();
                        Ok(())
                    };
                    reply_tx
                        .send(RunnerReply::Finished {
                            request_id,
                            generated_tokens: 0,
                            result,
                        })
                        .unwrap();
                }
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
        Scheduler::run_dispatch_batch(rx, batch_tx, CancellationToken::new()).await;
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
                for request in batch {
                    let id = request.assigned_request_id();
                    assert_eq!(request.session_id, Some(7));
                    reply_tx
                        .send(RunnerReply::Delta(LLMResponse {
                            request_id: id,
                            output: request.input,
                        }))
                        .unwrap();
                    if reply_tx
                        .send(RunnerReply::Finished {
                            request_id: id,
                            generated_tokens: 0,
                            result: Ok(()),
                        })
                        .is_err()
                    {
                        return;
                    }
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
        Scheduler::run_dispatch_batch(rx, runner_tx, CancellationToken::new()).await;
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
