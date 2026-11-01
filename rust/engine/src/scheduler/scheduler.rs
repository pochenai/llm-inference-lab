use super::{
    forwarding::{Forwarding, LocalDestination, LocalForwarder, RequestForwarder},
    observer::{RequestStats, SchedulerObserver},
    routing::{
        CacheNamespace, DestinationId, RandomPrefixPolicy, RoutingConfig, RoutingPolicy,
        RoutingState,
    },
    runtime::RoutingRuntime,
};
use crate::{error::EngineError, types::*};
use dashmap::DashMap;
use futures::{stream, Stream, TryStreamExt};
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
    time::Instant,
};

use tokio_util::{sync::CancellationToken, task::TaskTracker};

/// Invariant: reply/lifecycle metadata only; never retain the prompt, token history,
/// or buffers whose capacity/backing allocation scales with the full input length.
/// Full input belongs to the forwarding command; routing retains only capped prefixes.
pub(super) struct PendingRequest {
    pub(super) reply_tx: mpsc::UnboundedSender<Result<Option<LLMResponse>, EngineError>>,
    request_id: RequestId,
    pub(super) started: Instant,
    pub(super) first_text: Option<Duration>,
    pub(super) destination: Option<DestinationId>,
}

/// Invariant: per-request bookkeeping must not retain full request inputs, directly
/// or through shared slices/closures. The separate prefix index obeys RoutingConfig limits.
pub(super) type Requests = Arc<DashMap<RequestId, PendingRequest>>;

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
        let forwarding = LocalForwarder::connect([LocalDestination::new(
            DestinationId(0),
            CacheNamespace(0),
            runner_command_tx,
            batch_reply_rx,
        )])
        .expect("one local destination is unique");
        Self::with_routing(
            cancel_token,
            forwarding,
            RoutingConfig::default(),
            RandomPrefixPolicy,
            observers,
        )
        .expect("default local routing configuration is valid")
    }

    /// Configure named destinations and forwarding independently of prefix selection.
    /// The gateway retains reply delivery; Engine-local prefix reuse is not implied.
    pub fn with_routing<F: RequestForwarder, P: RoutingPolicy>(
        cancel_token: CancellationToken,
        forwarding: Forwarding<F>,
        config: RoutingConfig,
        policy: P,
        observers: Vec<Arc<dyn SchedulerObserver>>,
    ) -> Result<Self, EngineError> {
        let state = RoutingState::new(&forwarding.destinations, config)?;
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let requests = Arc::new(DashMap::new());
        let observers: Arc<[Arc<dyn SchedulerObserver>]> = observers.into();
        let task_tracker = TaskTracker::new();
        let lifetimes = forwarding
            .destinations
            .iter()
            .map(|d| (d.id, cancel_token.child_token()))
            .collect();
        let task = task_tracker
            .spawn(
                RoutingRuntime {
                    commands: command_rx,
                    forwarding,
                    state,
                    requests_by_destination: Default::default(),
                    policy,
                    requests: requests.clone(),
                    observers: observers.clone(),
                    cancel: cancel_token.clone(),
                    lifetimes,
                }
                .run(),
            )
            .abort_handle();
        task_tracker.close();
        Ok(Self {
            command_tx,
            requests,
            observers,
            next_id: Arc::new(AtomicU64::new(0)),
            next_session_id: Arc::new(AtomicU64::new(0)),
            cancel_token,
            tasks: vec![task],
            task_tracker,
        })
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
            destination: None,
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

    /// Release an idle session. Clients await their previous turn before closing;
    /// Runner rejects an active session instead of queueing the close.
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

    pub(super) fn finish_request(
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
    pub(super) fn fail_pending(requests: &Requests, observers: &impl SchedulerObserver) {
        let ids: Vec<_> = requests.iter().map(|entry| *entry.key()).collect();
        for id in ids {
            if let Some((_, request)) = requests.remove(&id) {
                Self::finish_request(request, None, Err(EngineError::ChannelClosed), observers);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::{DestinationCandidate, MetricsObserver};

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
            let (commands, mut rx) = mpsc::unbounded_channel();
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
            assert!(matches!(rx.recv().await, Some(RunnerCommand::Execute(_))));
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
            let mut batch = Vec::new();
            while batch.len() < 3 {
                let Some(RunnerCommand::Execute(mut requests)) = rx.recv().await else {
                    panic!("requests")
                };
                batch.append(&mut requests);
            }
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
                    assert_eq!(request.session_id, None);
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
        let request = SessionRequest::new(None, "same payload".to_owned());
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
    async fn session_management_unblocks_on_cancellation_or_runner_disconnect() {
        for opening in [true, false] {
            let cancel = CancellationToken::new();
            let (runner_tx, mut runner_rx) = mpsc::unbounded_channel();
            let (_reply_tx, reply_rx) = mpsc::unbounded_channel();
            let scheduler = Scheduler::new(cancel.child_token(), runner_tx, reply_rx);
            if !opening {
                let open = scheduler.open_session();
                tokio::pin!(open);
                tokio::select! {
                    result = &mut open => panic!("unexpected early open: {result:?}"),
                    command = runner_rx.recv() => {
                        let RunnerCommand::OpenSession { reply, .. } = command.unwrap() else { panic!("open") };
                        reply.send(Ok(())).unwrap();
                    }
                }
                assert_eq!(open.await.unwrap(), 0);
            }
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

    struct MockDestination {
        endpoint: LocalDestination,
        commands: mpsc::UnboundedReceiver<RunnerCommand>,
        replies: mpsc::UnboundedSender<RunnerReply>,
    }

    fn mock_destination(id: u64, namespace: u64) -> MockDestination {
        let (commands, rx) = mpsc::unbounded_channel();
        let (replies, reply_rx) = mpsc::unbounded_channel();
        MockDestination {
            endpoint: LocalDestination::new(
                DestinationId(id),
                CacheNamespace(namespace),
                commands,
                reply_rx,
            ),
            commands: rx,
            replies,
        }
    }

    fn echo_worker(
        mut commands: mpsc::UnboundedReceiver<RunnerCommand>,
        replies: mpsc::UnboundedSender<RunnerReply>,
        destination: u64,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut sessions = std::collections::HashSet::new();
            while let Some(command) = commands.recv().await {
                match command {
                    RunnerCommand::Execute(batch) => {
                        for request in batch {
                            if let Some(session) = request.session_id {
                                assert!(sessions.contains(&session));
                            }
                            let request_id = request.assigned_request_id();
                            let _ = replies.send(RunnerReply::Delta(LLMResponse {
                                request_id,
                                output: format!("{destination}:{}", request.input),
                            }));
                            let _ = replies.send(RunnerReply::Finished {
                                request_id,
                                generated_tokens: 1,
                                result: Ok(()),
                            });
                        }
                    }
                    RunnerCommand::OpenSession { id, reply } => {
                        assert!(sessions.insert(id));
                        let _ = reply.send(Ok(()));
                    }
                    RunnerCommand::CloseSession { id, reply } => {
                        assert!(sessions.remove(&id));
                        let _ = reply.send(Ok(()));
                    }
                }
            }
        })
    }

    #[tokio::test]
    async fn routed_prefixes_keep_full_inputs_and_sessions_stick_to_their_owner(
    ) -> Result<(), EngineError> {
        let cancel = CancellationToken::new();
        let mut endpoints = Vec::new();
        let mut workers = Vec::new();
        for (id, namespace) in [(1, 0), (2, 0), (3, 1)] {
            let MockDestination {
                endpoint,
                commands,
                replies,
            } = mock_destination(id, namespace);
            endpoints.push(endpoint);
            workers.push(echo_worker(commands, replies, id));
        }
        let scheduler = Scheduler::with_routing(
            cancel.child_token(),
            LocalForwarder::connect(endpoints)?,
            RoutingConfig::default(),
            RandomPrefixPolicy,
            vec![],
        )?;
        let input = format!("SYS:{}", "a".repeat(5000));
        let first = scheduler
            .execute(SessionRequest::new(None, input.clone()))
            .await?;
        assert!(first.output.starts_with("1:") || first.output.starts_with("2:"));
        assert_eq!(&first.output[2..], input);
        let next_input = format!("SYS:{}bbb", "a".repeat(5000));
        let second = scheduler
            .execute(SessionRequest::new(None, next_input.clone()))
            .await?;
        assert_eq!(&second.output[..2], &first.output[..2]);
        assert_eq!(&second.output[2..], next_input);
        let session = scheduler.open_session().await?;
        let turn = scheduler
            .execute(SessionRequest::new(Some(session), "first".into()))
            .await?;
        let followup = scheduler
            .execute(SessionRequest::new(Some(session), "unrelated".into()))
            .await?;
        assert_eq!(&turn.output[..2], &followup.output[..2]);
        scheduler.close_session(session).await?;
        assert!(matches!(
            scheduler
                .execute(SessionRequest::new(Some(session), "late".into()))
                .await,
            Err(EngineError::SessionLost(id)) if id == session
        ));
        cancel.cancel();
        scheduler.wait_stopped().await;
        for worker in workers {
            worker.await.unwrap();
        }
        Ok(())
    }

    struct InspectPolicy {
        observations: Arc<std::sync::Mutex<Vec<Vec<DestinationCandidate>>>>,
        selections: AtomicU64,
    }

    impl RoutingPolicy for InspectPolicy {
        fn select(&self, candidates: &[DestinationCandidate]) -> Option<DestinationCandidate> {
            self.observations.lock().unwrap().push(candidates.to_vec());
            let mut candidates = candidates.to_vec();
            candidates.sort_by_key(|c| c.destination);
            if candidates.is_empty() {
                return None;
            }
            let candidate = candidates
                [self.selections.fetch_add(1, Ordering::Relaxed) as usize % candidates.len()];
            Some(candidate)
        }
    }

    #[tokio::test]
    async fn disconnect_isolated_and_duplicate_or_wrong_source_replies_do_not_finish_peers(
    ) -> Result<(), EngineError> {
        let MockDestination {
            endpoint: first,
            commands: mut first_commands,
            replies: first_replies,
        } = mock_destination(1, 0);
        let MockDestination {
            endpoint: second,
            commands: mut second_commands,
            replies: second_replies,
        } = mock_destination(2, 0);
        let cancel = CancellationToken::new();
        let observations = Arc::new(std::sync::Mutex::new(Vec::new()));
        let log = Arc::new(ObserverLog::default());
        let scheduler = Scheduler::with_routing(
            cancel.child_token(),
            LocalForwarder::connect([first, second])?,
            RoutingConfig::default(),
            InspectPolicy {
                observations: observations.clone(),
                selections: AtomicU64::new(0),
            },
            vec![log.clone()],
        )?;
        let dropped_consumer =
            scheduler.execute_stream(SessionRequest::new(None, "first".into()))?;
        let peer = scheduler.execute_stream(SessionRequest::new(None, "second".into()))?;
        futures::pin_mut!(peer);
        let Some(RunnerCommand::Execute(first_batch)) = first_commands.recv().await else {
            panic!("first command")
        };
        let Some(RunnerCommand::Execute(second_batch)) = second_commands.recv().await else {
            panic!("second command")
        };
        let first_id = first_batch[0].assigned_request_id();
        let peer_id = second_batch[0].assigned_request_id();
        // Dropping delivery does not finish an admitted request.
        drop(dropped_consumer);
        assert_eq!(scheduler.requests.len(), 2);
        assert!(observations.lock().unwrap()[1]
            .iter()
            .any(|c| c.destination == DestinationId(1)));
        first_replies
            .send(RunnerReply::Finished {
                request_id: peer_id,
                generated_tokens: 99,
                result: Ok(()),
            })
            .unwrap();
        drop(first_commands);
        second_replies
            .send(RunnerReply::Delta(LLMResponse {
                request_id: peer_id,
                output: "peer survived".into(),
            }))
            .unwrap();
        second_replies
            .send(RunnerReply::Finished {
                request_id: peer_id,
                generated_tokens: 1,
                result: Ok(()),
            })
            .unwrap();
        second_replies
            .send(RunnerReply::Finished {
                request_id: peer_id,
                generated_tokens: 1,
                result: Ok(()),
            })
            .unwrap();
        assert_eq!(peer.try_next().await?.unwrap().output, "peer survived");
        assert!(peer.try_next().await?.is_none());
        // A following request is a routing-loop barrier after terminal processing.
        let last = scheduler.execute_stream(SessionRequest::new(None, "last".into()))?;
        futures::pin_mut!(last);
        let Some(RunnerCommand::Execute(batch)) = second_commands.recv().await else {
            panic!("survivor command")
        };
        let snapshot = observations.lock().unwrap().last().unwrap().clone();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].destination, DestinationId(2));
        let last_id = batch[0].assigned_request_id();
        second_replies
            .send(RunnerReply::Delta(LLMResponse {
                request_id: last_id,
                output: "ok".into(),
            }))
            .unwrap();
        second_replies
            .send(RunnerReply::Finished {
                request_id: last_id,
                generated_tokens: 1,
                result: Ok(()),
            })
            .unwrap();
        assert_eq!(last.try_next().await?.unwrap().output, "ok");
        assert!(last.try_next().await?.is_none());
        assert!(!scheduler.requests.contains_key(&first_id));
        assert!(scheduler.requests.is_empty());
        let finishes = log
            .0
            .lock()
            .unwrap()
            .iter()
            .filter(|e| matches!(e, Observation::Finished { .. }))
            .count();
        assert_eq!(finishes, 3);
        cancel.cancel();
        scheduler.wait_stopped().await;
        Ok(())
    }

    struct RejectFirst(LocalForwarder);

    impl RequestForwarder for RejectFirst {
        fn forward(
            &self,
            destination: DestinationId,
            command: RunnerCommand,
        ) -> Result<(), EngineError> {
            if destination == DestinationId(1) {
                Err(EngineError::ChannelClosed)
            } else {
                self.0.forward(destination, command)
            }
        }
    }

    #[tokio::test]
    async fn custom_forwarder_send_failure_does_not_count_or_stop_other_destinations(
    ) -> Result<(), EngineError> {
        let MockDestination {
            endpoint: first,
            commands: _held_commands,
            replies: _held_replies,
        } = mock_destination(1, 0);
        let MockDestination {
            endpoint: second,
            commands,
            replies,
        } = mock_destination(2, 0);
        let Forwarding {
            forwarder,
            destinations,
            events,
        } = LocalForwarder::connect([first, second])?;
        let forwarding = Forwarding::new(RejectFirst(forwarder), destinations, events);
        let observations = Arc::new(std::sync::Mutex::new(Vec::new()));
        let cancel = CancellationToken::new();
        let scheduler = Scheduler::with_routing(
            cancel.child_token(),
            forwarding,
            RoutingConfig::default(),
            InspectPolicy {
                observations: observations.clone(),
                selections: AtomicU64::new(0),
            },
            vec![],
        )?;
        let worker = echo_worker(commands, replies, 2);
        assert!(matches!(
            scheduler
                .execute(SessionRequest::new(None, "reject".into()))
                .await,
            Err(EngineError::ChannelClosed)
        ));
        assert_eq!(
            scheduler
                .execute(SessionRequest::new(None, "survive".into()))
                .await?
                .output,
            "2:survive"
        );
        assert_eq!(
            observations.lock().unwrap().last().unwrap()[0].destination,
            DestinationId(2)
        );
        assert!(scheduler.requests.is_empty());
        cancel.cancel();
        scheduler.wait_stopped().await;
        worker.await.unwrap();
        Ok(())
    }

    #[tokio::test]
    async fn busy_close_preserves_owner_and_disconnected_owner_unblocks_management(
    ) -> Result<(), EngineError> {
        let MockDestination {
            endpoint: first,
            mut commands,
            replies: _replies,
        } = mock_destination(1, 0);
        let MockDestination {
            endpoint: second,
            commands: other_commands,
            replies: other_replies,
        } = mock_destination(2, 0);
        let cancel = CancellationToken::new();
        let scheduler = Scheduler::with_routing(
            cancel.child_token(),
            LocalForwarder::connect([first, second])?,
            RoutingConfig::default(),
            InspectPolicy {
                observations: Default::default(),
                selections: AtomicU64::new(0),
            },
            vec![],
        )?;
        let worker = echo_worker(other_commands, other_replies, 2);
        let open = scheduler.open_session();
        tokio::pin!(open);
        tokio::select! {
            result = &mut open => panic!("early open: {result:?}"),
            command = commands.recv() => {
                let Some(RunnerCommand::OpenSession { reply, .. }) = command else { panic!("open") };
                reply.send(Ok(())).unwrap();
            }
        }
        let id = open.await?;
        let close = scheduler.close_session(id);
        tokio::pin!(close);
        tokio::select! {
            result = &mut close => panic!("early close: {result:?}"),
            command = commands.recv() => {
                let Some(RunnerCommand::CloseSession { reply, .. }) = command else { panic!("close") };
                reply.send(Err(EngineError::InvalidInput("busy".into()))).unwrap();
            }
        }
        assert!(matches!(close.await, Err(EngineError::InvalidInput(_))));
        let close = scheduler.close_session(id);
        tokio::pin!(close);
        let held_ack = tokio::select! {
            result = &mut close => panic!("early close: {result:?}"),
            command = commands.recv() => command.unwrap(),
        };
        drop(commands);
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(5), close)
                .await
                .unwrap(),
            Err(EngineError::ChannelClosed)
        ));
        drop(held_ack);
        assert_eq!(
            scheduler
                .execute(SessionRequest::new(None, "alive".into()))
                .await?
                .output,
            "2:alive"
        );
        assert!(scheduler
            .execute(SessionRequest::new(Some(id), "lost owner".into()))
            .await
            .is_err());
        cancel.cancel();
        scheduler.wait_stopped().await;
        worker.await.unwrap();
        Ok(())
    }

    #[tokio::test]
    async fn queued_terminal_result_is_drained_on_cancellation() -> Result<(), EngineError> {
        let cancel = CancellationToken::new();
        let (commands, mut rx) = mpsc::unbounded_channel();
        let (replies, reply_rx) = mpsc::unbounded_channel();
        let scheduler = Scheduler::new(cancel.child_token(), commands, reply_rx);
        let output = scheduler.execute_stream(SessionRequest::new(None, "queued".into()))?;
        futures::pin_mut!(output);
        let Some(RunnerCommand::Execute(batch)) = rx.recv().await else {
            panic!("execute")
        };
        let request_id = batch[0].assigned_request_id();
        replies
            .send(RunnerReply::Delta(LLMResponse {
                request_id,
                output: "complete".into(),
            }))
            .unwrap();
        replies
            .send(RunnerReply::Finished {
                request_id,
                generated_tokens: 1,
                result: Ok(()),
            })
            .unwrap();
        cancel.cancel();
        scheduler.task_tracker.wait().await;
        assert_eq!(output.try_next().await?.unwrap().output, "complete");
        assert!(output.try_next().await?.is_none());
        scheduler.wait_stopped().await;
        Ok(())
    }

    #[tokio::test]
    async fn lost_sessions_are_retired_on_execute_and_close_errors() -> Result<(), EngineError> {
        let cancel = CancellationToken::new();
        let (commands, mut rx) = mpsc::unbounded_channel();
        let (replies, reply_rx) = mpsc::unbounded_channel();
        let executed = Arc::new(AtomicU64::new(0));
        let closed = Arc::new(AtomicU64::new(0));
        let (execute_count, close_count) = (executed.clone(), closed.clone());
        let worker = tokio::spawn(async move {
            while let Some(command) = rx.recv().await {
                match command {
                    RunnerCommand::OpenSession { reply, .. } => {
                        let _ = reply.send(Ok(()));
                    }
                    RunnerCommand::Execute(batch) => {
                        for request in batch {
                            execute_count.fetch_add(1, Ordering::Relaxed);
                            let _ = replies.send(RunnerReply::Finished {
                                request_id: request.assigned_request_id(),
                                generated_tokens: 0,
                                result: Err(EngineError::SessionLost(request.session_id.unwrap())),
                            });
                        }
                    }
                    RunnerCommand::CloseSession { id, reply } => {
                        close_count.fetch_add(1, Ordering::Relaxed);
                        let _ = reply.send(Err(EngineError::SessionLost(id)));
                    }
                }
            }
        });
        let scheduler = Scheduler::new(cancel.child_token(), commands, reply_rx);
        for i in 0..100 {
            let id = scheduler.open_session().await?;
            if i % 2 == 0 {
                assert!(
                    matches!(scheduler.execute(SessionRequest::new(Some(id), "delta".into())).await,
                    Err(EngineError::SessionLost(session)) if session == id)
                );
            } else {
                assert!(
                    matches!(scheduler.close_session(id).await, Err(EngineError::SessionLost(session)) if session == id)
                );
            }
            // Once lost, neither a retry nor close may reach the old endpoint again.
            assert!(
                matches!(scheduler.execute(SessionRequest::new(Some(id), "retry".into())).await,
                Err(EngineError::SessionLost(session)) if session == id)
            );
            assert!(
                matches!(scheduler.close_session(id).await, Err(EngineError::SessionLost(session)) if session == id)
            );
        }
        assert_eq!(executed.load(Ordering::Relaxed), 50);
        assert_eq!(closed.load(Ordering::Relaxed), 50);
        assert!(scheduler.requests.is_empty());
        cancel.cancel();
        scheduler.wait_stopped().await;
        worker.await.unwrap();
        Ok(())
    }
}
