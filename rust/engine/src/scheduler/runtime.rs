use super::{
    forwarding::{ForwardEvent, Forwarding, RequestForwarder},
    routing::{DestinationCandidate, DestinationId, RoutingPolicy, RoutingState},
    scheduler::{Requests, Scheduler},
    SchedulerObserver,
};
use crate::{error::EngineError, types::*};
use futures::{stream::FuturesUnordered, FutureExt, StreamExt};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

struct SessionCompletion {
    id: SessionId,
    destination: DestinationId,
    opening: bool,
    reply: oneshot::Sender<Result<(), EngineError>>,
    result: Result<(), EngineError>,
}

/// One owner serializes routing/accounting transitions; never awaits GPU work.
pub(super) struct RoutingRuntime<F, P> {
    pub(super) commands: mpsc::UnboundedReceiver<RunnerCommand>,
    pub(super) forwarding: Forwarding<F>,
    pub(super) state: RoutingState,
    /// Reverse index of unresolved forwards, used only for targeted terminal cleanup.
    /// Disconnect visits only this destination's requests, not the global request map.
    /// This actor also owns future async completion updates; client drop is not completion.
    pub(super) requests_by_destination: HashMap<DestinationId, HashSet<RequestId>>,
    pub(super) policy: P,
    pub(super) requests: Requests,
    pub(super) observers: Arc<[Arc<dyn SchedulerObserver>]>,
    pub(super) cancel: CancellationToken,
    pub(super) lifetimes: HashMap<DestinationId, CancellationToken>,
}

impl<F: RequestForwarder, P: RoutingPolicy> RoutingRuntime<F, P> {
    pub(super) async fn run(mut self) {
        let mut sessions = FuturesUnordered::new();
        loop {
            if self.cancel.is_cancelled() {
                // Preserve queued terminal results without waiting for additional GPU work.
                while let Some(Some(event)) = self.forwarding.events.next().now_or_never() {
                    self.on_event(event);
                }
                break;
            }
            // Fair selection keeps busy response streams from starving ingress/session acks.
            tokio::select! {
                event = self.forwarding.events.next() => match event {
                    Some(event) => self.on_event(event),
                    None => break,
                },
                _ = self.cancel.cancelled() => continue,
                Some(completion) = sessions.next() => {
                    let completion: SessionCompletion = completion;
                    if matches!(&completion.result, Err(EngineError::SessionLost(id)) if *id == completion.id) {
                        self.state.closed(completion.id, completion.destination);
                    }
                    let result = completion.result.and_then(|()| {
                        if completion.opening { self.state.opened(completion.id, completion.destination) }
                        else { self.state.closed(completion.id, completion.destination); Ok(()) }
                    });
                    // Placement is installed before the caller receives its acknowledgement.
                    let created = completion.opening && result.is_ok();
                    if completion.reply.send(result).is_err() && created {
                        self.state.closed(completion.id, completion.destination);
                        let (reply, _) = oneshot::channel();
                        let _ = self.forwarding.forwarder.forward(completion.destination, RunnerCommand::CloseSession { id: completion.id, reply });
                    }
                },
                command = self.commands.recv() => match command {
                    Some(RunnerCommand::Execute(batch)) => {
                        for request in batch { self.execute(request); }
                    }
                    Some(command) => {
                        let (id, reply, opening) = match command {
                            RunnerCommand::OpenSession { id, reply } => (id, reply, true),
                            RunnerCommand::CloseSession { id, reply } => (id, reply, false),
                            RunnerCommand::Execute(_) => unreachable!(),
                        };
                        let selected = if opening {
                            self.route(None).map(|candidate| candidate.destination)
                        } else {
                            self.state.session_destination(id)
                        };
                        let destination = match selected {
                            Ok(destination) => destination,
                            Err(error) => { let _ = reply.send(Err(error)); continue; }
                        };
                        let (ack, response) = oneshot::channel();
                        let command = if opening { RunnerCommand::OpenSession { id, reply: ack } }
                            else { RunnerCommand::CloseSession { id, reply: ack } };
                        if let Err(error) = self.forwarding.forwarder.forward(destination, command) {
                            let _ = reply.send(Err(error));
                            self.disconnect(destination);
                            continue;
                        }
                        let lifetime = self.lifetimes[&destination].clone();
                        sessions.push(async move {
                            let result = tokio::select! {
                                biased;
                                result = response => result.unwrap_or(Err(EngineError::ChannelClosed)),
                                _ = lifetime.cancelled() => Err(EngineError::ChannelClosed),
                            };
                            SessionCompletion { id, destination, opening, reply, result }
                        });
                    }
                    None => break,
                },
            }
            if !self.state.any_available() {
                break;
            }
        }
        self.cancel.cancel();
        Scheduler::fail_pending(&self.requests, &self.observers);
    }

    fn on_event(&mut self, event: ForwardEvent) {
        match event {
            ForwardEvent::Reply { destination, reply } => self.receive(destination, reply),
            ForwardEvent::Disconnected(destination) => self.disconnect(destination),
        }
    }

    /// None opens an empty session; Some borrows an execution request without copying input.
    /// Existing sessions keep their next hop; otherwise select by prefix, or randomly on a miss.
    fn route(&self, request: Option<&SessionRequest>) -> Result<DestinationCandidate, EngineError> {
        if let Some(session) = request.and_then(|request| request.session_id) {
            let destination = self.state.session_destination(session)?;
            return Ok(DestinationCandidate {
                destination,
                matched_prefix_bytes: 0,
            });
        }
        let matches = request
            .map(|request| self.state.prefix_matches(&request.input))
            .unwrap_or_default();
        let candidates: Vec<_> = self
            .state
            .available_destinations()
            .map(|destination| DestinationCandidate {
                destination,
                matched_prefix_bytes: matches.get(&destination).copied().unwrap_or(0),
            })
            .collect();
        let selected = self
            .policy
            .select(&candidates)
            .ok_or(EngineError::ChannelClosed)?;
        candidates
            .into_iter()
            .find(|candidate| candidate.destination == selected.destination)
            .ok_or_else(|| {
                EngineError::Internal("routing policy selected an ineligible destination".into())
            })
    }

    fn execute(&mut self, request: SessionRequest) {
        let id = request.assigned_request_id();
        let decision = self.route(Some(&request));
        let destination = match decision {
            Ok(decision) => decision.destination,
            Err(error) => {
                if let Some((_, request)) = self.requests.remove(&id) {
                    Scheduler::finish_request(request, None, Err(error), &self.observers);
                }
                return;
            }
        };
        let Some(mut pending) = self.requests.get_mut(&id) else {
            return;
        };
        pending.destination = Some(destination);
        drop(pending);
        // Borrow before forwarding takes ownership; only the index copies capped labels.
        // This actor cannot expose the hint between staging and send-failure cleanup.
        if request.session_id.is_none() {
            self.state.record_dispatch(&request.input, destination);
        }
        if let Err(error) = self
            .forwarding
            .forwarder
            .forward(destination, RunnerCommand::Execute(vec![request]))
        {
            if let Some((_, request)) = self.requests.remove(&id) {
                Scheduler::finish_request(request, None, Err(error), &self.observers);
            }
            self.disconnect(destination);
            return;
        }
        self.requests_by_destination
            .entry(destination)
            .or_default()
            .insert(id);
    }

    fn receive(&mut self, destination: DestinationId, reply: RunnerReply) {
        match reply {
            RunnerReply::Delta(response) => {
                if let Some((id, mut request)) =
                    self.requests.remove_if(&response.request_id, |_, p| {
                        p.destination == Some(destination)
                    })
                {
                    let first_text = if !response.output.is_empty() && request.first_text.is_none()
                    {
                        let elapsed = request.started.elapsed();
                        request.first_text = Some(elapsed);
                        Some(elapsed)
                    } else {
                        None
                    };
                    self.observers.text_delta(&response, first_text);
                    let _ = request.reply_tx.send(Ok(Some(response)));
                    self.requests.insert(id, request);
                }
            }
            RunnerReply::Finished {
                request_id,
                generated_tokens,
                result,
            } => {
                if let Some((_, request)) = self
                    .requests
                    .remove_if(&request_id, |_, p| p.destination == Some(destination))
                {
                    if let Err(EngineError::SessionLost(session)) = &result {
                        self.state.closed(*session, destination);
                    }
                    self.remove_pending(destination, request_id);
                    Scheduler::finish_request(
                        request,
                        Some(generated_tokens),
                        result,
                        &self.observers,
                    );
                }
            }
        }
    }

    /// Called after terminal cleanup removed the request exactly once.
    fn remove_pending(&mut self, destination: DestinationId, request_id: RequestId) {
        let requests = self
            .requests_by_destination
            .get_mut(&destination)
            .expect("forward must be indexed before terminal handling");
        assert!(
            requests.remove(&request_id),
            "forward missing from destination index"
        );
    }

    fn disconnect(&mut self, destination: DestinationId) {
        self.state.disconnected(destination);
        if let Some(lifetime) = self.lifetimes.get(&destination) {
            lifetime.cancel();
        }
        let Some(ids) = self.requests_by_destination.remove(&destination) else {
            return;
        };
        for id in ids {
            if let Some((_, request)) = self.requests.remove(&id) {
                Scheduler::finish_request(
                    request,
                    None,
                    Err(EngineError::ChannelClosed),
                    &self.observers,
                );
            }
        }
    }
}
