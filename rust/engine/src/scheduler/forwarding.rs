use super::routing::{CacheNamespace, Destination, DestinationId};
use crate::{
    error::EngineError,
    types::{RunnerCommand, RunnerReply},
};
use futures::{
    stream::{self, BoxStream, SelectAll},
    Stream, StreamExt,
};
use std::collections::HashMap;
use tokio::sync::mpsc;

/// Replies stay on the gateway path; source identity isolates endpoint failures.
pub enum ForwardEvent {
    Reply {
        destination: DestinationId,
        reply: RunnerReply,
    },
    Disconnected(DestinationId),
}

pub trait RequestForwarder: Send + Sync + 'static {
    /// Ok accepts local ownership, not execution. Later failures must produce events.
    /// Err guarantees the command was not accepted, so no reply will follow it.
    fn forward(
        &self,
        destination: DestinationId,
        command: RunnerCommand,
    ) -> Result<(), EngineError>;
}

/// Couples a transport with endpoint metadata and its independent reply/event stream.
pub struct Forwarding<F> {
    pub(super) forwarder: F,
    pub(super) destinations: Vec<Destination>,
    pub(super) events: BoxStream<'static, ForwardEvent>,
}

impl<F: RequestForwarder> Forwarding<F> {
    pub fn new(
        forwarder: F,
        destinations: Vec<Destination>,
        events: impl Stream<Item = ForwardEvent> + Send + 'static,
    ) -> Self {
        Self {
            forwarder,
            destinations,
            events: events.boxed(),
        }
    }
}

pub struct LocalDestination {
    destination: Destination,
    commands: mpsc::UnboundedSender<RunnerCommand>,
    replies: mpsc::UnboundedReceiver<RunnerReply>,
}

impl LocalDestination {
    pub fn new(
        id: DestinationId,
        namespace: CacheNamespace,
        commands: mpsc::UnboundedSender<RunnerCommand>,
        replies: mpsc::UnboundedReceiver<RunnerReply>,
    ) -> Self {
        Self {
            destination: Destination { id, namespace },
            commands,
            replies,
        }
    }
}

pub struct LocalForwarder {
    endpoints: HashMap<DestinationId, mpsc::UnboundedSender<RunnerCommand>>,
}

impl LocalForwarder {
    /// Reuse Runner channels; no relay task, HTTP server or second response format.
    pub fn connect(
        destinations: impl IntoIterator<Item = LocalDestination>,
    ) -> Result<Forwarding<Self>, EngineError> {
        let mut endpoints = HashMap::new();
        let mut registrations = Vec::new();
        let mut events = SelectAll::new();
        for LocalDestination {
            destination,
            commands,
            replies,
        } in destinations
        {
            if endpoints.insert(destination.id, commands.clone()).is_some() {
                return Err(EngineError::InvalidInput(
                    "duplicate forwarding destination".into(),
                ));
            }
            registrations.push(destination);
            let id = destination.id;
            let replies = stream::unfold(
                (replies, commands),
                move |(mut replies, commands)| async move {
                    let reply = tokio::select! {
                        biased;
                        // Deliver already-queued terminal results before reporting closure.
                        reply = replies.recv() => reply?,
                        _ = commands.closed() => return None,
                    };
                    Some((
                        ForwardEvent::Reply {
                            destination: id,
                            reply,
                        },
                        (replies, commands),
                    ))
                },
            );
            events.push(
                replies
                    .chain(stream::once(async move { ForwardEvent::Disconnected(id) }))
                    .boxed(),
            );
        }
        Ok(Forwarding::new(Self { endpoints }, registrations, events))
    }
}

impl RequestForwarder for LocalForwarder {
    fn forward(
        &self,
        destination: DestinationId,
        command: RunnerCommand,
    ) -> Result<(), EngineError> {
        self.endpoints
            .get(&destination)
            .ok_or(EngineError::ChannelClosed)?
            .send(command)
            .map_err(|_| EngineError::ChannelClosed)
    }
}
