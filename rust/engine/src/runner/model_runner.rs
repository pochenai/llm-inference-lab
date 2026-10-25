use std::{future::Future, sync::Arc};

use cuda_async::{device_context::global_policy, scheduling_policies::SchedulingPolicy};
use dashmap::DashMap;
use tokenizers::Encoding;
use tokio::sync::{mpsc, Mutex, OwnedMutexGuard};
use tokio_util::sync::CancellationToken;

use crate::{
    error::EngineError,
    runner::{
        kv_pool::{KVContext, KVPool, KVPoolError, NaiveKVPool, NaiveKVPoolConfig},
        model::{CausalModel, ModelError, ModelInput, SimpleMLPEngine},
    },
    types::{
        BatchIn, LLMResponse, RunnerCommand, RunnerReply, SamplingConfig, SessionId, SessionRequest,
    },
};

type Policy = Arc<dyn SchedulingPolicy + 'static>;

/// Static factory contract; each implementation returns its concrete, Send future.
pub trait AsyncFactory<Args>: Send {
    type Output: Send;

    fn create(self, args: Args) -> impl Future<Output = Result<Self::Output, EngineError>> + Send;
}

impl<F, Fut, T> AsyncFactory<()> for F
where
    F: FnOnce() -> Fut + Send,
    Fut: Future<Output = Result<T, EngineError>> + Send,
    T: Send,
{
    type Output = T;

    fn create(self, (): ()) -> impl Future<Output = Result<T, EngineError>> + Send {
        self()
    }
}

impl<F, Fut, T> AsyncFactory<Policy> for F
where
    F: FnOnce(Policy) -> Fut + Send,
    Fut: Future<Output = Result<T, EngineError>> + Send,
    T: Send,
{
    type Output = T;

    fn create(self, policy: Policy) -> impl Future<Output = Result<T, EngineError>> + Send {
        self(policy)
    }
}

pub struct DefaultModelFactory;

impl AsyncFactory<()> for DefaultModelFactory {
    type Output = SimpleMLPEngine;

    async fn create(self, (): ()) -> Result<Self::Output, EngineError> {
        Ok(SimpleMLPEngine::new(0).await?)
    }
}

pub struct DefaultPoolFactory;

impl AsyncFactory<Policy> for DefaultPoolFactory {
    type Output = Arc<NaiveKVPool>;

    async fn create(self, policy: Policy) -> Result<Self::Output, EngineError> {
        Ok(NaiveKVPool::new(NaiveKVPoolConfig::default(), &policy).await?)
    }
}

pub struct RunnerBuilder<
    MF = DefaultModelFactory,
    PF = DefaultPoolFactory,
    SF = fn() -> Result<Policy, EngineError>,
> {
    model_factory: MF,
    pool_factory: PF,
    policy_factory: SF,
}

impl Default for RunnerBuilder {
    fn default() -> Self {
        Self {
            model_factory: DefaultModelFactory,
            pool_factory: DefaultPoolFactory,
            policy_factory: || Ok(global_policy(0)?),
        }
    }
}

impl<MF, PF, SF> RunnerBuilder<MF, PF, SF> {
    pub fn with_model_factory<F>(self, factory: F) -> RunnerBuilder<F, PF, SF> {
        RunnerBuilder {
            model_factory: factory,
            pool_factory: self.pool_factory,
            policy_factory: self.policy_factory,
        }
    }

    pub fn with_pool_factory<F>(self, factory: F) -> RunnerBuilder<MF, F, SF> {
        RunnerBuilder {
            model_factory: self.model_factory,
            pool_factory: factory,
            policy_factory: self.policy_factory,
        }
    }

    pub fn with_policy_factory<F>(self, factory: F) -> RunnerBuilder<MF, PF, F> {
        RunnerBuilder {
            model_factory: self.model_factory,
            pool_factory: self.pool_factory,
            policy_factory: factory,
        }
    }

    pub fn build<M, P>(
        self,
        command_rx: mpsc::UnboundedReceiver<RunnerCommand>,
        batch_reply_tx: mpsc::UnboundedSender<RunnerReply>,
    ) -> impl Future<Output = Result<ModelRunner<M, P>, EngineError>> + Send
    where
        P: KVPool,
        M: CausalModel<P>,
        MF: AsyncFactory<(), Output = M>,
        PF: AsyncFactory<Policy, Output = Arc<P>>,
        SF: FnOnce() -> Result<Policy, EngineError> + Send,
    {
        async move {
            let policy = (self.policy_factory)()?;
            let model = self.model_factory.create(()).await?;
            let pool = self.pool_factory.create(policy.clone()).await?;
            Ok(ModelRunner::new(
                command_rx,
                batch_reply_tx,
                model,
                pool,
                policy,
            ))
        }
    }
}

/// Each entry has its own async lock; the directory is never held across GPU awaits.
type SessionEntry<P> = Arc<Mutex<Option<SessionState<P>>>>;

pub struct ModelRunner<M = SimpleMLPEngine, P: KVPool = NaiveKVPool> {
    command_rx: mpsc::UnboundedReceiver<RunnerCommand>,
    batch_reply_tx: mpsc::UnboundedSender<RunnerReply>,
    model: M,
    pool: Arc<P>,
    policy: Policy,
    sessions: DashMap<SessionId, SessionEntry<P>>,
}

struct SessionState<P: KVPool> {
    tokens: Arc<Vec<u32>>,
    kv: KVContext<P>,
    sampling: SamplingConfig,
}

impl<M: CausalModel<P>, P: KVPool> ModelRunner<M, P> {
    fn new(
        command_rx: mpsc::UnboundedReceiver<RunnerCommand>,
        batch_reply_tx: mpsc::UnboundedSender<RunnerReply>,
        model: M,
        pool: Arc<P>,
        policy: Policy,
    ) -> Self {
        Self {
            command_rx,
            batch_reply_tx,
            model,
            pool,
            policy,
            sessions: DashMap::new(),
        }
    }

    /// Allocate an empty session. Opening and executing different sessions need only &self.
    fn open_session(&self, id: SessionId) -> Result<(), EngineError> {
        match self.sessions.entry(id) {
            dashmap::mapref::entry::Entry::Occupied(_) => {
                Err(EngineError::InvalidInput("session already exists".into()))
            }
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                entry.insert(Arc::new(Mutex::new(Some(SessionState {
                    tokens: Arc::new(Vec::new()),
                    kv: KVContext::new(self.pool.clone())?,
                    sampling: SamplingConfig::default(),
                }))));
                Ok(())
            }
        }
    }

    /// Wait for this session's active turn, release its state, and invalidate queued users
    /// of the old entry. A newly opened session with the same ID has a distinct entry.
    async fn close_session(&self, id: SessionId) -> Result<(), EngineError> {
        let entry = self
            .sessions
            .get(&id)
            .map(|entry| entry.value().clone())
            .ok_or_else(|| EngineError::InvalidInput("unknown session".into()))?;
        let mut guard = entry.clone().lock_owned().await;
        if let Some(state) = guard.take() {
            state.kv.release();
        }
        self.sessions
            .remove_if(&id, |_, current| Arc::ptr_eq(current, &entry));
        Ok(())
    }

    /// Stream each request independently and finish it after restoring or releasing its KV.
    /// Cancellation drops states still executing, even if they already emitted text.
    async fn execute_model_batch(
        &self,
        encoded: Vec<SessionRequest<Encoding>>,
        mut guards: Vec<Option<OwnedMutexGuard<Option<SessionState<P>>>>>,
        replies: &mpsc::UnboundedSender<RunnerReply>,
    ) {
        let mut states: Vec<_> = guards
            .iter_mut()
            .map(|guard| guard.as_mut().unwrap().take())
            .collect();
        let original_progress: Vec<_> = states
            .iter()
            .map(|state| {
                let state = state.as_ref().unwrap();
                (state.tokens.len(), state.kv.ready_len())
            })
            .collect();
        for (state, request) in states.iter_mut().zip(&encoded) {
            let tokens = Arc::make_mut(&mut state.as_mut().unwrap().tokens);
            tokens.reserve(
                request.input.len()
                    + request
                        .sampling
                        .as_ref()
                        .unwrap()
                        .max_new_tokens
                        .unwrap_or(1),
            );
            tokens.extend_from_slice(request.input.get_ids());
        }
        let mut remaining: Vec<_> = encoded
            .iter()
            .map(|request| {
                request
                    .sampling
                    .as_ref()
                    .unwrap()
                    .max_new_tokens
                    .unwrap_or(1)
            })
            .collect();
        let mut errors = vec![None::<ModelError>; states.len()];
        let mut decode_errors = vec![None::<ModelError>; states.len()];
        let mut emitted = vec![false; states.len()];
        let mut decoders: Vec<_> = (0..states.len())
            .map(|_| Some(self.model.tokenizer().decode_stream(true)))
            .collect();
        let mut active = Vec::with_capacity(states.len());
        loop {
            // Settle stopped requests before executing any peer's next step, including zero budgets.
            for index in 0..states.len() {
                if remaining[index] != 0 {
                    continue;
                }
                let Some(mut state) = states[index].take() else {
                    continue;
                };
                let request_id = encoded[index].assigned_request_id();
                let generated = errors[index].is_none();
                drop(decoders[index].take());
                let result = match errors[index].take().or_else(|| decode_errors[index].take()) {
                    Some(error) => Err(error.into()),
                    None => {
                        // No extra decode/flush: emit only chunks produced by the library.
                        if !emitted[index] {
                            let _ = replies.send(RunnerReply::Delta(LLMResponse {
                                request_id,
                                output: String::new(),
                            }));
                        }
                        Ok(())
                    }
                };
                let mut guard = guards[index].take().unwrap();
                let (old_len, old_ready) = original_progress[index];
                if generated {
                    state.sampling = encoded[index].sampling.as_ref().unwrap().clone();
                    *guard = Some(state);
                } else if !state.kv.is_pending() && state.kv.ready_len() == old_ready {
                    Arc::make_mut(&mut state.tokens).truncate(old_len);
                    *guard = Some(state);
                } else {
                    drop(state);
                }
                // Unlock persistent sessions and release temporary contexts before announcing completion.
                drop(guard);
                let _ = replies.send(RunnerReply::Finished { request_id, result });
            }
            active.clear();
            let mut inputs: Vec<_> = states
                .iter_mut()
                .enumerate()
                .filter_map(|(index, state)| {
                    state.as_mut().map(|state| {
                        active.push(index);
                        ModelInput {
                            token_ids: &state.tokens,
                            kv: &mut state.kv,
                        }
                    })
                })
                .collect();
            if inputs.is_empty() {
                break;
            }
            let outputs = self
                .model
                .next_tokens(&mut inputs, self.policy.clone())
                .await;
            assert_eq!(
                outputs.len(),
                active.len(),
                "one next-token result per input"
            );
            for (index, output) in active.iter().copied().zip(outputs) {
                match output {
                    Ok(token) => {
                        Arc::make_mut(&mut states[index].as_mut().unwrap().tokens).push(token);
                        remaining[index] -= 1;
                        match decoders[index].as_mut().unwrap().step(token) {
                            Ok(Some(output)) => {
                                emitted[index] = true;
                                let _ = replies.send(RunnerReply::Delta(LLMResponse {
                                    request_id: encoded[index].assigned_request_id(),
                                    output,
                                }));
                            }
                            Ok(None) => {}
                            Err(error) => {
                                decode_errors[index] =
                                    Some(ModelError::Execution(error.to_string()));
                                remaining[index] = 0;
                            }
                        }
                        let eos = match &encoded[index].sampling.as_ref().unwrap().eos_token_id {
                            Some(crate::runner::loader::EosTokenId::One(id)) => token == *id,
                            Some(crate::runner::loader::EosTokenId::Many(ids)) => {
                                ids.contains(&token)
                            }
                            None => self.model.is_eos(token),
                        };
                        if eos {
                            remaining[index] = 0;
                        }
                    }
                    Err(error) => {
                        errors[index] = Some(error);
                        remaining[index] = 0;
                    }
                }
            }
        }
    }

    /// Scheduler supplies candidates with unique Some(session_id). Runner validates each
    /// request, then splits execution batches by local free KV tokens and context rows.
    /// run is serial; concurrent multi-batch execution would need consistent lock order.
    async fn prepare_batch(&self, batch: BatchIn, replies: &mpsc::UnboundedSender<RunnerReply>) {
        let mut candidates = batch.into_iter();
        let mut encoded = Vec::new();
        let mut guards = Vec::new();
        let mut append_tokens = 0usize;
        loop {
            let prepared = if let Some(request) = candidates.next() {
                let request_id = request.assigned_request_id();
                let prepared = async {
                    let tokens = self
                        .model
                        .tokenizer()
                        .encode(request.input.as_str(), true)
                        .map_err(|error| EngineError::InvalidInput(error.to_string()))?;
                    if tokens.is_empty() {
                        return Err(EngineError::InvalidInput(
                            "prompt delta must not be empty".into(),
                        ));
                    }
                    let guard = match request.session_id {
                        Some(id) => {
                            let entry = self
                                .sessions
                                .get(&id)
                                .map(|entry| entry.value().clone())
                                .ok_or_else(|| {
                                EngineError::InvalidInput("unknown session".into())
                            })?;
                            let guard = entry.lock_owned().await;
                            if guard.is_none() {
                                return Err(EngineError::InvalidInput(
                                    "session closed or invalidated".into(),
                                ));
                            }
                            Some(guard)
                        }
                        None => None,
                    };
                    let state = guard.as_ref().map(|guard| guard.as_ref().unwrap());
                    let old_len = state.map_or(0, |state| state.tokens.len());
                    let ready_len = state.map_or(0, |state| state.kv.ready_len());
                    let max_seq_len =
                        state.map_or(self.pool.max_seq_len(), |state| state.kv.max_seq_len());
                    let full_len = old_len
                        .checked_add(tokens.len())
                        .filter(|&len| len <= max_seq_len)
                        .ok_or_else(|| {
                            EngineError::InvalidInput("KV context capacity exceeded".into())
                        })?;
                    let sampling = request.sampling.clone().unwrap_or_else(|| {
                        state.map_or_else(SamplingConfig::default, |state| state.sampling.clone())
                    });
                    if sampling.do_sample == Some(true)
                        || sampling.temperature.is_some()
                        || sampling.top_k.is_some()
                        || sampling.top_p.is_some()
                    {
                        return Err(EngineError::InvalidInput(
                            "only pure greedy generation is supported".into(),
                        ));
                    }
                    self.model.validate_input(tokens.get_ids())?;
                    let generated_kv = sampling.max_new_tokens.unwrap_or(1).saturating_sub(1);
                    let end = full_len
                        .checked_add(generated_kv)
                        .filter(|&end| end <= max_seq_len)
                        .ok_or_else(|| {
                            EngineError::InvalidInput(
                                "generation exceeds KV context capacity".into(),
                            )
                        })?;
                    let mut encoded = SessionRequest::new(request.session_id, tokens)
                        .with_request_id(request.assigned_request_id());
                    encoded.sampling = Some(sampling);
                    Ok((encoded, guard, end - ready_len))
                }
                .await;
                let (request, guard, needed) = match prepared {
                    Ok(prepared) => prepared,
                    Err(error) => {
                        let _ = replies.send(RunnerReply::Finished {
                            request_id,
                            result: Err(error),
                        });
                        continue;
                    }
                };
                Some((request, guard, needed))
            } else {
                None
            };
            // Capacity boundaries and end-of-input flush through the same execution path.
            let flush = prepared.as_ref().is_none_or(|(_, guard, needed)| {
                !self.pool.can_append(
                    append_tokens.saturating_add(*needed),
                    usize::from(guard.is_none()),
                )
            });
            if flush && !encoded.is_empty() {
                self.execute_model_batch(
                    std::mem::take(&mut encoded),
                    std::mem::take(&mut guards),
                    replies,
                )
                .await;
                append_tokens = 0;
            }
            let Some((request, guard, needed)) = prepared else {
                break;
            };
            let new_contexts = usize::from(guard.is_none());
            if !self
                .pool
                .can_append(append_tokens.saturating_add(needed), new_contexts)
            {
                drop(guard);
                let _ = replies.send(RunnerReply::Finished {
                    request_id: request.assigned_request_id(),
                    result: Err(KVPoolError::CapacityExhausted(
                        "insufficient KV capacity for request".into(),
                    )
                    .into()),
                });
                continue;
            }
            let guard = match guard {
                Some(guard) => guard,
                None => {
                    let kv = match KVContext::new(self.pool.clone()) {
                        Ok(kv) => kv,
                        Err(error) => {
                            let _ = replies.send(RunnerReply::Finished {
                                request_id: request.assigned_request_id(),
                                result: Err(error.into()),
                            });
                            continue;
                        }
                    };
                    Arc::new(Mutex::new(Some(SessionState {
                        tokens: Arc::new(Vec::new()),
                        kv,
                        sampling: SamplingConfig::default(),
                    })))
                    .lock_owned()
                    .await
                }
            };
            append_tokens += needed;
            encoded.push(request);
            guards.push(Some(guard));
        }
    }

    /// Execute Scheduler commands serially, preserving session lifecycle ordering.
    pub async fn run(mut self, cancel_token: CancellationToken) {
        loop {
            let command = tokio::select! {
                biased;
                _ = cancel_token.cancelled() => break,
                command = self.command_rx.recv() => match command { Some(command) => command, None => break },
            };
            match command {
                RunnerCommand::Execute(batch) => {
                    // Delta and Finished are sent inside execution; this await only drives the batch.
                    tokio::select! {
                        biased;
                        _ = cancel_token.cancelled() => break,
                        _ = self.prepare_batch(batch, &self.batch_reply_tx) => {},
                    }
                }
                RunnerCommand::OpenSession { id, reply } => {
                    let _ = reply.send(self.open_session(id));
                }
                RunnerCommand::CloseSession { id, reply } => {
                    let _ = reply.send(self.close_session(id).await);
                }
            }
        }
        self.sessions.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RequestId;
    use std::time::Duration;
    use tokenizers::Tokenizer;

    impl<M: CausalModel<P>, P: KVPool> ModelRunner<M, P> {
        async fn execute_collect(
            &self,
            batch: BatchIn,
        ) -> Vec<(RequestId, Result<LLMResponse, EngineError>)> {
            let mut results: Vec<_> = batch
                .iter()
                .map(|request| {
                    let id = request.assigned_request_id();
                    (
                        id,
                        Ok(LLMResponse {
                            request_id: id,
                            output: String::new(),
                        }),
                    )
                })
                .collect();
            let (replies, mut rx) = mpsc::unbounded_channel();
            self.prepare_batch(batch, &replies).await;
            let mut finished = std::collections::HashSet::new();
            while let Ok(reply) = rx.try_recv() {
                match reply {
                    RunnerReply::Delta(response) => {
                        let result = &mut results
                            .iter_mut()
                            .find(|(id, _)| *id == response.request_id)
                            .unwrap()
                            .1;
                        result.as_mut().unwrap().output.push_str(&response.output);
                    }
                    RunnerReply::Finished { request_id, result } => {
                        assert!(finished.insert(request_id));
                        if let Err(error) = result {
                            results
                                .iter_mut()
                                .find(|(id, _)| *id == request_id)
                                .unwrap()
                                .1 = Err(error);
                        }
                    }
                }
            }
            assert_eq!(finished.len(), results.len());
            results
        }

        async fn execute_session(
            &self,
            request: SessionRequest,
        ) -> Result<LLMResponse, EngineError> {
            self.execute_collect(vec![request]).await.pop().unwrap().1
        }
    }

    #[test]
    fn library_stream_decodes_complete_reference_without_a_flush() {
        use tokenizers::{decoders::fuse::Fuse, models::wordlevel::WordLevel, AddedToken};
        let ids = [9707, 0, 2585, 646, 358, 7789, 498, 3351, 30, 151645];
        let words = [
            "Hello",
            "!",
            " How",
            " can",
            " I",
            " assist",
            " you",
            " today",
            "?",
            "<|im_end|>",
        ];
        let mut tokenizer = Tokenizer::new(
            WordLevel::builder()
                .vocab(
                    words
                        .into_iter()
                        .zip(ids)
                        .map(|(word, id)| (word.to_owned(), id))
                        .collect(),
                )
                .build()
                .unwrap(),
        );
        tokenizer.with_decoder(Some(Fuse::new()));
        tokenizer
            .add_special_tokens([AddedToken::from("<|im_end|>", true)])
            .unwrap();
        let mut decoder = tokenizer.decode_stream(true);
        let mut output = String::new();
        for id in ids {
            if let Some(delta) = decoder.step(id).unwrap() {
                output.push_str(&delta);
            }
        }
        assert_eq!(output, "Hello! How can I assist you today?");
        assert_eq!(output, tokenizer.decode(&ids, true).unwrap());
    }

    #[tokio::test]
    async fn stopped_requests_finish_and_release_kv_before_a_peers_next_step(
    ) -> Result<(), EngineError> {
        use futures::TryStreamExt;

        struct PausedModel {
            inner: SimpleMLPEngine,
            entered: Arc<tokio::sync::Notify>,
            resume: Arc<tokio::sync::Notify>,
        }
        impl CausalModel<NaiveKVPool> for PausedModel {
            fn tokenizer(&self) -> &Tokenizer {
                self.inner.tokenizer()
            }
            fn validate_input(&self, ids: &[u32]) -> Result<(), ModelError> {
                self.inner.validate_input(ids)
            }
            async fn next_tokens(
                &self,
                batch: &mut [ModelInput<'_, NaiveKVPool>],
                policy: Policy,
            ) -> Vec<Result<u32, ModelError>> {
                if batch.iter().any(|input| input.token_ids.len() > 2) {
                    assert_eq!(batch.len(), 1);
                    self.entered.notify_one();
                    self.resume.notified().await;
                }
                self.inner.next_tokens(batch, policy).await
            }
        }
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let entered = Arc::new(tokio::sync::Notify::new());
        let resume = Arc::new(tokio::sync::Notify::new());
        let model = PausedModel {
            inner: SimpleMLPEngine::random(0).await?,
            entered: entered.clone(),
            resume: resume.clone(),
        };
        let policy = global_policy(0)?;
        let pool = NaiveKVPool::new(NaiveKVPoolConfig::default(), &policy).await?;
        let initial = pool.metrics();
        let owned_pool = pool.clone();
        let cancel = CancellationToken::new();
        let (commands, rx) = mpsc::unbounded_channel();
        let (replies, reply_rx) = mpsc::unbounded_channel();
        let scheduler = crate::scheduler::Scheduler::new(cancel.child_token(), commands, reply_rx);
        let runner = RunnerBuilder::default()
            .with_model_factory(move || async move { Ok(model) })
            .with_pool_factory(move |_| async move { Ok(owned_pool) })
            .with_policy_factory(move || Ok(policy))
            .build(rx, replies)
            .await?;
        let task = tokio::spawn(runner.run(cancel.child_token()));
        let result = tokio::time::timeout(Duration::from_secs(30), async {
            let mut request = SessionRequest::new(None, "1 2".into());
            request.sampling = Some(SamplingConfig {
                max_new_tokens: Some(1),
                ..Default::default()
            });
            let short = scheduler.execute_stream(request.clone())?;
            request.sampling.as_mut().unwrap().max_new_tokens = Some(2);
            let long = scheduler.execute_stream(request.clone())?;
            request.sampling.as_mut().unwrap().eos_token_id =
                Some(crate::runner::loader::EosTokenId::Many((0..256).collect()));
            let eos = scheduler.execute_stream(request.clone())?;
            request.sampling.as_mut().unwrap().max_new_tokens = Some(0);
            let zero = scheduler.execute_stream(request)?;
            let invalid = scheduler.execute_stream(SessionRequest::new(None, String::new()))?;
            futures::pin_mut!(short, long, eos, zero, invalid);
            entered.notified().await;
            assert!(short.try_next().await?.is_some());
            assert!(short.try_next().await?.is_none());
            assert!(eos.try_next().await?.is_some());
            assert!(eos.try_next().await?.is_none());
            assert!(zero.try_next().await?.unwrap().output.is_empty());
            assert!(zero.try_next().await?.is_none());
            assert!(matches!(
                invalid.try_next().await,
                Err(EngineError::InvalidInput(_))
            ));
            assert!(invalid.try_next().await?.is_none());
            assert_eq!(pool.metrics().index.assigned_bytes, pool.max_seq_len() * 4);
            resume.notify_one();
            assert_eq!(long.try_collect::<Vec<_>>().await?.len(), 2);
            Ok::<_, EngineError>(())
        })
        .await;
        cancel.cancel();
        scheduler.wait_stopped().await;
        task.await.unwrap();
        result.expect("stopped requests must finish while the peer is paused")?;
        assert_eq!(pool.metrics(), initial);
        Ok(())
    }

    #[tokio::test]
    async fn streaming_decodes_split_utf8_before_generation_finishes() -> Result<(), EngineError> {
        use futures::TryStreamExt;
        use std::sync::atomic::{AtomicBool, Ordering};
        use tokenizers::{decoders::byte_fallback::ByteFallback, models::wordlevel::WordLevel};

        struct ScriptedModel {
            inner: SimpleMLPEngine,
            tokenizer: Tokenizer,
            entered: Arc<tokio::sync::Notify>,
            resume: Arc<tokio::sync::Notify>,
            paused: AtomicBool,
        }
        impl CausalModel<NaiveKVPool> for ScriptedModel {
            fn tokenizer(&self) -> &Tokenizer {
                &self.tokenizer
            }

            async fn next_tokens(
                &self,
                batch: &mut [ModelInput<'_, NaiveKVPool>],
                policy: Policy,
            ) -> Vec<Result<u32, ModelError>> {
                if batch[0].token_ids.len() == 2 && !self.paused.swap(true, Ordering::Relaxed) {
                    self.entered.notify_one();
                    self.resume.notified().await;
                }
                let mut outputs = self.inner.next_tokens(batch, policy).await;
                for (output, input) in outputs.iter_mut().zip(batch) {
                    if let Ok(token) = output {
                        *token = input.token_ids.len() as u32;
                    }
                }
                outputs
            }
        }
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let vocab = ["prompt", "H", "<0xE4>", "<0xBD>", "<0xA0>", "[UNK]"]
            .into_iter()
            .enumerate()
            .map(|(id, token)| (token.to_owned(), id as u32))
            .collect();
        let mut tokenizer = Tokenizer::new(
            WordLevel::builder()
                .vocab(vocab)
                .unk_token("[UNK]".into())
                .build()
                .unwrap(),
        );
        tokenizer.with_decoder(Some(ByteFallback::new()));
        let entered = Arc::new(tokio::sync::Notify::new());
        let resume = Arc::new(tokio::sync::Notify::new());
        let model = ScriptedModel {
            inner: SimpleMLPEngine::random(0).await?,
            tokenizer,
            entered: entered.clone(),
            resume: resume.clone(),
            paused: AtomicBool::new(false),
        };
        let policy = global_policy(0)?;
        let pool = NaiveKVPool::new(NaiveKVPoolConfig::default(), &policy).await?;
        let initial = pool.metrics();
        let owned_pool = pool.clone();
        let cancel = CancellationToken::new();
        let (commands, rx) = mpsc::unbounded_channel();
        let (replies, reply_rx) = mpsc::unbounded_channel();
        let scheduler = crate::scheduler::Scheduler::new(cancel.child_token(), commands, reply_rx);
        let runner = RunnerBuilder::default()
            .with_model_factory(move || async move { Ok(model) })
            .with_pool_factory(move |_| async move { Ok(owned_pool) })
            .with_policy_factory(move || Ok(policy))
            .build(rx, replies)
            .await?;
        let task = tokio::spawn(runner.run(cancel.child_token()));
        let result = async {
            let mut request = SessionRequest::new(None, "prompt".into());
            request.sampling = Some(SamplingConfig {
                max_new_tokens: Some(4),
                ..Default::default()
            });
            let stream = scheduler.execute_stream(request.clone())?;
            futures::pin_mut!(stream);
            assert_eq!(
                tokio::time::timeout(Duration::from_secs(30), stream.try_next())
                    .await
                    .unwrap()?
                    .unwrap()
                    .output,
                "H"
            );
            entered.notified().await;
            assert!(
                tokio::time::timeout(Duration::from_millis(10), stream.try_next())
                    .await
                    .is_err()
            );
            resume.notify_one();
            assert_eq!(stream.try_next().await?.unwrap().output, "你");
            assert!(stream.try_next().await?.is_none());
            assert_eq!(scheduler.execute(request.clone()).await?.output, "H你");
            request.sampling.as_mut().unwrap().max_new_tokens = Some(3);
            // The library buffers the unfinished character; stopping does not synthesize replacements.
            assert_eq!(scheduler.execute(request.clone()).await?.output, "H");
            request.sampling.as_mut().unwrap().max_new_tokens = Some(0);
            assert!(scheduler.execute(request).await?.output.is_empty());
            Ok::<_, EngineError>(())
        }
        .await;
        cancel.cancel();
        scheduler.wait_stopped().await;
        task.await.unwrap();
        result?;
        assert_eq!(pool.metrics(), initial);
        Ok(())
    }

    #[tokio::test]
    async fn greedy_turns_keep_generated_history_and_honor_eos_and_budget(
    ) -> Result<(), EngineError> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let runner = runner().await?;
        let initial = runner.pool.metrics();
        let first = runner.execute_session(request(1, None, "1 2")).await?;
        let eos = first.output.parse::<u32>().unwrap();
        runner.open_session(10)?;
        runner.open_session(20)?;
        let mut longer = request(2, Some(10), "1 2");
        longer.sampling = Some(SamplingConfig {
            max_new_tokens: Some(3),
            ..Default::default()
        });
        let mut stopped = request(3, Some(20), "1 2");
        stopped.sampling = Some(SamplingConfig {
            max_new_tokens: Some(4),
            eos_token_id: Some(crate::runner::loader::EosTokenId::Many(vec![eos])),
            ..Default::default()
        });
        let mut oversized = request(4, None, "1");
        oversized.sampling = Some(SamplingConfig {
            max_new_tokens: Some(usize::MAX),
            ..Default::default()
        });
        let reply = runner
            .execute_collect(vec![longer, stopped, oversized])
            .await;
        assert!(reply[0].1.is_ok() && reply[1].1.is_ok());
        assert!(matches!(reply[2].1, Err(EngineError::InvalidInput(_))));
        for (id, expected_len, index) in [(10, 5, 0), (20, 3, 1)] {
            let entry = runner.sessions.get(&id).unwrap().value().clone();
            let guard = entry.lock().await;
            let state = guard.as_ref().unwrap();
            assert_eq!(state.tokens.len(), expected_len);
            assert_eq!(state.kv.ready_len() + 1, expected_len);
            assert_eq!(
                runner
                    .model
                    .tokenizer()
                    .decode(&state.tokens[2..], true)
                    .unwrap(),
                reply[index].1.as_ref().unwrap().output
            );
        }
        let mut zero = request(5, Some(20), "8");
        zero.sampling = Some(SamplingConfig {
            max_new_tokens: Some(0),
            ..Default::default()
        });
        assert!(runner.execute_session(zero).await?.output.is_empty());
        let mut resumed = request(6, Some(20), "9");
        resumed.sampling = Some(SamplingConfig {
            max_new_tokens: Some(1),
            ..Default::default()
        });
        runner.execute_session(resumed).await?;
        let entry = runner.sessions.get(&20).unwrap().value().clone();
        assert_eq!(entry.lock().await.as_ref().unwrap().kv.ready_len(), 5);
        runner.close_session(10).await?;
        runner.close_session(20).await?;
        assert_eq!(runner.pool.metrics(), initial);
        Ok(())
    }

    fn request(request_id: u64, session_id: Option<u64>, input: &str) -> SessionRequest {
        SessionRequest::new(session_id, input.to_owned()).with_request_id(request_id)
    }

    async fn runner() -> Result<ModelRunner, EngineError> {
        let (_tx, rx) = mpsc::unbounded_channel();
        let (tx, _rx) = mpsc::unbounded_channel();
        let build = RunnerBuilder::default()
            .with_model_factory(|| async {
                let model = SimpleMLPEngine::random(0).await?;
                Ok(model)
            })
            .with_pool_factory(|policy| async move {
                Ok(NaiveKVPool::new(
                    NaiveKVPoolConfig {
                        max_rows: 8,
                        num_slots: 64,
                        max_seq_len: 16,
                        ..Default::default()
                    },
                    &policy,
                )
                .await?)
            })
            .build(rx, tx);
        // Guard against losing Send when changing factory/future abstractions.
        fn require_send<T: Send>(value: T) -> T {
            value
        }
        require_send(build).await
    }

    #[tokio::test]
    async fn invalid_requests_do_not_fail_valid_peers() -> Result<(), EngineError> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let runner = runner().await?;
        let initial = runner.pool.metrics();
        runner.open_session(10)?;
        let first = runner.execute_session(request(1, Some(10), "1 2")).await?;
        let retained = runner.pool.metrics();
        let mut unsupported = request(16, None, "1");
        unsupported.sampling = Some(SamplingConfig {
            do_sample: Some(true),
            ..Default::default()
        });
        let oversized = vec!["1"; 15].join(" ");
        let reply = runner
            .execute_collect(vec![
                request(10, Some(10), &oversized),
                request(11, None, "unknown"),
                request(12, None, ""),
                request(13, Some(999), "1"),
                request(14, None, "3 4"),
                request(15, None, "5"),
                unsupported,
            ])
            .await;
        assert_eq!(
            reply.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![10, 11, 12, 13, 14, 15, 16]
        );
        for (index, (_, result)) in reply.iter().enumerate() {
            if [4, 5].contains(&index) {
                assert!(result
                    .as_ref()
                    .unwrap()
                    .output
                    .parse::<f32>()
                    .unwrap()
                    .is_finite());
            } else {
                if index == 1 {
                    assert!(matches!(
                        result,
                        Err(EngineError::Model(ModelError::InvalidInput(_)))
                    ));
                } else {
                    assert!(matches!(result, Err(EngineError::InvalidInput(_))));
                }
            }
        }
        let entry = runner.sessions.get(&10).unwrap().value().clone();
        assert_eq!(
            entry.lock().await.as_ref().unwrap().tokens.as_slice(),
            &[1, 2, first.output.parse::<u32>().unwrap()]
        );
        assert_eq!(runner.pool.metrics(), retained);
        runner.close_session(10).await?;
        assert_eq!(runner.pool.metrics(), initial);
        Ok(())
    }

    #[tokio::test]
    async fn runner_splits_candidates_by_token_and_context_capacity() -> Result<(), EngineError> {
        struct RecordingModel {
            inner: SimpleMLPEngine,
            batches: Arc<std::sync::Mutex<Vec<usize>>>,
        }
        impl CausalModel<NaiveKVPool> for RecordingModel {
            fn tokenizer(&self) -> &Tokenizer {
                self.inner.tokenizer()
            }

            fn validate_input(&self, tokens: &[u32]) -> Result<(), ModelError> {
                self.inner.validate_input(tokens)
            }
            async fn next_tokens(
                &self,
                batch: &mut [ModelInput<'_, NaiveKVPool>],
                policy: Policy,
            ) -> Vec<Result<u32, ModelError>> {
                self.batches.lock().unwrap().push(batch.len());
                self.inner.next_tokens(batch, policy).await
            }
        }
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        for (num_slots, max_rows, expected_batches) in [
            (3, 4, vec![1, 1, 1]),
            (16, 1, vec![1, 1, 1]),
            (5, 4, vec![2, 1]),
        ] {
            let policy = global_policy(0)?;
            let pool = NaiveKVPool::new(
                NaiveKVPoolConfig {
                    num_slots,
                    max_rows,
                    max_seq_len: 8,
                    ..Default::default()
                },
                &policy,
            )
            .await?;
            let initial = pool.metrics();
            let batches = Arc::new(std::sync::Mutex::new(Vec::new()));
            let model = RecordingModel {
                inner: SimpleMLPEngine::random(0).await?,
                batches: batches.clone(),
            };
            let (_tx, rx) = mpsc::unbounded_channel();
            let (tx, _rx) = mpsc::unbounded_channel();
            let runner = ModelRunner::new(rx, tx, model, pool.clone(), policy);
            let reply = runner
                .execute_collect(vec![
                    request(1, None, "1 2"),
                    request(2, None, "3 4"),
                    request(3, None, "5 6"),
                ])
                .await;
            assert!(reply.iter().all(|(_, result)| result.is_ok()));
            assert_eq!(*batches.lock().unwrap(), expected_batches);
            assert_eq!(pool.metrics(), initial);
            if num_slots == 3 {
                let reply = runner
                    .execute_collect(vec![request(4, None, "1 2 3 4"), request(5, None, "8 9")])
                    .await;
                assert!(matches!(
                    reply[0].1,
                    Err(EngineError::KVPool(KVPoolError::CapacityExhausted(_)))
                ));
                assert!(reply[1].1.is_ok());
                assert_eq!(pool.metrics(), initial);
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn stream_failure_is_routed_per_request_and_sessions_can_retry() -> Result<(), EngineError>
    {
        struct FailedPolicy;
        impl SchedulingPolicy for FailedPolicy {
            fn next_stream(
                &self,
            ) -> Result<Arc<cutile::cuda_core::Stream>, cuda_async::error::DeviceError>
            {
                Err(cuda_async::error::DeviceError::Scheduling(
                    "injected stream failure".into(),
                ))
            }
        }
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let good_policy = global_policy(0)?;
        let pool = NaiveKVPool::new(NaiveKVPoolConfig::default(), &good_policy).await?;
        let initial = pool.metrics();
        let model = SimpleMLPEngine::random(0).await?;
        let (_tx, rx) = mpsc::unbounded_channel();
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut runner = ModelRunner::new(rx, tx, model, pool.clone(), Arc::new(FailedPolicy));
        runner.open_session(10)?;
        runner.open_session(20)?;
        let idle = pool.metrics();
        let reply = runner
            .execute_collect(vec![request(1, Some(10), "1 2"), request(2, Some(20), "8")])
            .await;
        for (_, result) in reply {
            assert!(
                matches!(&result, Err(EngineError::Model(ModelError::Execution(message))) if message.contains("injected stream failure"))
            );
        }
        assert_eq!(pool.metrics(), idle);
        runner.policy = good_policy;
        let reply = runner
            .execute_collect(vec![request(3, Some(10), "1 2"), request(4, Some(20), "8")])
            .await;
        assert!(reply.iter().all(|(_, result)| result.is_ok()));
        runner.close_session(10).await?;
        runner.close_session(20).await?;
        assert_eq!(pool.metrics(), initial);
        Ok(())
    }

    #[tokio::test]
    async fn cancellation_releases_local_state_without_a_turn_guard() -> Result<(), EngineError> {
        struct PausedModel {
            inner: SimpleMLPEngine,
            entered: Arc<tokio::sync::Notify>,
        }
        impl CausalModel<NaiveKVPool> for PausedModel {
            fn tokenizer(&self) -> &Tokenizer {
                self.inner.tokenizer()
            }

            async fn next_tokens(
                &self,
                _batch: &mut [ModelInput<'_, NaiveKVPool>],
                _policy: Policy,
            ) -> Vec<Result<u32, ModelError>> {
                self.entered.notify_one();
                std::future::pending::<()>().await;
                Vec::new()
            }
        }
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = global_policy(0)?;
        let model = PausedModel {
            inner: SimpleMLPEngine::random(0).await?,
            entered: Arc::new(tokio::sync::Notify::new()),
        };
        let pool = NaiveKVPool::new(NaiveKVPoolConfig::default(), &policy).await?;
        let initial = pool.metrics();
        let (commands, rx) = mpsc::unbounded_channel();
        let (tx, _rx) = mpsc::unbounded_channel();
        let runner = ModelRunner::new(rx, tx, model, pool.clone(), policy);
        runner.open_session(10)?;
        tokio::time::timeout(Duration::from_secs(30), async {
            let execution = runner.execute_collect(vec![request(1, Some(10), "1 2")]);
            tokio::pin!(execution);
            tokio::select! {
                _ = runner.model.entered.notified() => {},
                _ = &mut execution => panic!("paused model must remain in flight"),
            }
            // Dropping execution cancels the request after its session state was taken.
        })
        .await
        .expect("model should enter execution");
        let entry = runner.sessions.get(&10).unwrap().value().clone();
        assert!(entry.lock().await.is_none());
        assert_eq!(pool.metrics(), initial);
        runner.close_session(10).await?;
        assert!(!runner.sessions.contains_key(&10));
        runner.open_session(20)?;
        let entered = runner.model.entered.clone();
        let cancel = CancellationToken::new();
        commands
            .send(RunnerCommand::Execute(vec![request(2, Some(20), "3")]))
            .unwrap();
        let task = tokio::spawn(runner.run(cancel.child_token()));
        tokio::time::timeout(Duration::from_secs(30), entered.notified())
            .await
            .expect("Runner enters its model step");
        cancel.cancel();
        tokio::time::timeout(Duration::from_secs(30), task)
            .await
            .expect("Runner cancellation interrupts the model step")
            .expect("Runner task succeeds");
        assert_eq!(pool.metrics(), initial);
        Ok(())
    }

    #[tokio::test]
    async fn mixed_batch_sessions_and_cleanup() -> Result<(), EngineError> {
        let _guard = crate::runner::GPU_TEST_LOCK.lock().await;
        let runner = runner().await?;
        let initial = runner.pool.metrics();
        runner.open_session(10)?;
        runner.open_session(20)?;
        assert!(runner.open_session(10).is_err());
        let reply = runner
            .execute_collect(vec![request(2, Some(10), "1 2"), request(3, Some(20), "8")])
            .await;
        let outputs = reply
            .into_iter()
            .map(|(_, result)| result)
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(
            outputs.iter().map(|out| out.request_id).collect::<Vec<_>>(),
            vec![2, 3]
        );
        let first_token = outputs[0].output.parse::<u32>().unwrap();
        let reference = runner
            .execute_session(request(1, None, &format!("1 2 {first_token} 3 4 5")))
            .await?;
        let outputs = runner
            .execute_collect(vec![request(4, Some(10), "3 4 5"), request(5, None, "9 2")])
            .await
            .into_iter()
            .map(|(_, result)| result)
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(
            outputs.iter().map(|out| out.request_id).collect::<Vec<_>>(),
            vec![4, 5]
        );
        assert_eq!(outputs[0].output, reference.output);
        let entry = runner.sessions.get(&10).unwrap().value().clone();
        {
            let guard = entry.lock().await;
            let state = guard.as_ref().unwrap();
            assert_eq!(
                state.tokens.as_slice(),
                &[
                    1,
                    2,
                    first_token,
                    3,
                    4,
                    5,
                    outputs[0].output.parse::<u32>().unwrap()
                ]
            );
            assert_eq!(state.kv.ready_len(), 6);
        }
        assert!(runner
            .execute_session(request(6, Some(10), "unknown"))
            .await
            .is_err());
        assert_eq!(entry.lock().await.as_ref().unwrap().tokens.len(), 7);
        let mut unsupported = request(7, Some(10), "3");
        unsupported.sampling = Some(SamplingConfig {
            do_sample: Some(true),
            ..Default::default()
        });
        assert!(runner.execute_session(unsupported).await.is_err());
        assert_eq!(entry.lock().await.as_ref().unwrap().tokens.len(), 7);
        runner.execute_session(request(8, Some(10), "9")).await?;
        assert!(runner.execute_collect(vec![]).await.is_empty());
        runner.close_session(10).await?;
        runner.close_session(20).await?;
        assert!(runner.close_session(10).await.is_err());
        assert_eq!(runner.pool.metrics(), initial);
        Ok(())
    }

    #[tokio::test]
    async fn blocked_session_does_not_block_another_session() -> Result<(), EngineError> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let runner = runner().await?;
        runner.open_session(10)?;
        runner.open_session(20)?;
        let entry = runner.sessions.get(&10).unwrap().value().clone();
        let held = entry.lock_owned().await;
        let (blocked, independent) =
            tokio::join!(runner.execute_session(request(1, Some(10), "1 2")), async {
                let result = tokio::time::timeout(
                    Duration::from_secs(30),
                    runner.execute_session(request(2, Some(20), "8 9 7")),
                )
                .await;
                drop(held);
                result
            });
        blocked?;
        independent.expect("an unrelated session must not wait for session 10")?;
        // Same-session concurrent turns must preserve both deltas with no lost updates.
        let (a, b) = tokio::join!(
            runner.execute_session(request(3, Some(10), "3")),
            runner.execute_session(request(4, Some(10), "4"))
        );
        let a_token = a?.output.parse::<u32>().unwrap();
        let b_token = b?.output.parse::<u32>().unwrap();
        let entry = runner.sessions.get(&10).unwrap().value().clone();
        let guard = entry.lock().await;
        let state = guard.as_ref().unwrap();
        assert_eq!(state.kv.ready_len(), 6);
        assert_eq!(state.tokens.len(), 7);
        assert_eq!(&state.tokens[..2], &[1, 2]);
        let mut prompts = vec![state.tokens[3], state.tokens[5]];
        prompts.sort();
        assert_eq!(prompts, vec![3, 4]);
        let mut generated = vec![state.tokens[4], state.tokens[6]];
        generated.sort();
        let mut expected = vec![a_token, b_token];
        expected.sort();
        assert_eq!(generated, expected);
        drop(guard);
        runner.close_session(10).await?;
        runner.close_session(20).await?;
        assert_eq!(runner.pool.metrics().layers.assigned_bytes, 0);
        Ok(())
    }

    #[tokio::test]
    async fn serial_batches_preserve_input_order() -> Result<(), EngineError> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let runner = runner().await?;
        runner.open_session(10)?;
        runner.open_session(20)?;
        let first = runner
            .execute_collect(vec![request(1, Some(10), "1"), request(2, Some(20), "2")])
            .await
            .into_iter()
            .map(|(_, result)| result)
            .collect::<Result<Vec<_>, _>>()?;
        let second = runner
            .execute_collect(vec![request(3, Some(20), "3"), request(4, Some(10), "4")])
            .await
            .into_iter()
            .map(|(_, result)| result)
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(
            first
                .iter()
                .map(|output| output.request_id)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(
            second
                .iter()
                .map(|output| output.request_id)
                .collect::<Vec<_>>(),
            vec![3, 4]
        );
        for id in [10, 20] {
            let entry = runner.sessions.get(&id).unwrap().value().clone();
            assert_eq!(entry.lock().await.as_ref().unwrap().kv.ready_len(), 3);
            runner.close_session(id).await?;
        }
        assert_eq!(runner.pool.metrics().layers.assigned_bytes, 0);
        Ok(())
    }

    #[tokio::test]
    async fn scheduler_preserves_session_request_envelope() -> Result<(), EngineError> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let cancel = CancellationToken::new();
        let (request_tx, request_rx) = mpsc::unbounded_channel();
        let (reply_tx, reply_rx) = mpsc::unbounded_channel();
        let runner = RunnerBuilder::default()
            .with_model_factory(|| async {
                let model = SimpleMLPEngine::random(0).await?;
                Ok(model)
            })
            .with_pool_factory(|policy| async move {
                Ok(NaiveKVPool::new(
                    NaiveKVPoolConfig {
                        max_rows: 2,
                        ..Default::default()
                    },
                    &policy,
                )
                .await?)
            })
            .build(request_rx, reply_tx)
            .await?;
        let pool = runner.pool.clone();
        let initial = pool.metrics();
        let scheduler =
            crate::scheduler::Scheduler::new(cancel.child_token(), request_tx, reply_rx);
        let task = tokio::spawn(runner.run(cancel.child_token()));
        let session_id = scheduler.open_session().await?;
        let second = scheduler.clone().open_session().await?;
        assert_ne!(session_id, second);
        assert!(matches!(
            scheduler.open_session().await,
            Err(EngineError::KVContext(
                crate::runner::kv_pool::KVContextError::Pool(KVPoolError::CapacityExhausted(_))
            ))
        ));
        scheduler.close_session(second).await?;
        let (a, b, stateless) = tokio::join!(
            scheduler.execute(SessionRequest::new(Some(session_id), "1 2".into())),
            scheduler.execute(SessionRequest::new(Some(session_id), "3".into())),
            scheduler.execute(SessionRequest::new(None, "8 9".into()))
        );
        let a = a?;
        let b = b?;
        let next = scheduler
            .execute(SessionRequest::new(Some(session_id), "4 5".into()))
            .await;
        let reference = scheduler
            .execute(SessionRequest::new(
                None,
                format!("1 2 {} 3 {} 4 5", a.output, b.output),
            ))
            .await;
        assert!(pool.metrics().layers.assigned_bytes > 0);
        scheduler.close_session(session_id).await?;
        assert_eq!(pool.metrics(), initial);
        assert!(matches!(
            scheduler
                .execute(SessionRequest::new(Some(session_id), "6".into()))
                .await,
            Err(EngineError::InvalidInput(_))
        ));
        assert!(scheduler.close_session(session_id).await.is_err());
        cancel.cancel();
        scheduler.wait_stopped().await;
        task.await
            .map_err(|error| EngineError::Internal(error.to_string()))?;
        assert_ne!(a.request_id, b.request_id);
        stateless?;
        assert_eq!(next?.output, reference?.output);
        assert_eq!(pool.metrics().layers.assigned_bytes, 0);
        Ok(())
    }
}
