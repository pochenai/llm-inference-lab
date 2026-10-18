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
        model::{Model, ModelError, ModelInput, ModelOutput, SimpleMLPEngine},
    },
    types::{
        BatchIn, BatchReply, LLMResponse, RunnerCommand, SamplingConfig, SessionId, SessionRequest,
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
        batch_reply_tx: mpsc::UnboundedSender<BatchReply>,
    ) -> impl Future<Output = Result<ModelRunner<M, P>, EngineError>> + Send
    where
        P: KVPool,
        M: Model<P>,
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
    batch_reply_tx: mpsc::UnboundedSender<BatchReply>,
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

impl<M: Model<P>, P: KVPool> ModelRunner<M, P> {
    fn new(
        command_rx: mpsc::UnboundedReceiver<RunnerCommand>,
        batch_reply_tx: mpsc::UnboundedSender<BatchReply>,
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

    /// Test convenience wrapper; production ingress always uses Scheduler batches.
    #[cfg(test)]
    async fn execute_session(&self, request: SessionRequest) -> Result<LLMResponse, EngineError> {
        self.execute_batch(vec![request])
            .await
            .results
            .pop()
            .unwrap()
            .1
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

    fn decode(&self, item: ModelOutput) -> Result<LLMResponse, EngineError> {
        let output = self
            .model
            .tokenizer()
            .decode(&item.result?, true)
            .map_err(|e| ModelError::Execution(e.to_string()))?;
        Ok(LLMResponse {
            request_id: item.request_id,
            output,
        })
    }

    /// Execute one admitted batch, storing one result per original request index.
    /// Cancellation drops local states; successful states are restored before decoding.
    async fn execute_model_batch(
        &self,
        encoded: Vec<(usize, SessionRequest<Encoding>)>,
        mut guards: Vec<OwnedMutexGuard<Option<SessionState<P>>>>,
        responses: &mut [Option<Result<LLMResponse, EngineError>>],
    ) {
        if encoded.is_empty() {
            return;
        }
        let mut states: Vec<_> = guards
            .iter_mut()
            .map(|guard| guard.take().unwrap())
            .collect();
        let original_progress: Vec<_> = states
            .iter()
            .map(|state| (state.tokens.len(), state.kv.ready_len()))
            .collect();
        let inputs = states
            .iter_mut()
            .zip(&encoded)
            .map(|(state, (_, encoded))| {
                Arc::make_mut(&mut state.tokens).extend_from_slice(encoded.input.get_ids());
                let mut request = SessionRequest::new(
                    encoded.session_id,
                    ModelInput {
                        token_ids: &state.tokens,
                        kv: &mut state.kv,
                    },
                )
                .with_request_id(encoded.assigned_request_id());
                request.sampling = encoded.sampling.clone();
                request
            })
            .collect();
        let outputs = self.model.generate(inputs, self.policy.clone()).await;
        for (index, (mut state, output)) in states.into_iter().zip(outputs).enumerate() {
            let (old_len, old_ready) = original_progress[index];
            if output.result.is_ok() {
                state.sampling = encoded[index].1.sampling.as_ref().unwrap().clone();
                *guards[index] = Some(state);
            } else if !state.kv.is_pending() && state.kv.ready_len() == old_ready {
                Arc::make_mut(&mut state.tokens).truncate(old_len);
                *guards[index] = Some(state);
            }
            responses[encoded[index].0] = Some(self.decode(output));
        }
        // Guards release persistent sessions; local temporary entries release their KV.
    }

    /// Scheduler supplies candidates with unique Some(session_id). Runner validates each
    /// request, then splits execution batches by local free KV tokens and context rows.
    /// run is serial; concurrent multi-batch execution would need consistent lock order.
    async fn execute_batch(&self, batch: BatchIn) -> BatchReply {
        let request_ids: Vec<_> = batch
            .iter()
            .map(SessionRequest::assigned_request_id)
            .collect();
        let mut responses: Vec<_> = (0..batch.len()).map(|_| None).collect();
        let mut encoded = Vec::new();
        let mut guards = Vec::new();
        let mut append_tokens = 0usize;
        for (index, request) in batch.into_iter().enumerate() {
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
                            .ok_or_else(|| EngineError::InvalidInput("unknown session".into()))?;
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
                self.model.validate_input(tokens.get_ids(), &sampling)?;
                let mut encoded = SessionRequest::new(request.session_id, tokens)
                    .with_request_id(request.assigned_request_id());
                encoded.sampling = Some(sampling);
                Ok((encoded, guard, full_len - ready_len))
            }
            .await;
            let (request, guard, needed) = match prepared {
                Ok(prepared) => prepared,
                Err(error) => {
                    responses[index] = Some(Err(error));
                    continue;
                }
            };
            let new_contexts = usize::from(guard.is_none());
            if !self
                .pool
                .can_append(append_tokens.saturating_add(needed), new_contexts)
                && !encoded.is_empty()
            {
                self.execute_model_batch(
                    std::mem::take(&mut encoded),
                    std::mem::take(&mut guards),
                    &mut responses,
                )
                .await;
                append_tokens = 0;
            }
            if !self
                .pool
                .can_append(append_tokens.saturating_add(needed), new_contexts)
            {
                responses[index] = Some(Err(KVPoolError::CapacityExhausted(
                    "insufficient KV capacity for request".into(),
                )
                .into()));
                continue;
            }
            let guard = match guard {
                Some(guard) => guard,
                None => {
                    let kv = match KVContext::new(self.pool.clone()) {
                        Ok(kv) => kv,
                        Err(error) => {
                            responses[index] = Some(Err(error.into()));
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
            encoded.push((index, request));
            guards.push(guard);
        }
        self.execute_model_batch(encoded, guards, &mut responses)
            .await;
        BatchReply {
            results: request_ids
                .into_iter()
                .zip(responses)
                .map(|(id, response)| {
                    (
                        id,
                        response.expect("Runner completes every admitted request"),
                    )
                })
                .collect(),
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
                    let output = self.execute_batch(batch).await;
                    if self.batch_reply_tx.send(output).is_err() {
                        break;
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
    use std::time::Duration;

    fn request(request_id: u64, session_id: Option<u64>, input: &str) -> SessionRequest {
        SessionRequest::new(session_id, input.to_owned()).with_request_id(request_id)
    }

    async fn runner() -> Result<ModelRunner, EngineError> {
        let (_tx, rx) = mpsc::unbounded_channel();
        let (tx, _rx) = mpsc::unbounded_channel();
        let build = RunnerBuilder::default()
            .with_model_factory(|| async { Ok(SimpleMLPEngine::random(0).await?) })
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
        runner.execute_session(request(1, Some(10), "1 2")).await?;
        let retained = runner.pool.metrics();
        let mut unsupported = request(16, None, "1");
        unsupported.sampling = Some(SamplingConfig {
            do_sample: Some(true),
            ..Default::default()
        });
        let oversized = vec!["1"; 15].join(" ");
        let reply = runner
            .execute_batch(vec![
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
            reply.results.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![10, 11, 12, 13, 14, 15, 16]
        );
        for (index, (_, result)) in reply.results.iter().enumerate() {
            if [4, 5].contains(&index) {
                assert!(result
                    .as_ref()
                    .unwrap()
                    .output
                    .parse::<f32>()
                    .unwrap()
                    .is_finite());
            } else {
                if [1, 6].contains(&index) {
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
            &[1, 2]
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
        impl Model<NaiveKVPool> for RecordingModel {
            fn tokenizer(&self) -> &tokenizers::Tokenizer {
                self.inner.tokenizer()
            }
            fn validate_input(
                &self,
                tokens: &[u32],
                sampling: &SamplingConfig,
            ) -> Result<(), ModelError> {
                self.inner.validate_input(tokens, sampling)
            }
            async fn generate(
                &self,
                batch: BatchIn<ModelInput<'_, NaiveKVPool>>,
                policy: Policy,
            ) -> Vec<ModelOutput> {
                self.batches.lock().unwrap().push(batch.len());
                self.inner.generate(batch, policy).await
            }
        }
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        for (num_slots, max_rows) in [(3, 4), (16, 1)] {
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
                .execute_batch(vec![
                    request(1, None, "1 2"),
                    request(2, None, "3 4"),
                    request(3, None, "5 6"),
                ])
                .await;
            assert!(reply.results.iter().all(|(_, result)| result.is_ok()));
            assert_eq!(*batches.lock().unwrap(), vec![1, 1, 1]);
            assert_eq!(pool.metrics(), initial);
            if num_slots == 3 {
                let reply = runner
                    .execute_batch(vec![request(4, None, "1 2 3 4"), request(5, None, "8 9")])
                    .await;
                assert!(matches!(
                    reply.results[0].1,
                    Err(EngineError::KVPool(KVPoolError::CapacityExhausted(_)))
                ));
                assert!(reply.results[1].1.is_ok());
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
            .execute_batch(vec![request(1, Some(10), "1 2"), request(2, Some(20), "8")])
            .await;
        for (_, result) in reply.results {
            assert!(
                matches!(&result, Err(EngineError::Model(ModelError::Execution(message))) if message.contains("injected stream failure"))
            );
        }
        assert_eq!(pool.metrics(), idle);
        runner.policy = good_policy;
        let reply = runner
            .execute_batch(vec![request(3, Some(10), "1 2"), request(4, Some(20), "8")])
            .await;
        assert!(reply.results.iter().all(|(_, result)| result.is_ok()));
        runner.close_session(10).await?;
        runner.close_session(20).await?;
        assert_eq!(pool.metrics(), initial);
        Ok(())
    }

    #[tokio::test]
    async fn cancellation_releases_local_state_without_a_turn_guard() -> Result<(), EngineError> {
        struct PausedModel {
            inner: SimpleMLPEngine,
            entered: tokio::sync::Notify,
        }
        impl Model<NaiveKVPool> for PausedModel {
            fn tokenizer(&self) -> &tokenizers::Tokenizer {
                self.inner.tokenizer()
            }
            async fn generate(
                &self,
                batch: BatchIn<ModelInput<'_, NaiveKVPool>>,
                _policy: Policy,
            ) -> Vec<ModelOutput> {
                self.entered.notify_one();
                std::future::pending::<()>().await;
                drop(batch);
                Vec::new()
            }
        }
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = global_policy(0)?;
        let model = PausedModel {
            inner: SimpleMLPEngine::random(0).await?,
            entered: tokio::sync::Notify::new(),
        };
        let pool = NaiveKVPool::new(NaiveKVPoolConfig::default(), &policy).await?;
        let initial = pool.metrics();
        let (_tx, rx) = mpsc::unbounded_channel();
        let (tx, _rx) = mpsc::unbounded_channel();
        let runner = ModelRunner::new(rx, tx, model, pool.clone(), policy);
        runner.open_session(10)?;
        tokio::time::timeout(Duration::from_secs(30), async {
            let execution = runner.execute_batch(vec![request(1, Some(10), "1 2")]);
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
        let reference = runner
            .execute_session(request(1, None, "1 2 3 4 5"))
            .await?;
        let reply = runner
            .execute_batch(vec![request(2, Some(10), "1 2"), request(3, Some(20), "8")])
            .await;
        let outputs = reply
            .results
            .into_iter()
            .map(|(_, result)| result)
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(
            outputs.iter().map(|out| out.request_id).collect::<Vec<_>>(),
            vec![2, 3]
        );
        let outputs = runner
            .execute_batch(vec![request(4, Some(10), "3 4 5"), request(5, None, "9 2")])
            .await
            .results
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
            assert_eq!(state.tokens.as_slice(), &[1, 2, 3, 4, 5]);
            assert_eq!(state.kv.ready_len(), 5);
        }
        assert!(runner
            .execute_session(request(6, Some(10), "unknown"))
            .await
            .is_err());
        assert_eq!(entry.lock().await.as_ref().unwrap().tokens.len(), 5);
        let mut unsupported = request(7, Some(10), "3");
        unsupported.sampling = Some(SamplingConfig {
            do_sample: Some(true),
            ..Default::default()
        });
        assert!(runner.execute_session(unsupported).await.is_err());
        assert_eq!(entry.lock().await.as_ref().unwrap().tokens.len(), 5);
        runner.execute_session(request(8, Some(10), "9")).await?;
        assert!(runner.execute_batch(vec![]).await.results.is_empty());
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
        a?;
        b?;
        let entry = runner.sessions.get(&10).unwrap().value().clone();
        let guard = entry.lock().await;
        let state = guard.as_ref().unwrap();
        assert_eq!(state.kv.ready_len(), 4);
        assert_eq!(&state.tokens[..2], &[1, 2]);
        let mut suffix = state.tokens[2..].to_vec();
        suffix.sort();
        assert_eq!(suffix, vec![3, 4]);
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
            .execute_batch(vec![request(1, Some(10), "1"), request(2, Some(20), "2")])
            .await
            .results
            .into_iter()
            .map(|(_, result)| result)
            .collect::<Result<Vec<_>, _>>()?;
        let second = runner
            .execute_batch(vec![request(3, Some(20), "3"), request(4, Some(10), "4")])
            .await
            .results
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
            assert_eq!(entry.lock().await.as_ref().unwrap().kv.ready_len(), 2);
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
            .with_model_factory(|| async { Ok(SimpleMLPEngine::random(0).await?) })
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
        let next = scheduler
            .execute(SessionRequest::new(Some(session_id), "4 5".into()))
            .await;
        let reference = scheduler
            .execute(SessionRequest::new(None, "1 2 3 4 5".into()))
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
        assert_ne!(a?.request_id, b?.request_id);
        stateless?;
        assert_eq!(next?.output, reference?.output);
        assert_eq!(pool.metrics().layers.assigned_bytes, 0);
        Ok(())
    }
}
