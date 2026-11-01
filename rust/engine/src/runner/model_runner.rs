//! Each Runner submits complete model forwards on one ordered compute stream.
//! Membership changes between full submissions, never between layers or CTAs.
//!
//! ```text
//! submit F0([A,B]);
//! submit F1([A,B,C]); wait/commit/process F0;
//! submit F2([B,C]);   wait/commit/process F1;
//! ```
//! One-round lookahead overlaps submission/GPU work with the previous result.
//! GPU execution stays ordered; this is not several independent batches racing
//! across compute streams. Pending records retain IDs, predictions and shared GPU completion,
//! not request borrows. KV allocated_len includes reservations; ready_len advances
//! after matching completion. CPU history/EOS/text may lag one prediction.
//! Finished requests drain any submitted extra member before cleanup/EOF.
//! MixedChunkPolicy budgets packed input tokens; intermediate prefill predictions
//! are discarded. Tokens/KV survive steps; hidden intermediates belong to a forward.

mod batch_policy;

use self::batch_policy::{BatchPolicy, MixedChunkPolicy};
use super::model::GpuWorkCompletionFuture;
use super::prefix_cache::{PrefixCache, PrefixCacheConfig};
use std::{collections::VecDeque, future::Future, num::NonZeroUsize, sync::Arc};

use cuda_async::{
    device_context::global_policy,
    scheduling_policies::{SchedulingPolicy, SingleStream},
};
use dashmap::DashMap;
use tokenizers::{Encoding, Tokenizer};
use tokio::sync::{mpsc, Mutex, OwnedMutexGuard};
use tokio_util::sync::CancellationToken;

use crate::{
    error::EngineError,
    runner::{
        kv_pool::{KVContext, KVPool, KVPoolError, NaiveKVPool, NaiveKVPoolConfig},
        model::{
            CausalModel, GpuTokenPredictionHandle, ModelError, ModelInput, ModelTokens, NextToken,
            SimpleMLPEngine, SubmittedGpuWorkWithCompletion,
        },
    },
    types::{
        LLMResponse, RequestId, RunnerCommand, RunnerReply, SamplingConfig, SessionId,
        SessionRequest,
    },
};

type Policy = Arc<dyn SchedulingPolicy + 'static>;
const DEFAULT_MAX_BATCH_TOKENS: NonZeroUsize = NonZeroUsize::new(512).unwrap();
type Decoder<'a> = tokenizers::DecodeStream<
    'a,
    tokenizers::ModelWrapper,
    tokenizers::NormalizerWrapper,
    tokenizers::PreTokenizerWrapper,
    tokenizers::PostProcessorWrapper,
    tokenizers::DecoderWrapper,
>;

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
    max_batch_tokens: NonZeroUsize,
    overlap: bool,
    prefix_cache_config: Option<PrefixCacheConfig>,
}

impl Default for RunnerBuilder {
    fn default() -> Self {
        Self {
            model_factory: DefaultModelFactory,
            pool_factory: DefaultPoolFactory,
            policy_factory: || Ok(global_policy(0)?),
            max_batch_tokens: DEFAULT_MAX_BATCH_TOKENS,
            overlap: true,
            prefix_cache_config: None,
        }
    }
}

impl<MF, PF, SF> RunnerBuilder<MF, PF, SF> {
    pub fn with_model_factory<F>(self, factory: F) -> RunnerBuilder<F, PF, SF> {
        RunnerBuilder {
            model_factory: factory,
            pool_factory: self.pool_factory,
            policy_factory: self.policy_factory,
            max_batch_tokens: self.max_batch_tokens,
            overlap: self.overlap,
            prefix_cache_config: self.prefix_cache_config,
        }
    }

    pub fn with_pool_factory<F>(self, factory: F) -> RunnerBuilder<MF, F, SF> {
        RunnerBuilder {
            model_factory: self.model_factory,
            pool_factory: factory,
            policy_factory: self.policy_factory,
            max_batch_tokens: self.max_batch_tokens,
            overlap: self.overlap,
            prefix_cache_config: self.prefix_cache_config,
        }
    }

    pub fn with_policy_factory<F>(self, factory: F) -> RunnerBuilder<MF, PF, F> {
        RunnerBuilder {
            model_factory: self.model_factory,
            pool_factory: self.pool_factory,
            policy_factory: factory,
            max_batch_tokens: self.max_batch_tokens,
            overlap: self.overlap,
            prefix_cache_config: self.prefix_cache_config,
        }
    }

    /// Limit uncached input tokens per forward, including one token per decode.
    /// MixedChunkPolicy reserves decode tokens and fills the remaining budget with
    /// prefill chunks in active order; the default is 512 tokens.
    pub fn with_max_batch_tokens(mut self, limit: NonZeroUsize) -> Self {
        self.max_batch_tokens = limit;
        self
    }

    /// Enqueue one lookahead forward before settling the previous completion and
    /// CPU result, when the model supports submission-first feedback. Disable for serial.
    pub fn with_overlap(mut self, enabled: bool) -> Self {
        self.overlap = enabled;
        self
    }

    /// Enable cross-request immutable KV reuse; disabled by default for clean benchmarks.
    pub fn with_prefix_cache(mut self, config: PrefixCacheConfig) -> Self {
        self.prefix_cache_config = Some(config);
        self
    }

    pub fn build<M, P>(
        self,
        command_rx: mpsc::UnboundedReceiver<RunnerCommand>,
        batch_reply_tx: mpsc::UnboundedSender<RunnerReply>,
    ) -> impl Future<Output = Result<ModelRunner<M, P>, EngineError>> + Send
    where
        P: KVPool + 'static,
        M: CausalModel<P> + 'static,
        MF: AsyncFactory<(), Output = M>,
        PF: AsyncFactory<Policy, Output = Arc<P>>,
        SF: FnOnce() -> Result<Policy, EngineError> + Send,
    {
        async move {
            let policy = (self.policy_factory)()?;
            let model = self.model_factory.create(()).await?;
            let pool = self.pool_factory.create(policy.clone()).await?;
            let mut runner = ModelRunner::new(command_rx, batch_reply_tx, model, pool, policy);
            runner.max_batch_tokens = self.max_batch_tokens;
            runner.overlap = self.overlap;
            runner.prefix_cache = self
                .prefix_cache_config
                .map(|config| PrefixCache::new(config, runner.pool.clone()));
            Ok(runner)
        }
    }
}

/// Each entry has its own async lock; the directory is never held across GPU awaits.
type SessionEntry<P> = Arc<Mutex<Option<SessionState<P>>>>;

pub struct ModelRunner<M = SimpleMLPEngine, P: KVPool = NaiveKVPool> {
    command_rx: Option<mpsc::UnboundedReceiver<RunnerCommand>>,
    batch_reply_tx: mpsc::UnboundedSender<RunnerReply>,
    model: Arc<M>,
    pool: Arc<P>,
    policy: Policy,
    sessions: DashMap<SessionId, SessionEntry<P>>,
    max_batch_tokens: NonZeroUsize,
    overlap: bool,
    prefix_cache: Option<PrefixCache<P>>,
}

/// CPU history length and confirmed KV progress saved for unchanged-state rollback.
struct SessionProgressBeforeRequest {
    token_count: usize,
    ready_len: usize,
}

struct SessionState<P: KVPool> {
    tokens: Arc<Vec<u32>>,
    kv: KVContext<P>,
    sampling: SamplingConfig,
}

/// Owns one turn across forward boundaries; dropping it invalidates an unfinished
/// persistent turn and releases temporary KV through the existing context leases.
/// State/counters/decoder are used per step; guard/progress/errors serve cleanup.
/// Keep compiler-selected layout: this single-owner record spans multiple cache
/// lines, so fixed alignment needs CPU-profile evidence rather than field ordering.
struct ActiveRequest<'a, P: KVPool> {
    request_id: RequestId,
    guard: OwnedMutexGuard<Option<SessionState<P>>>,
    state: SessionState<P>,
    original_progress: SessionProgressBeforeRequest,
    generation: RequestTokenGenerationState<'a>,
    gpu_token: Option<GpuTokenPredictionHandle>,
}

/// Per-turn sampling budget, decoder and errors, kept separate from session KV ownership.
/// Accepted CPU tokens advance this state only after successful D2H.
struct RequestTokenGenerationState<'a> {
    sampling: SamplingConfig,
    remaining: usize,
    generated_tokens: usize,
    model_error: Option<ModelError>,
    decode_error: Option<ModelError>,
    emitted: bool,
    decoder: Decoder<'a>,
}

impl RequestTokenGenerationState<'_> {
    fn accept(
        &mut self,
        tokens: &mut Arc<Vec<u32>>,
        output: Result<u32, ModelError>,
        request_id: RequestId,
        replies: &mpsc::UnboundedSender<RunnerReply>,
        default_eos: impl FnOnce(u32) -> bool,
    ) {
        // Synchronous CPU history/decoder/EOS work; this range never spans an await.
        let _range = crate::profiling::ScopedRange::new(c"runner.accept_token");
        let token = match output {
            Ok(token) => token,
            Err(error) => {
                self.model_error = Some(error);
                self.remaining = 0;
                return;
            }
        };
        Arc::make_mut(tokens).push(token);
        self.remaining -= 1;
        self.generated_tokens += 1;
        match self.decoder.step(token) {
            Ok(Some(output)) => {
                self.emitted = true;
                nvtx_sys::mark_ascii(c"runner.delta.emit");
                let _ = replies.send(RunnerReply::Delta(LLMResponse { request_id, output }));
            }
            Ok(None) => {}
            Err(error) => {
                self.decode_error = Some(ModelError::Execution(error.to_string()));
                self.remaining = 0;
            }
        }
        let eos = match &self.sampling.eos_token_id {
            Some(crate::runner::loader::EosTokenId::One(id)) => token == *id,
            Some(crate::runner::loader::EosTokenId::Many(ids)) => ids.contains(&token),
            None => default_eos(token),
        };
        if eos {
            self.remaining = 0;
        }
    }
}

impl<P: KVPool> ActiveRequest<'_, P> {
    /// A sampled GPU prediction enters decode before its CPU history is updated.
    fn is_decoding_or_has_gpu_feedback(&self) -> bool {
        self.generation.generated_tokens > 0 || self.gpu_token.is_some()
    }

    /// Whether the latest GPU prediction has not yet been appended to CPU history.
    fn has_pending_gpu_prediction(&self) -> bool {
        self.gpu_token
            .as_ref()
            .is_some_and(|token| token.position >= self.state.tokens.len())
    }

    fn sequence_len(&self) -> usize {
        self.gpu_token
            .as_ref()
            .map_or(self.state.tokens.len(), |token| {
                self.state.tokens.len().max(token.position + 1)
            })
    }

    /// Require uncomputed input and budget beyond any unaccepted GPU prediction.
    /// A final pending prediction is processed without enqueueing another append.
    fn has_input_and_budget_for_forward(&self) -> bool {
        self.generation.remaining > usize::from(self.has_pending_gpu_prediction())
            && self.sequence_len() > self.state.kv.allocated_len()
    }

    /// Account for future appends before admitting peers; physical slots are still
    /// allocated incrementally by prepare_append on each submitted model step.
    fn unreserved_kv_tokens_for_generation(&self) -> usize {
        (self.state.tokens.len() + self.generation.remaining.saturating_sub(1))
            .saturating_sub(self.state.kv.allocated_len())
    }

    fn finish(
        mut self,
        replies: &mpsc::UnboundedSender<RunnerReply>,
        cache: Option<&PrefixCache<P>>,
    ) where
        P: 'static,
    {
        if let Some(cache) = cache.filter(|_| self.generation.model_error.is_none()) {
            if let Err(error) = cache.publish(&self.state.tokens, &mut self.state.kv) {
                // A pre-transfer failure is advisory; invalidated ownership must fail the turn.
                if !self.state.kv.has_owned_pool_state() {
                    self.generation.model_error = Some(error.into());
                } else {
                    tracing::warn!(target: "engine::prefix_cache", %error, "prefix publication skipped");
                }
            }
        }
        let generated = self.generation.model_error.is_none();
        drop(self.generation.decoder);
        let result = match self.generation.model_error.or(self.generation.decode_error) {
            Some(error) => Err(error.into()),
            None => {
                if !self.generation.emitted {
                    nvtx_sys::mark_ascii(c"runner.delta.emit");
                    let _ = replies.send(RunnerReply::Delta(LLMResponse {
                        request_id: self.request_id,
                        output: String::new(),
                    }));
                }
                Ok(())
            }
        };
        if generated {
            self.state.sampling = self.generation.sampling;
            *self.guard = Some(self.state);
        } else if self.state.kv.has_owned_pool_state()
            && !self.state.kv.is_pending()
            && self.state.kv.ready_len() == self.original_progress.ready_len
        {
            Arc::make_mut(&mut self.state.tokens).truncate(self.original_progress.token_count);
            *self.guard = Some(self.state);
        } else {
            drop(self.state);
        }
        // Restore/unlock sessions and release temporary KV before publishing EOF.
        drop(self.guard);
        let _ = replies.send(RunnerReply::Finished {
            request_id: self.request_id,
            generated_tokens: self.generation.generated_tokens,
            result,
        });
    }
}

/// One request's submitted range, prediction and shared batch GPU completion.
/// Keeps request IDs, not ActiveRequest borrows, until KV and token processing finishes.
struct RequestForwardAwaitingGpuCompletion {
    request_id: RequestId,
    compute_end: usize,
    /// Only a final prefill chunk or decode step can advance output history/budget.
    produces_output_token: bool,
    prediction: Result<NextToken, ModelError>,
    gpu_completion: GpuWorkCompletionFuture,
}

impl<M: CausalModel<P> + 'static, P: KVPool + 'static> ModelRunner<M, P> {
    /// Wait for GPU completion, commit each request's FIFO KV append, then accept its token.
    /// Discard intermediate-prefill predictions and drain extra work after EOS/errors.
    async fn complete_forward_and_emit_tokens(
        &self,
        pending_requests: Vec<RequestForwardAwaitingGpuCompletion>,
        active: &mut [ActiveRequest<'_, P>],
        replies: &mpsc::UnboundedSender<RunnerReply>,
    ) {
        for pending in pending_requests {
            let completed = pending.gpu_completion.await;
            let request = active
                .iter_mut()
                .find(|request| request.request_id == pending.request_id)
                .expect("in-flight requests must remain active until drained");
            let result = completed.and(pending.prediction).and_then(|token| {
                if !request.state.kv.has_owned_pool_state() {
                    return Err(ModelError::Execution(
                        "context invalidated by later submission".into(),
                    ));
                }
                // Default completed models already committed; queued Qwen appends commit here.
                if request.state.kv.ready_len() < pending.compute_end {
                    request
                        .state
                        .kv
                        .commit_completed_append(pending.compute_end)?;
                }
                Ok(token)
            });
            let output = match result {
                Ok(_) if !pending.produces_output_token || request.generation.remaining == 0 => {
                    continue;
                }
                Ok(token) => token.read().await,
                Err(error) => Err(error),
            };
            request.generation.accept(
                &mut request.state.tokens,
                output,
                request.request_id,
                replies,
                |id| self.model.is_eos(id),
            );
        }
    }

    fn new(
        command_rx: mpsc::UnboundedReceiver<RunnerCommand>,
        batch_reply_tx: mpsc::UnboundedSender<RunnerReply>,
        model: M,
        pool: Arc<P>,
        policy: Policy,
    ) -> Self {
        Self {
            command_rx: Some(command_rx),
            batch_reply_tx,
            model: Arc::new(model),
            pool,
            policy,
            sessions: DashMap::new(),
            max_batch_tokens: DEFAULT_MAX_BATCH_TOKENS,
            overlap: true,
            prefix_cache: None,
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

    /// Clients close after their turn finishes; reject an active session rather than
    /// waiting inside the step loop. Existing GPU leases still protect released storage.
    fn close_session(&self, id: SessionId) -> Result<(), EngineError> {
        let entry = self
            .sessions
            .get(&id)
            .map(|entry| entry.value().clone())
            .ok_or(EngineError::SessionLost(id))?;
        let mut guard = entry
            .clone()
            .try_lock_owned()
            .map_err(|_| EngineError::InvalidInput("session is busy".into()))?;
        if let Some(state) = guard.take() {
            state.kv.release();
        }
        self.sessions
            .remove_if(&id, |_, current| Arc::ptr_eq(current, &entry));
        Ok(())
    }

    /// Encode once at ingress; waiting requests retain token IDs without repeatedly
    /// tokenizing their prompts while waiting for KV capacity.
    fn enqueue(
        &self,
        command: RunnerCommand,
        waiting: &mut VecDeque<SessionRequest<Encoding>>,
        replies: &mpsc::UnboundedSender<RunnerReply>,
    ) {
        match command {
            RunnerCommand::Execute(batch) => {
                for request in batch {
                    match self.model.tokenizer().encode(request.input.as_str(), true) {
                        Ok(tokens) => {
                            if tokens.is_empty() {
                                let _ = replies.send(RunnerReply::Finished {
                                    request_id: request.assigned_request_id(),
                                    generated_tokens: 0,
                                    result: Err(EngineError::InvalidInput(
                                        "prompt delta must not be empty".into(),
                                    )),
                                });
                                continue;
                            }
                            let mut next = SessionRequest::new(request.session_id, tokens)
                                .with_request_id(request.assigned_request_id());
                            next.sampling = request.sampling;
                            waiting.push_back(next);
                        }
                        Err(error) => {
                            let _ = replies.send(RunnerReply::Finished {
                                request_id: request.assigned_request_id(),
                                generated_tokens: 0,
                                result: Err(EngineError::InvalidInput(error.to_string())),
                            });
                        }
                    }
                }
            }
            RunnerCommand::OpenSession { id, reply } => {
                let _ = reply.send(self.open_session(id));
            }
            RunnerCommand::CloseSession { id, reply } => {
                let _ = reply.send(self.close_session(id));
            }
        }
    }

    /// Clients serialize a session's turns. Reject accidental overlap; only capacity
    /// exhaustion is retried after active work can release resources.
    fn try_admit<'a>(
        &self,
        request: &SessionRequest<Encoding>,
        reserved_tokens: usize,
        tokenizer: &'a Tokenizer,
    ) -> Result<ActiveRequest<'a, P>, EngineError> {
        let guard = match request.session_id {
            Some(id) => {
                let entry = self
                    .sessions
                    .get(&id)
                    .map(|entry| entry.value().clone())
                    .ok_or(EngineError::SessionLost(id))?;
                let guard = entry
                    .try_lock_owned()
                    .map_err(|_| EngineError::InvalidInput("session is busy".into()))?;
                if guard.is_none() {
                    // Gateway retires SessionLost placement, so remove this empty tombstone too.
                    self.sessions.remove_if(&id, |_, current| {
                        Arc::ptr_eq(current, OwnedMutexGuard::mutex(&guard))
                    });
                    return Err(EngineError::SessionLost(id));
                }
                Some(guard)
            }
            None => None,
        };
        let state = guard.as_ref().map(|guard| guard.as_ref().unwrap());
        let old_len = state.map_or(0, |state| state.tokens.len());
        let ready_len = state.map_or(0, |state| state.kv.ready_len());
        let max_seq_len = state.map_or(self.pool.max_seq_len(), |state| state.kv.max_seq_len());
        let full_len = old_len
            .checked_add(request.input.len())
            .filter(|&len| len <= max_seq_len)
            .ok_or_else(|| EngineError::InvalidInput("KV context capacity exceeded".into()))?;
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
        self.model.validate_input(request.input.get_ids())?;
        let remaining = sampling.max_new_tokens.unwrap_or(1);
        let end = full_len
            .checked_add(remaining.saturating_sub(1))
            .filter(|&end| end <= max_seq_len)
            .ok_or_else(|| {
                EngineError::InvalidInput("generation exceeds KV context capacity".into())
            })?;
        let cache = self.prefix_cache.as_ref().filter(|_| old_len == 0);
        let cached = cache
            .as_ref()
            .and_then(|cache| cache.find(request.input.get_ids()));
        let reused_tokens = cached.as_ref().map_or(0, |prefix| prefix.token_len());
        let admission_ready = ready_len.max(reused_tokens);
        if let Some(cache) = &self.prefix_cache {
            cache.evict_for_capacity(
                reserved_tokens.saturating_add(end - admission_ready),
                usize::from(guard.is_none()),
            );
        }
        if !self.pool.can_append(
            reserved_tokens.saturating_add(end - admission_ready),
            usize::from(guard.is_none()),
        ) {
            return Err(KVPoolError::CapacityExhausted(
                "insufficient KV capacity for request".into(),
            )
            .into());
        }
        let mut guard = match guard {
            Some(guard) => guard,
            None => Arc::new(Mutex::new(Some(SessionState {
                tokens: Arc::new(Vec::new()),
                kv: KVContext::new(self.pool.clone())?,
                sampling: SamplingConfig::default(),
            })))
            .try_lock_owned()
            .unwrap(),
        };
        let mut state = guard.take().unwrap();
        if let Some(prefix) = cached {
            cache.unwrap().reuse(&prefix, &mut state.kv)?;
            metrics::counter!("engine_prefix_cache_hits_total").increment(1);
            metrics::counter!("engine_prefix_cache_reused_tokens_total")
                .increment(prefix.token_len() as u64);
            tracing::debug!(target: "engine::prefix_cache", event = "prefix_reused",
                request_id = request.assigned_request_id(), tokens = prefix.token_len());
        }
        let tokens = Arc::make_mut(&mut state.tokens);
        tokens.reserve(request.input.len() + remaining);
        tokens.extend_from_slice(request.input.get_ids());
        Ok(ActiveRequest {
            request_id: request.assigned_request_id(),
            guard,
            state,
            original_progress: SessionProgressBeforeRequest {
                token_count: old_len,
                ready_len,
            },
            gpu_token: None,
            generation: RequestTokenGenerationState {
                sampling,
                remaining,
                generated_tokens: 0,
                model_error: None,
                decode_error: None,
                emitted: false,
                decoder: tokenizer.decode_stream(true),
            },
        })
    }

    /// Ingest/admit, enqueue a packed forward, then wait/commit/process previous results.
    /// Decode gets budget first; only final prefill chunks consume generation budget.
    async fn drive(
        &self,
        commands: &mut mpsc::UnboundedReceiver<RunnerCommand>,
        replies: &mpsc::UnboundedSender<RunnerReply>,
    ) {
        let tokenizer = self.model.tokenizer();
        let mut waiting = VecDeque::<SessionRequest<Encoding>>::new();
        let mut active: Vec<ActiveRequest<'_, P>> = Vec::new();
        let mut batch_policy = MixedChunkPolicy::new(self.max_batch_tokens);
        // Only overlap execution retains results across scheduling iterations.
        let mut pending = Vec::<RequestForwardAwaitingGpuCompletion>::new();
        let mut compute_policy: Option<Policy> = None;
        loop {
            for index in (0..active.len()).rev() {
                if active[index].generation.remaining == 0
                    && (!self.overlap
                        || !pending
                            .iter()
                            .any(|request| request.request_id == active[index].request_id))
                {
                    active
                        .swap_remove(index)
                        .finish(replies, self.prefix_cache.as_ref());
                }
            }
            // A finite ingress snapshot prevents a continuous producer from starving GPU steps.
            for _ in 0..commands.len() {
                match commands.try_recv() {
                    Ok(command) => self.enqueue(command, &mut waiting, replies),
                    Err(_) => break,
                }
            }
            let mut reserved_tokens = active.iter().fold(0usize, |sum, request| {
                sum.saturating_add(request.unreserved_kv_tokens_for_generation())
            });
            for _ in 0..waiting.len() {
                let request = waiting.pop_front().unwrap();
                match self.try_admit(&request, reserved_tokens, tokenizer) {
                    Ok(turn) => {
                        if turn.generation.remaining == 0 {
                            turn.finish(replies, self.prefix_cache.as_ref());
                        } else {
                            reserved_tokens = reserved_tokens
                                .saturating_add(turn.unreserved_kv_tokens_for_generation());
                            active.push(turn);
                        }
                    }
                    Err(EngineError::KVPool(KVPoolError::CapacityExhausted(_)))
                        if !active.is_empty() =>
                    {
                        waiting.push_back(request)
                    }
                    Err(error) => {
                        let _ = replies.send(RunnerReply::Finished {
                            request_id: request.assigned_request_id(),
                            generated_tokens: 0,
                            result: Err(error),
                        });
                    }
                }
            }
            if active.is_empty() {
                match commands.recv().await {
                    Some(command) => self.enqueue(command, &mut waiting, replies),
                    None => break,
                }
                continue;
            }
            let mut scheduled = batch_policy.plan(&mut active);
            // quick path for empty requests
            if scheduled.is_empty() {
                drop(scheduled);
                if self.overlap {
                    self.complete_forward_and_emit_tokens(
                        std::mem::take(&mut pending),
                        &mut active,
                        replies,
                    )
                    .await;
                }
                tokio::task::yield_now().await;
                continue;
            }
            let mut inputs: Vec<_> = scheduled
                .iter_mut()
                .map(|step| {
                    let request = &mut *step.request;
                    let token_ids = if self.overlap {
                        match &request.gpu_token {
                            Some(token) => ModelTokens::Device(token.clone()),
                            None => ModelTokens::SharedHost {
                                tokens: request.state.tokens.clone(),
                                end: step.compute_end,
                            },
                        }
                    } else {
                        ModelTokens::BorrowedHost(&request.state.tokens[..step.compute_end])
                    };
                    ModelInput {
                        token_ids,
                        kv: &mut request.state.kv,
                    }
                })
                .collect();
            let policy = match &compute_policy {
                Some(policy) => Ok(policy.clone()),
                None => self.policy.next_stream().and_then(|stream| {
                    SingleStream::new(stream.device()).map(|policy| {
                        let policy: Policy = Arc::new(policy);
                        compute_policy = Some(policy.clone());
                        policy
                    })
                }),
            };
            // sequential scenario
            if !self.overlap {
                // Serial mode returns CPU IDs and consumes them in this iteration.
                // It creates neither a submission wrapper nor pending request records.
                let outputs = match policy {
                    Ok(policy) => self.model.next_tokens(&mut inputs, policy).await,
                    Err(error) => (0..inputs.len())
                        .map(|_| Err(ModelError::from(error.clone())))
                        .collect(),
                };
                assert_eq!(outputs.len(), inputs.len(), "one prediction per input");
                drop(inputs);
                for (step, output) in scheduled.into_iter().zip(outputs) {
                    if step.compute_end == step.request.sequence_len() || output.is_err() {
                        let request = step.request;
                        request.generation.accept(
                            &mut request.state.tokens,
                            output,
                            request.request_id,
                            replies,
                            |id| self.model.is_eos(id),
                        );
                    }
                }
                tokio::task::yield_now().await;
                continue;
            }
            // overlap scenario
            let forward = match policy {
                Ok(policy) => {
                    self.model
                        .clone()
                        .enqueue_forward_and_return_completion(&mut inputs, policy)
                        .await
                }
                Err(error) => SubmittedGpuWorkWithCompletion::from_completed_output(
                    (0..inputs.len())
                        .map(|_| Err(ModelError::from(error.clone())))
                        .collect(),
                ),
            };
            assert_eq!(
                forward.output.len(),
                inputs.len(),
                "one prediction per input"
            );
            drop(inputs);
            let current = scheduled
                .into_iter()
                .zip(forward.output)
                .map(|(step, prediction)| {
                    let produces_output_token = step.compute_end == step.request.sequence_len();
                    if produces_output_token {
                        step.request.gpu_token = match &prediction {
                            Ok(NextToken::Device(token)) => Some(token.clone()),
                            _ => None,
                        };
                    }
                    RequestForwardAwaitingGpuCompletion {
                        request_id: step.request.request_id,
                        compute_end: step.compute_end,
                        produces_output_token,
                        prediction,
                        gpu_completion: forward.gpu_completion.clone(),
                    }
                })
                .collect::<Vec<_>>();
            // Enqueue current first, then complete/process the prior iteration.
            self.complete_forward_and_emit_tokens(pending, &mut active, replies)
                .await;
            // Known errors stop rescheduling now; their GPU work still drains via pending.
            for failed in &current {
                if let Err(error) = &failed.prediction {
                    let request = active
                        .iter_mut()
                        .find(|request| request.request_id == failed.request_id)
                        .unwrap();
                    request.generation.model_error = Some(error.clone());
                    request.generation.remaining = 0;
                }
            }
            pending = current;
            // Even immediately-ready model futures must let ingress/routing tasks run.
            tokio::task::yield_now().await;
        }
    }

    /// Receive new work at every forward boundary; cancellation drops active turns
    /// while GPU leases retain any storage still in flight.
    pub async fn run(mut self, cancel_token: CancellationToken) {
        let mut commands = self.command_rx.take().unwrap();
        tokio::select! {
            biased;
            _ = cancel_token.cancelled() => {},
            _ = self.drive(&mut commands, &self.batch_reply_tx) => {},
        }
        self.sessions.clear();
        self.prefix_cache = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BatchIn;
    use std::{collections::HashMap, time::Duration};

    impl<M: CausalModel<P> + 'static, P: KVPool + 'static> ModelRunner<M, P> {
        async fn prepare_batch(
            &self,
            batch: BatchIn,
            replies: &mpsc::UnboundedSender<RunnerReply>,
        ) {
            let (tx, mut rx) = mpsc::unbounded_channel();
            tx.send(RunnerCommand::Execute(batch)).unwrap();
            drop(tx);
            self.drive(&mut rx, replies).await;
        }

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
                    RunnerReply::Finished {
                        request_id, result, ..
                    } => {
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

    #[derive(Debug)]
    struct ForwardStep {
        histories: Vec<Vec<u32>>,
        ready_lens: Vec<usize>,
        resume: tokio::sync::oneshot::Sender<()>,
    }

    impl ForwardStep {
        fn members(&self) -> Vec<u32> {
            self.histories.iter().map(|tokens| tokens[0]).collect()
        }
    }

    struct SteppedModel {
        inner: SimpleMLPEngine,
        steps: mpsc::UnboundedSender<ForwardStep>,
    }

    impl CausalModel<NaiveKVPool> for SteppedModel {
        fn tokenizer(&self) -> &Tokenizer {
            self.inner.tokenizer()
        }
        fn validate_input(&self, ids: &[u32]) -> Result<(), ModelError> {
            self.inner.validate_input(ids)
        }
        async fn forward(
            &self,
            batch: &mut [ModelInput<'_, NaiveKVPool>],
            policy: Policy,
        ) -> Vec<Result<NextToken, ModelError>> {
            let (resume, proceed) = tokio::sync::oneshot::channel();
            self.steps
                .send(ForwardStep {
                    histories: batch
                        .iter()
                        .map(|input| input.token_ids.host().unwrap().to_vec())
                        .collect(),
                    ready_lens: batch.iter().map(|input| input.kv.ready_len()).collect(),
                    resume,
                })
                .unwrap();
            proceed.await.unwrap();
            CausalModel::forward(&self.inner, batch, policy).await
        }
    }

    struct ContinuousRunner {
        scheduler: crate::scheduler::Scheduler,
        pool: Arc<NaiveKVPool>,
        steps: mpsc::UnboundedReceiver<ForwardStep>,
        finished_tokens: Arc<std::sync::Mutex<HashMap<RequestId, usize>>>,
        cancel: CancellationToken,
        task: tokio::task::JoinHandle<()>,
    }

    impl ContinuousRunner {
        async fn new(config: NaiveKVPoolConfig) -> Result<Self, EngineError> {
            Self::with_budget(config, DEFAULT_MAX_BATCH_TOKENS).await
        }

        async fn with_budget(
            config: NaiveKVPoolConfig,
            max_batch_tokens: NonZeroUsize,
        ) -> Result<Self, EngineError> {
            let policy = global_policy(0)?;
            let pool = NaiveKVPool::new(config, &policy).await?;
            let (steps, step_rx) = mpsc::unbounded_channel();
            let model = SteppedModel {
                inner: SimpleMLPEngine::random(0).await?,
                steps,
            };
            let cancel = CancellationToken::new();
            let (commands, rx) = mpsc::unbounded_channel();
            let (replies, reply_rx) = mpsc::unbounded_channel();
            let finished_tokens = Arc::new(std::sync::Mutex::new(HashMap::new()));
            let recorded = finished_tokens.clone();
            let scheduler = crate::scheduler::Scheduler::new_with_observers(
                cancel.child_token(),
                commands,
                reply_rx,
                vec![Arc::new(
                    move |request_id: RequestId,
                          stats: &crate::scheduler::RequestStats,
                          _result: &Result<(), EngineError>| {
                        if let Some(tokens) = stats.generated_tokens {
                            recorded.lock().unwrap().insert(request_id, tokens);
                        }
                    },
                )],
            );
            let mut runner = ModelRunner::new(rx, replies, model, pool.clone(), policy);
            runner.max_batch_tokens = max_batch_tokens;
            // SteppedModel wraps Host-only SimpleMLP; accept its tokens in the same iteration.
            runner.overlap = false;
            let task = tokio::spawn(runner.run(cancel.child_token()));
            Ok(Self {
                scheduler,
                pool,
                steps: step_rx,
                finished_tokens,
                cancel,
                task,
            })
        }

        async fn step(&mut self) -> ForwardStep {
            tokio::time::timeout(Duration::from_secs(30), self.steps.recv())
                .await
                .expect("next forward must become runnable")
                .expect("Runner must remain connected")
        }

        async fn stop(self) {
            self.cancel.cancel();
            self.scheduler.wait_stopped().await;
            self.task.await.unwrap();
            assert_eq!(self.pool.metrics().index.assigned_bytes, 0);
            assert_eq!(self.pool.metrics().layers.assigned_bytes, 0);
        }
    }

    fn generation(session: Option<SessionId>, input: &str, budget: usize) -> SessionRequest {
        let mut request = SessionRequest::new(session, input.to_owned());
        request.sampling = Some(SamplingConfig {
            max_new_tokens: Some(budget),
            ..Default::default()
        });
        request
    }

    /// Controlled output delivery exercises the pipeline, not neural numerics.
    /// The first CPU result cannot arrive until a second GPU submission starts.
    struct FeedbackModel {
        tokenizer: Tokenizer,
        calls: Arc<std::sync::atomic::AtomicUsize>,
        release_host: Arc<tokio::sync::Notify>,
        gate_first: bool,
        fail_host: bool,
        host_started: Option<mpsc::UnboundedSender<()>>,
    }

    impl CausalModel<NaiveKVPool> for FeedbackModel {
        fn tokenizer(&self) -> &Tokenizer {
            &self.tokenizer
        }

        async fn forward(
            &self,
            batch: &mut [ModelInput<'_, NaiveKVPool>],
            policy: Policy,
        ) -> Vec<Result<NextToken, ModelError>> {
            use cuda_async::device_operation::*;
            use cutile::{
                api,
                cuda_core::{CudaContext, PinnedHostBuffer},
            };
            use std::sync::atomic::Ordering;
            let call = self.calls.fetch_add(1, Ordering::Relaxed);
            if call == 1 && self.gate_first {
                assert!(
                    matches!(batch[0].token_ids, ModelTokens::Device(_)),
                    "next forward must use GPU feedback before CPU history is available"
                );
            }
            let count = batch.len();
            let result = Self::execute_append_batch(batch, &policy, |_, exec| {
                // SAFETY: execute_append_batch owns and awaits this GPU submission.
                let values = unsafe {
                    api::copy_host_vec_to_device(&Arc::new(vec![42i32; count])).execute(exec)?
                };
                if call == 1 {
                    self.release_host.notify_one();
                }
                Ok(values)
            })
            .await;
            let values = match result {
                Ok(values) => Arc::new(values),
                Err(error) => return (0..count).map(|_| Err(error.clone())).collect(),
            };
            let gate = self.release_host.clone();
            let wait = self.gate_first && call == 0;
            let device = values.device_id();
            let fail_host = self.fail_host;
            let host_started = self.host_started.clone();
            let host = async move {
                if let Some(started) = host_started {
                    let _ = started.send(());
                }
                if wait {
                    gate.notified().await;
                }
                if fail_host {
                    return Err(ModelError::Execution("readback failed".into()));
                }
                let values = Arc::new(
                    PinnedHostBuffer::<i32>::zeroed(
                        &CudaContext::new(device).map_err(cuda_async::error::DeviceError::from)?,
                        count,
                    )
                    .map_err(cuda_async::error::DeviceError::from)?,
                );
                // SAFETY: this test owns the buffer; no DMA or reader exists yet.
                for index in 0..count {
                    unsafe {
                        values.as_ptr().cast_mut().add(index).write(42);
                    }
                }
                Ok(values)
            };
            GpuTokenPredictionHandle::batch(
                values,
                host,
                256,
                batch.iter().map(|input| input.kv.ready_len()),
            )
            .into_iter()
            .map(|token| Ok(NextToken::Device(token)))
            .collect()
        }
    }

    #[tokio::test]
    async fn overlap_and_serial_modes_control_feedback_and_eos_lookahead() -> Result<(), EngineError>
    {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let template = runner().await?;
        struct CompletionGated {
            inner: FeedbackModel,
            released: tokio::sync::Notify,
        }
        impl CausalModel<NaiveKVPool> for CompletionGated {
            fn tokenizer(&self) -> &Tokenizer {
                self.inner.tokenizer()
            }
            async fn forward(
                &self,
                batch: &mut [ModelInput<'_, NaiveKVPool>],
                policy: Policy,
            ) -> Vec<Result<NextToken, ModelError>> {
                assert!(batch
                    .iter()
                    .all(|input| matches!(input.token_ids, ModelTokens::BorrowedHost(_))));
                self.inner.forward(batch, policy).await
            }
            async fn enqueue_forward_and_return_completion(
                self: Arc<Self>,
                batch: &mut [ModelInput<'_, NaiveKVPool>],
                policy: Policy,
            ) -> SubmittedGpuWorkWithCompletion {
                use futures::FutureExt;
                let call = self.inner.calls.load(Ordering::Relaxed);
                let outputs = self.inner.forward(batch, policy).await;
                if call == 1 {
                    self.released.notify_one();
                }
                let completion = async move {
                    // A hidden await of F0 before submission of F1 deadlocks this test.
                    if self.inner.gate_first && call == 0 {
                        self.released.notified().await;
                    }
                    Ok(())
                }
                .boxed()
                .shared();
                SubmittedGpuWorkWithCompletion {
                    output: outputs,
                    gpu_completion: completion,
                    compute_stream_id: None,
                }
            }
        }
        for overlap in [false, true] {
            for (budget, eos, expected_calls, expected_history, expected_ready) in [
                (3, None, 3, vec![1, 42, 42, 42], 3),
                (
                    4,
                    Some(42),
                    if overlap { 2 } else { 1 },
                    vec![1, 42],
                    if overlap { 2 } else { 1 },
                ),
                (1, None, 1, vec![1, 42], 1),
            ] {
                let calls = Arc::new(AtomicUsize::new(0));
                let model = FeedbackModel {
                    tokenizer: template.model.tokenizer().clone(),
                    calls: calls.clone(),
                    release_host: Arc::default(),
                    gate_first: overlap && budget > 1,
                    fail_host: false,
                    host_started: None,
                };
                let model = CompletionGated {
                    inner: model,
                    released: tokio::sync::Notify::new(),
                };
                let (_, rx) = mpsc::unbounded_channel();
                let (tx, _) = mpsc::unbounded_channel();
                let mut runner = ModelRunner::new(
                    rx,
                    tx,
                    model,
                    template.pool.clone(),
                    template.policy.clone(),
                );
                runner.overlap = overlap;
                runner.open_session(0)?;
                let mut request = generation(Some(0), "1", budget).with_request_id(0);
                request.sampling.as_mut().unwrap().eos_token_id =
                    eos.map(crate::runner::loader::EosTokenId::One);
                tokio::time::timeout(Duration::from_secs(20), runner.execute_session(request))
                    .await
                    .expect(
                        "neither prior completion nor CPU readback may gate the next submission",
                    )?;
                assert_eq!(calls.load(Ordering::Relaxed), expected_calls);
                let entry = runner.sessions.get(&0).unwrap().value().clone();
                let guard = entry.lock().await;
                let state = guard.as_ref().unwrap();
                assert_eq!(*state.tokens, expected_history);
                assert_eq!(state.kv.ready_len(), expected_ready);
                drop(guard);
                runner.close_session(0)?;
                assert_eq!(runner.pool.metrics().layers.assigned_bytes, 0);
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn mixed_host_and_device_predictions_wait_for_missing_feedback() -> Result<(), EngineError>
    {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let template = runner().await?;
        struct MixedFeedback(FeedbackModel);
        impl CausalModel<NaiveKVPool> for MixedFeedback {
            fn tokenizer(&self) -> &Tokenizer {
                self.0.tokenizer()
            }
            async fn forward(
                &self,
                batch: &mut [ModelInput<'_, NaiveKVPool>],
                policy: Policy,
            ) -> Vec<Result<NextToken, ModelError>> {
                let mut outputs = self.0.forward(batch, policy).await;
                outputs[0] = Ok(NextToken::Host(42));
                outputs
            }
        }
        let model = MixedFeedback(FeedbackModel {
            tokenizer: template.model.tokenizer().clone(),
            calls: Arc::default(),
            release_host: Arc::default(),
            gate_first: false,
            fail_host: false,
            host_started: None,
        });
        let expected = model.tokenizer().decode(&[42, 42, 42], true).unwrap();
        let (_, rx) = mpsc::unbounded_channel();
        let (tx, _) = mpsc::unbounded_channel();
        let runner = ModelRunner::new(
            rx,
            tx,
            model,
            template.pool.clone(),
            template.policy.clone(),
        );
        let outputs = tokio::time::timeout(
            Duration::from_secs(20),
            runner.execute_collect(vec![
                generation(None, "1", 3).with_request_id(0),
                generation(None, "2", 3).with_request_id(1),
            ]),
        )
        .await
        .unwrap();
        for (_, output) in outputs {
            assert_eq!(output?.output, expected);
        }
        assert_eq!(runner.pool.metrics().layers.assigned_bytes, 0);
        Ok(())
    }

    #[tokio::test]
    async fn overlap_readback_failure_invalidates_the_advanced_session() -> Result<(), EngineError>
    {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let template = runner().await?;
        let model = FeedbackModel {
            tokenizer: template.model.tokenizer().clone(),
            calls: Arc::default(),
            release_host: Arc::default(),
            gate_first: true,
            fail_host: true,
            host_started: None,
        };
        let (_, rx) = mpsc::unbounded_channel();
        let (tx, _) = mpsc::unbounded_channel();
        let runner = ModelRunner::new(
            rx,
            tx,
            model,
            template.pool.clone(),
            template.policy.clone(),
        );
        runner.open_session(0)?;
        let result = tokio::time::timeout(
            Duration::from_secs(20),
            runner.execute_session(generation(Some(0), "1", 3).with_request_id(0)),
        )
        .await
        .unwrap();
        assert!(matches!(
            result,
            Err(EngineError::Model(ModelError::Execution(_)))
        ));
        let entry = runner.sessions.get(&0).unwrap().value().clone();
        assert!(entry.lock().await.is_none());
        runner.close_session(0)?;
        assert_eq!(runner.pool.metrics().layers.assigned_bytes, 0);
        Ok(())
    }

    #[tokio::test]
    async fn overlap_cancellation_releases_a_pending_host_result() -> Result<(), EngineError> {
        use futures::TryStreamExt;
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let template = runner().await?;
        let (started, mut host_started) = mpsc::unbounded_channel();
        let model = FeedbackModel {
            tokenizer: template.model.tokenizer().clone(),
            calls: Arc::default(),
            release_host: Arc::default(),
            gate_first: true,
            fail_host: false,
            host_started: Some(started),
        };
        let cancel = CancellationToken::new();
        let (commands, rx) = mpsc::unbounded_channel();
        let (replies, reply_rx) = mpsc::unbounded_channel();
        let scheduler = crate::scheduler::Scheduler::new(cancel.child_token(), commands, reply_rx);
        let runner = ModelRunner::new(
            rx,
            replies,
            model,
            template.pool.clone(),
            template.policy.clone(),
        );
        let task = tokio::spawn(runner.run(cancel.child_token()));
        // Final token cannot launch lookahead and the controlled CPU result never arrives.
        let stream = scheduler.execute_stream(generation(None, "1", 1))?;
        tokio::time::timeout(Duration::from_secs(20), host_started.recv())
            .await
            .unwrap()
            .unwrap();
        cancel.cancel();
        scheduler.wait_stopped().await;
        task.await.unwrap();
        assert!(matches!(
            stream.try_collect::<Vec<_>>().await,
            Err(EngineError::ChannelClosed)
        ));
        assert_eq!(template.pool.metrics().layers.assigned_bytes, 0);
        Ok(())
    }

    #[tokio::test]
    async fn mixed_chunk_policy_reserves_decode_without_prioritizing_one_token_prompts(
    ) -> Result<(), EngineError> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let runner = runner().await?;
        let tokenizer = runner.model.tokenizer();
        let mut active = Vec::new();
        for (id, input) in [(0, "1 2 3"), (1, "7"), (2, "9")] {
            let encoded = tokenizer.encode(input, true).unwrap();
            let request = SessionRequest::new(None, encoded).with_request_id(id);
            active.push(runner.try_admit(&request, 0, tokenizer)?);
        }
        // A one-token prompt does not bypass the older prefill.
        let mut policy = MixedChunkPolicy::new(NonZeroUsize::new(1).unwrap());
        let steps = policy.plan(&mut active);
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].request.request_id, 0);
        assert_eq!(steps[0].compute_end, 1);
        drop(steps);

        active[2].generation.generated_tokens = 1;
        // The decoder is last in active order; its reservation still prevents
        // the earlier long prefill from consuming the entire budget.
        let mut policy = MixedChunkPolicy::new(NonZeroUsize::new(2).unwrap());
        let steps = policy.plan(&mut active);
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].request.request_id, 0);
        assert_eq!(steps[1].request.request_id, 2);
        assert!(steps.iter().all(|step| step.compute_end == 1));
        drop(steps);
        assert!(active
            .iter()
            .all(|request| request.state.kv.ready_len() == 0));

        // No prefill budget: select pure decode. Excess decoders wait unchanged.
        active[1].generation.generated_tokens = 1;
        let mut policy = MixedChunkPolicy::new(NonZeroUsize::new(1).unwrap());
        let steps = policy.plan(&mut active);
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].request.request_id, 1);
        drop(steps);
        assert_eq!(active[2].state.tokens.as_slice(), [9]);
        assert!(policy.plan::<NaiveKVPool>(&mut []).is_empty());
        drop(active);
        assert_eq!(runner.pool.metrics().index.assigned_bytes, 0);
        Ok(())
    }

    /// Intermediate chunks only advance KV: no sampled token is appended, counted
    /// or decoded. A short peer joins and decodes before the long prompt finishes.
    #[tokio::test]
    async fn chunked_prefill_interleaves_decode_and_session_continuation() -> Result<(), EngineError>
    {
        use futures::TryStreamExt;
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let reference = runner().await?;
        reference.open_session(0)?;
        let expected_a = reference
            .execute_session(generation(Some(0), "1 2 3 4 5", 2).with_request_id(100))
            .await?;
        let expected_a2 = reference
            .execute_session(generation(Some(0), "10 11 12", 1).with_request_id(101))
            .await?;
        reference.close_session(0)?;
        let expected_b = reference
            .execute_session(generation(None, "7", 5).with_request_id(102))
            .await?;
        let mut harness = ContinuousRunner::with_budget(
            NaiveKVPoolConfig {
                max_rows: 2,
                num_slots: 32,
                max_seq_len: 32,
                ..Default::default()
            },
            NonZeroUsize::new(2).unwrap(),
        )
        .await?;
        let scheduler = harness.scheduler.clone();
        let result = tokio::time::timeout(Duration::from_secs(30), async {
            let session = scheduler.open_session().await?;
            let b = scheduler.execute_stream(generation(None, "7", 5))?;
            let first = harness.step().await;
            assert_eq!(first.histories, [vec![7]]);
            assert_eq!(first.ready_lens, [0]);
            let a = scheduler.execute_stream(generation(Some(session), "1 2 3 4 5", 2))?;
            first.resume.send(()).unwrap();
            for end in 1..=4 {
                let step = harness.step().await;
                assert_eq!(step.members(), [7, 1]);
                assert_eq!(step.histories[0].len(), end + 1);
                assert_eq!(step.histories[1], (1..=end as u32).collect::<Vec<_>>());
                assert_eq!(step.ready_lens, [end, end - 1]);
                // One decode token plus one prefill token fills the two-token
                // budget. Intermediate chunks never append a prediction to A.
                assert_eq!(
                    step.histories
                        .iter()
                        .zip(&step.ready_lens)
                        .map(|(ids, &ready)| ids.len() - ready)
                        .sum::<usize>(),
                    2
                );
                step.resume.send(()).unwrap();
            }
            let last_prefill = harness.step().await;
            let b = b.try_collect::<Vec<_>>().await?;
            assert_eq!(
                b.iter().map(|r| r.output.as_str()).collect::<String>(),
                expected_b.output
            );
            assert_eq!(harness.finished_tokens.lock().unwrap().get(&0), Some(&5));
            assert!(!harness.finished_tokens.lock().unwrap().contains_key(&1));
            // B finished while A's prompt still needed its final token's KV.
            assert_eq!(last_prefill.histories, [vec![1, 2, 3, 4, 5]]);
            assert_eq!(last_prefill.ready_lens, [4]);
            last_prefill.resume.send(()).unwrap();
            let decode = harness.step().await;
            assert_eq!(decode.histories[0].len(), 6);
            assert_eq!(decode.ready_lens, [5]);
            decode.resume.send(()).unwrap();
            let a = a.try_collect::<Vec<_>>().await?;
            assert_eq!(
                a.iter().map(|r| r.output.as_str()).collect::<String>(),
                expected_a.output
            );
            let a2 = scheduler.execute_stream(generation(Some(session), "10 11 12", 1))?;
            let sixth = harness.step().await;
            assert_eq!(sixth.ready_lens, [6]);
            assert_eq!(sixth.histories[0].len(), 8);
            assert_eq!(sixth.histories[0][7], 10);
            sixth.resume.send(()).unwrap();
            let seventh = harness.step().await;
            assert_eq!(seventh.ready_lens, [8]);
            assert_eq!(seventh.histories[0].len(), 10);
            assert_eq!(&seventh.histories[0][7..], [10, 11, 12]);
            seventh.resume.send(()).unwrap();
            let a2 = a2.try_collect::<Vec<_>>().await?;
            assert_eq!(
                a2.iter().map(|r| r.output.as_str()).collect::<String>(),
                expected_a2.output
            );
            assert_eq!(
                *harness.finished_tokens.lock().unwrap(),
                HashMap::from([(0, 5), (1, 2), (2, 1)])
            );
            scheduler.close_session(session).await?;
            Ok::<_, EngineError>(())
        })
        .await
        .expect("chunked generation must make progress");
        harness.stop().await;
        result
    }

    #[tokio::test]
    async fn chunked_prefill_budget_one_keeps_unscheduled_requests_intact(
    ) -> Result<(), EngineError> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let mut runner = runner().await?;
        let batch = vec![
            generation(None, "1 2 3 4 5", 3).with_request_id(0),
            generation(None, "7", 3).with_request_id(1),
            generation(None, "10 11 12", 2).with_request_id(2),
        ];
        let expected = runner.execute_collect(batch.clone()).await;
        for limit in [1, 2, 3] {
            runner.max_batch_tokens = NonZeroUsize::new(limit).unwrap();
            let actual = tokio::time::timeout(
                Duration::from_secs(30),
                runner.execute_collect(batch.clone()),
            )
            .await
            .expect("a budget smaller than the active set must still make progress");
            for ((id, value), (expected_id, expected)) in actual.into_iter().zip(&expected) {
                assert_eq!(id, *expected_id);
                assert_eq!(value?.output, expected.as_ref().unwrap().output);
            }
            assert_eq!(runner.pool.metrics().layers.assigned_bytes, 0);
        }
        Ok(())
    }

    /// Happy path: [A] -> [A, B] -> [A] -> [C] -> [C, A2].
    /// B joins before A finishes; A2 continues A's session only after A's EOF.
    /// Pauses control arrival timing only; all forwards execute on the GPU.
    #[tokio::test]
    async fn continuous_batching_happy_path_matches_serial_generation() -> Result<(), EngineError> {
        use futures::TryStreamExt;
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let reference = runner().await?;
        reference.open_session(0)?;
        let expected_a = reference
            .execute_session(generation(Some(0), "1 2", 3).with_request_id(100))
            .await?
            .output;
        let expected_a2 = reference
            .execute_session(generation(Some(0), "5", 1).with_request_id(101))
            .await?
            .output;
        reference.close_session(0)?;
        let expected_b = reference
            .execute_session(generation(None, "3 4", 1).with_request_id(102))
            .await?
            .output;
        let expected_c = reference
            .execute_session(generation(None, "7 8", 2).with_request_id(103))
            .await?
            .output;

        let mut harness = ContinuousRunner::new(NaiveKVPoolConfig {
            max_rows: 2,
            num_slots: 16,
            max_seq_len: 16,
            ..Default::default()
        })
        .await?;
        let scheduler = harness.scheduler.clone();
        let result = tokio::time::timeout(Duration::from_secs(30), async {
            let session = scheduler.open_session().await?;
            let a = scheduler.execute_stream(generation(Some(session), "1 2", 3))?;
            let first = harness.step().await;
            assert_eq!(first.members(), [1]);
            assert_eq!(first.histories[0], [1, 2]);
            // Admit B while A still has two generation steps remaining.
            let b = scheduler.execute_stream(generation(None, "3 4", 1))?;
            first.resume.send(()).unwrap();
            let second = harness.step().await;
            assert_eq!(second.members(), [1, 3]);
            // step() pauses before execution: A has appended exactly one token
            // from the first completed forward, while B still has only its prompt.
            assert_eq!(second.histories[0].len(), first.histories[0].len() + 1);
            assert!(second.histories[0].starts_with(&first.histories[0]));
            assert_eq!(second.histories[1], [3, 4]);
            second.resume.send(()).unwrap();
            let third = harness.step().await;
            assert_eq!(third.members(), [1]);
            assert_eq!(third.histories[0].len(), second.histories[0].len() + 1);
            assert!(third.histories[0].starts_with(&second.histories[0]));
            let actual_b = b.try_collect::<Vec<_>>().await?;
            assert_eq!(harness.finished_tokens.lock().unwrap().get(&1), Some(&1));
            assert_eq!(
                actual_b
                    .into_iter()
                    .map(|response| response.output)
                    .collect::<String>(),
                expected_b
            );
            let c = scheduler.execute_stream(generation(None, "7 8", 2))?;
            third.resume.send(()).unwrap();
            let fourth = harness.step().await;
            assert_eq!(fourth.members(), [7]);
            assert_eq!(fourth.histories[0], [7, 8]);
            let actual_a = a
                .try_collect::<Vec<_>>()
                .await?
                .into_iter()
                .map(|response| response.output)
                .collect::<String>();
            assert_eq!(actual_a, expected_a);
            assert_eq!(harness.finished_tokens.lock().unwrap().get(&0), Some(&3));
            // A's client submits its next turn only after observing completion.
            let next = scheduler.execute_stream(generation(Some(session), "5", 1))?;
            fourth.resume.send(()).unwrap();
            let fifth = harness.step().await;
            assert_eq!(fifth.members(), [7, 1]);
            assert_eq!(fifth.histories[0].len(), fourth.histories[0].len() + 1);
            assert!(fifth.histories[0].starts_with(&fourth.histories[0]));
            assert_eq!(fifth.histories[1].len(), 6);
            assert_eq!(fifth.histories[1].last(), Some(&5));
            fifth.resume.send(()).unwrap();
            let actual_a2 = next
                .try_collect::<Vec<_>>()
                .await?
                .into_iter()
                .map(|response| response.output)
                .collect::<String>();
            assert_eq!(actual_a2, expected_a2);
            let actual_c = c.try_collect::<Vec<_>>().await?;
            // Scheduler assigns A/B/C/A2 IDs 0/1/2/3. Count model tokens, not text
            // deltas: a decoder may buffer tokens or emit a variable-sized chunk.
            assert_eq!(
                *harness.finished_tokens.lock().unwrap(),
                HashMap::from([(0, 3), (1, 1), (2, 2), (3, 1)])
            );
            assert_eq!(
                actual_c
                    .into_iter()
                    .map(|response| response.output)
                    .collect::<String>(),
                expected_c
            );
            scheduler.close_session(session).await?;
            assert_eq!(harness.pool.metrics().index.assigned_bytes, 0);
            Ok::<_, EngineError>(())
        })
        .await;
        harness.stop().await;
        result.expect("dynamic membership and client-serialized turns must make progress")
    }

    #[tokio::test]
    async fn continuous_admission_protects_future_decode_slots_and_skips_waiting_peers(
    ) -> Result<(), EngineError> {
        use futures::TryStreamExt;
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let mut harness = ContinuousRunner::new(NaiveKVPoolConfig {
            max_rows: 2,
            num_slots: 4,
            max_seq_len: 8,
            ..Default::default()
        })
        .await?;
        let scheduler = harness.scheduler.clone();
        let result = tokio::time::timeout(Duration::from_secs(30), async {
            let a = scheduler.execute_stream(generation(None, "1", 3))?;
            let first = harness.step().await;
            assert_eq!(first.members(), [1]);
            let b = scheduler.execute_stream(generation(None, "7 8", 1))?;
            let c = scheduler.execute_stream(generation(None, "9", 1))?;
            first.resume.send(()).unwrap();
            let second = harness.step().await;
            // A owns one slot and needs two more; only C fits the remaining budget.
            assert_eq!(second.members(), [1, 9]);
            second.resume.send(()).unwrap();
            let third = harness.step().await;
            assert_eq!(third.members(), [1]);
            assert_eq!(c.try_collect::<Vec<_>>().await?.len(), 1);
            third.resume.send(()).unwrap();
            let fourth = harness.step().await;
            assert_eq!(fourth.members(), [7]);
            assert_eq!(a.try_collect::<Vec<_>>().await?.len(), 3);
            assert_eq!(harness.pool.metrics().index.assigned_bytes, 8 * 4);
            fourth.resume.send(()).unwrap();
            assert_eq!(b.try_collect::<Vec<_>>().await?.len(), 1);
            Ok::<_, EngineError>(())
        })
        .await;
        harness.stop().await;
        result.expect("waiting work must progress after active KV is released")
    }

    #[tokio::test]
    async fn continuous_cancellation_releases_active_and_waiting_requests(
    ) -> Result<(), EngineError> {
        use futures::TryStreamExt;
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let mut harness = ContinuousRunner::new(NaiveKVPoolConfig {
            max_rows: 2,
            num_slots: 16,
            max_seq_len: 8,
            ..Default::default()
        })
        .await?;
        let scheduler = harness.scheduler.clone();
        let a = scheduler.execute_stream(generation(None, "1 2", 3))?;
        let b = scheduler.execute_stream(generation(None, "3 4", 3))?;
        let first = harness.step().await;
        assert_eq!(first.members(), [1, 3]);
        let waiting = scheduler.execute_stream(generation(None, "7", 1))?;
        first.resume.send(()).unwrap();
        let second = harness.step().await;
        assert_eq!(second.members(), [1, 3]);
        harness.stop().await;
        for result in [
            a.try_collect::<Vec<_>>().await,
            b.try_collect::<Vec<_>>().await,
            waiting.try_collect::<Vec<_>>().await,
        ] {
            assert!(matches!(result, Err(EngineError::ChannelClosed)));
        }
        drop(second);
        Ok(())
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
            async fn forward(
                &self,
                batch: &mut [ModelInput<'_, NaiveKVPool>],
                policy: Policy,
            ) -> Vec<Result<NextToken, ModelError>> {
                if batch.iter().any(|input| input.sequence_len() > 2) {
                    assert_eq!(batch.len(), 1);
                    self.entered.notify_one();
                    self.resume.notified().await;
                }
                CausalModel::forward(&self.inner, batch, policy).await
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

            async fn forward(
                &self,
                batch: &mut [ModelInput<'_, NaiveKVPool>],
                policy: Policy,
            ) -> Vec<Result<NextToken, ModelError>> {
                if batch[0].sequence_len() == 2 && !self.paused.swap(true, Ordering::Relaxed) {
                    self.entered.notify_one();
                    self.resume.notified().await;
                }
                let mut outputs = CausalModel::forward(&self.inner, batch, policy).await;
                for (output, input) in outputs.iter_mut().zip(batch) {
                    if let Ok(NextToken::Host(token)) = output {
                        *token = input.sequence_len() as u32;
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
        runner.close_session(10)?;
        runner.close_session(20)?;
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
            .with_overlap(false)
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
                } else if index == 3 {
                    assert!(matches!(result, Err(EngineError::SessionLost(999))));
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
        runner.close_session(10)?;
        assert_eq!(runner.pool.metrics(), initial);
        runner.open_session(30)?;
        let entry = runner.sessions.get(&30).unwrap().value().clone();
        entry.lock().await.take();
        assert!(matches!(
            runner.execute_session(request(30, Some(30), "1")).await,
            Err(EngineError::SessionLost(30))
        ));
        assert!(!runner.sessions.contains_key(&30));
        assert!(matches!(
            runner.close_session(30),
            Err(EngineError::SessionLost(30))
        ));
        assert!(matches!(
            runner.close_session(999),
            Err(EngineError::SessionLost(999))
        ));
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
            async fn forward(
                &self,
                batch: &mut [ModelInput<'_, NaiveKVPool>],
                policy: Policy,
            ) -> Vec<Result<NextToken, ModelError>> {
                self.batches.lock().unwrap().push(batch.len());
                CausalModel::forward(&self.inner, batch, policy).await
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
        runner.close_session(10)?;
        runner.close_session(20)?;
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

            async fn forward(
                &self,
                _batch: &mut [ModelInput<'_, NaiveKVPool>],
                _policy: Policy,
            ) -> Vec<Result<NextToken, ModelError>> {
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
        runner.close_session(10)?;
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
        runner.close_session(10)?;
        runner.close_session(20)?;
        assert!(runner.close_session(10).is_err());
        assert_eq!(runner.pool.metrics(), initial);
        Ok(())
    }

    #[tokio::test]
    async fn busy_session_is_rejected_without_blocking_another_session() -> Result<(), EngineError>
    {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let runner = runner().await?;
        runner.open_session(10)?;
        runner.open_session(20)?;
        let entry = runner.sessions.get(&10).unwrap().value().clone();
        let held = entry.lock_owned().await;
        assert!(matches!(runner.close_session(10),
            Err(EngineError::InvalidInput(message)) if message == "session is busy"));
        let replies = runner
            .execute_collect(vec![
                request(1, Some(10), "1 2"),
                request(2, Some(20), "8 9 7"),
            ])
            .await;
        assert!(matches!(&replies[0].1,
            Err(EngineError::InvalidInput(message)) if message == "session is busy"));
        assert!(replies[1].1.is_ok());
        drop(held);
        runner.execute_session(request(3, Some(10), "1 2")).await?;
        let a = runner.execute_session(request(4, Some(10), "3")).await;
        let b = runner.execute_session(request(5, Some(10), "4")).await;
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
        runner.close_session(10)?;
        runner.close_session(20)?;
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
            runner.close_session(id)?;
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
        let (a, stateless) = tokio::join!(
            scheduler.execute(SessionRequest::new(Some(session_id), "1 2".into())),
            scheduler.execute(SessionRequest::new(None, "8 9".into()))
        );
        let a = a?;
        let b = scheduler
            .execute(SessionRequest::new(Some(session_id), "3".into()))
            .await?;
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
            Err(EngineError::SessionLost(id)) if id == session_id
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
