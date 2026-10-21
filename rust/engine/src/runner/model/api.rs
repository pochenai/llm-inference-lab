use std::{future::Future, ops::Range, sync::Arc};

use cuda_async::{device_operation::*, scheduling_policies::SchedulingPolicy};
use cutile::tensor::Tensor;
use futures::{stream, Stream};
use tokenizers::Tokenizer;

use crate::{
    runner::kv_pool::{KVContext, KVPool},
    types::{BatchIn, RequestId, SamplingConfig},
};

use super::ModelError;

pub struct ModelInput<'a, P: KVPool> {
    /// Full accepted token history; only token_ids[kv.ready_len()..] is computed.
    pub token_ids: &'a [u32],
    pub kv: &'a mut KVContext<P>,
}

#[derive(Debug)]
pub struct ModelOutput {
    pub request_id: RequestId,
    pub result: Result<Vec<u32>, ModelError>,
}

pub trait Model<P: KVPool>: Send + Sync {
    fn tokenizer(&self) -> &Tokenizer;

    /// CPU validation of newly encoded tokens and effective sampling, before admission.
    fn validate_input(
        &self,
        _token_ids: &[u32],
        _sampling: &SamplingConfig,
    ) -> Result<(), ModelError> {
        Ok(())
    }

    /// Pack uncached token suffixes for i32 GPU inputs and pad to a positive multiple
    /// chosen by the model. Query ranges index packed rows and exclude final padding.
    fn pack_token_ids(
        batch: &[ModelInput<'_, P>],
        pad_multiple: usize,
    ) -> (Vec<i32>, Vec<Range<usize>>) {
        let mut packed_token_ids = Vec::new();
        let mut query_ranges = Vec::new();
        for input in batch {
            let start = packed_token_ids.len();
            let suffix = &input.token_ids[input.kv.ready_len()..];
            packed_token_ids.extend(suffix.iter().map(|&id| id as i32));
            query_ranges.push(start..packed_token_ids.len());
        }
        let rows = packed_token_ids.len().div_ceil(pad_multiple) * pad_multiple;
        packed_token_ids.resize(rows, 0);
        (packed_token_ids, query_ranges)
    }

    /// Prepare mappings for a same-pool batch before submitting K/V work on exec.
    /// Await successful GPU completion before committing logical progress. The
    /// callback must submit all required layers; failure leaves appends uncommitted.
    fn execute_append_batch<T: Send>(
        batch: &mut [ModelInput<'_, P>],
        policy: &Arc<dyn SchedulingPolicy>,
        submit: impl FnOnce(&mut [ModelInput<'_, P>], &ExecutionContext) -> Result<T, ModelError> + Send,
    ) -> impl Future<Output = Result<T, ModelError>> + Send {
        execute_append(batch, policy, submit)
    }

    /// Runner guarantees nonempty uncached suffixes and idle contexts within capacity.
    /// Return exactly one ordered result per input, including shared GPU failures.
    fn generate(
        &self,
        batch: BatchIn<ModelInput<'_, P>>,
        policy: Arc<dyn SchedulingPolicy>,
    ) -> impl Future<Output = Vec<ModelOutput>> + Send;
}

/// Shared append submission boundary for batch and causal model implementations.
pub(super) async fn execute_append<P: KVPool, T: Send>(
    batch: &mut [ModelInput<'_, P>],
    policy: &Arc<dyn SchedulingPolicy>,
    submit: impl FnOnce(&mut [ModelInput<'_, P>], &ExecutionContext) -> Result<T, ModelError> + Send,
) -> Result<T, ModelError> {
    let output = with_context(|exec| {
        value((|| {
            let mut appends: Vec<_> = batch
                .iter_mut()
                .map(|input| {
                    let count = input.token_ids.len() - input.kv.ready_len();
                    (&mut *input.kv, count)
                })
                .collect();
            KVContext::prepare_append_batch(&mut appends, exec)?;
            submit(batch, exec)
        })())
    })
    .schedule(policy)?
    .await??;
    for input in batch {
        input.kv.commit_append()?;
    }
    Ok(output)
}

/// Token-ID stepping capability. The framework owns the greedy loop and stopping rules;
/// scalar demonstration models do not need to implement this contract.
pub trait CausalModel<P: KVPool>: Send + Sync {
    /// Safe submission entry for causal backends, including implementations outside
    /// engine. Reserve the suffix, run all required KV writes, await, then commit.
    fn execute_step<T: Send>(
        input: ModelInput<'_, P>,
        policy: &Arc<dyn SchedulingPolicy>,
        submit: impl FnOnce(&mut ModelInput<'_, P>, &ExecutionContext) -> Result<T, ModelError> + Send,
    ) -> impl Future<Output = Result<T, ModelError>> + Send {
        async move {
            let mut batch = [input];
            execute_append(&mut batch, policy, |inputs, exec| {
                submit(&mut inputs[0], exec)
            })
            .await
        }
    }

    /// Compute the uncached suffix, commit KV after GPU completion, and return greedy argmax.
    /// History must match this context, with a nonempty appended suffix.
    fn next_token(
        &self,
        token_ids: &[u32],
        kv: &mut KVContext<P>,
        policy: Arc<dyn SchedulingPolicy>,
    ) -> impl Future<Output = Result<u32, ModelError>> + Send;

    fn is_eos(&self, token: u32) -> bool;

    /// Lazily generate one owned token ID per poll-driven step, appending it to history
    /// before yielding. EOS is yielded once; EOS, the length limit, or an error ends
    /// the stream. No output Vec is allocated. The last yielded token has no KV yet.
    /// Dropping between items preserves resumable history/KV; dropping an in-flight
    /// step may leave a pending append, which must be released rather than reused.
    fn generate_greedy(
        &self,
        token_ids: &mut Vec<u32>,
        kv: &mut KVContext<P>,
        max_new_tokens: usize,
        policy: Arc<dyn SchedulingPolicy>,
    ) -> impl Stream<Item = Result<u32, ModelError>> + Send {
        stream::try_unfold(
            (token_ids, kv, max_new_tokens, policy),
            move |(token_ids, kv, remaining, policy)| async move {
                if remaining == 0 {
                    return Ok(None);
                }
                let next = self.next_token(token_ids, kv, policy.clone()).await?;
                token_ids.push(next);
                let remaining = if self.is_eos(next) { 0 } else { remaining - 1 };
                Ok(Some((next, (token_ids, kv, remaining, policy))))
            },
        )
    }
}

/// Compile-time pairing of an attention implementation and its supported pool layout.
pub trait AttentionBackend<P: KVPool> {
    /// Submit on exec; callers establish completion before consuming the tensor.
    /// Write views must be consumed or dropped before creating attention read views.
    ///
    /// ```compile_fail,E0308
    /// use engine::runner::{model::AttentionBackend, kv_pool::{NaiveKVPool, NaiveWriteView}};
    /// use cuda_async::device_operation::ExecutionContext;
    /// use cutile::tensor::Tensor;
    /// use std::sync::Arc;
    /// fn invalid_attention(backend: &impl AttentionBackend<NaiveKVPool>,
    ///     q: Arc<Tensor<f32>>, dst: NaiveWriteView<'_>, exec: &ExecutionContext) {
    ///     backend.forward(q, &[dst], &[0..1], exec).unwrap();
    /// }
    /// ```
    fn forward(
        &self,
        q: Arc<Tensor<f32>>,
        kv: &[P::ReadView<'_>],
        query_ranges: &[Range<usize>],
        exec: &ExecutionContext,
    ) -> Result<Arc<Tensor<f32>>, ModelError>;
}
