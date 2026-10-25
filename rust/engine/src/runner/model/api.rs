use std::{future::Future, ops::Range, sync::Arc};

use cuda_async::{device_operation::*, scheduling_policies::SchedulingPolicy};
use cutile::tensor::Tensor;
use tokenizers::Tokenizer;

use crate::runner::kv_pool::{KVContext, KVPool};

use super::ModelError;

pub struct ModelInput<'a, P: KVPool> {
    /// Full accepted token history; only token_ids[kv.ready_len()..] is computed.
    pub token_ids: &'a [u32],
    pub kv: &'a mut KVContext<P>,
}

pub trait CausalModel<P: KVPool>: Send + Sync {
    /// Borrow the matching tokenizer; Runner controls encoding and decoding.
    fn tokenizer(&self) -> &Tokenizer;

    /// CPU validation of newly encoded tokens, before Runner admission.
    fn validate_input(&self, _token_ids: &[u32]) -> Result<(), ModelError> {
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
        async move {
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
    }

    /// Compute one next token per input in order, including per-input validation/GPU errors.
    /// Valid inputs in a packed forward must use one compatible KV pool.
    /// Reject pending contexts, empty suffixes, invalid IDs, and exceeded capacity.
    /// Success commits KV for the full input history before returning its next token.
    fn next_tokens(
        &self,
        batch: &mut [ModelInput<'_, P>],
        policy: Arc<dyn SchedulingPolicy>,
    ) -> impl Future<Output = Vec<Result<u32, ModelError>>> + Send;

    fn is_eos(&self, _token: u32) -> bool {
        false
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
