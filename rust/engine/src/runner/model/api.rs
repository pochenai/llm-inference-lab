use std::{future::Future, ops::Range, sync::Arc};

use cuda_async::{device_operation::*, scheduling_policies::SchedulingPolicy};
use cutile::tensor::Tensor;
use tokenizers::Tokenizer;

use crate::runner::kv_pool::{KVContext, KVPool};

use super::{ModelError, ModelTokens, NextToken};

pub struct ModelInput<'a, P: KVPool> {
    /// A host history prefix through this step's end, or one GPU-resident decode
    /// token. Host inputs compute only [kv.ready_len()..]; device inputs append one.
    pub token_ids: ModelTokens<'a>,
    pub kv: &'a mut KVContext<P>,
}

impl<P: KVPool> ModelInput<'_, P> {
    /// Logical history end after this input is computed, independent of CPU readback.
    pub fn sequence_len(&self) -> usize {
        match &self.token_ids {
            ModelTokens::Host(tokens) => tokens.len(),
            ModelTokens::Device(_) => self.kv.ready_len() + 1,
        }
    }

    /// Invalid host histories return zero; model validation reports the input error.
    pub fn uncached_len(&self) -> usize {
        self.sequence_len().saturating_sub(self.kv.ready_len())
    }
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
            let suffix =
                &input.token_ids.host().expect("validated host input")[input.kv.ready_len()..];
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
                            let count = input.uncached_len();
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

    /// Return one ordered token result after computation and successful KV commit.
    /// Device results defer CPU reads; ready CPU results use NextToken::Host.
    /// Validate pending contexts, input bounds/IDs, and compatible pool layout.
    fn forward(
        &self,
        batch: &mut [ModelInput<'_, P>],
        policy: Arc<dyn SchedulingPolicy>,
    ) -> impl Future<Output = Vec<Result<NextToken, ModelError>>> + Send;

    /// Convenience API: run forward, then await each token's CPU value.
    fn next_tokens(
        &self,
        batch: &mut [ModelInput<'_, P>],
        policy: Arc<dyn SchedulingPolicy>,
    ) -> impl Future<Output = Vec<Result<u32, ModelError>>> + Send {
        async move {
            let outputs = self.forward(batch, policy).await;
            let mut values = Vec::with_capacity(outputs.len());
            for output in outputs {
                values.push(match output {
                    Ok(token) => token.read().await,
                    Err(error) => Err(error),
                });
            }
            values
        }
    }

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
