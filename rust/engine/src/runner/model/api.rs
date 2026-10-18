use std::{future::Future, ops::Range, sync::Arc};

use cuda_async::{device_operation::*, scheduling_policies::SchedulingPolicy};
use cutile::tensor::Tensor;
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

    /// Reserve each uncached suffix and enqueue mappings before calling submit on the
    /// same exec. The callback must submit every KV cache update required by this
    /// forward; caller should ensure every layer call this function.
    /// The framework does not inspect network topology or verify cache coverage.
    /// Await successful GPU completion before committing KV. The callback must not
    /// prepare appends again. Cancellation/failure leaves them uncommitted.
    fn execute_append_batch<T: Send>(
        batch: &mut [ModelInput<'_, P>],
        policy: &Arc<dyn SchedulingPolicy>,
        submit: impl FnOnce(&mut [ModelInput<'_, P>], &ExecutionContext) -> Result<T, ModelError> + Send,
    ) -> impl Future<Output = Result<T, ModelError>> + Send {
        async move {
            let output = with_context(|exec| {
                value((|| {
                    for input in batch.iter_mut() {
                        let count = input.token_ids.len() - input.kv.ready_len();
                        input.kv.prepare_append(count, exec)?;
                    }
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

    /// Runner guarantees nonempty uncached suffixes and idle contexts within capacity.
    /// Return exactly one ordered result per input, including shared GPU failures.
    fn generate(
        &self,
        batch: BatchIn<ModelInput<'_, P>>,
        policy: Arc<dyn SchedulingPolicy>,
    ) -> impl Future<Output = Vec<ModelOutput>> + Send;
}

/// Compile-time pairing of an attention implementation and its supported pool layout.
pub trait AttentionBackend<P: KVPool> {
    /// Submit on exec; callers establish completion before consuming the tensor.
    fn forward(
        &self,
        q: Arc<Tensor<f32>>,
        kv: &[P::LayerView<'_>],
        query_ranges: &[Range<usize>],
        exec: &ExecutionContext,
    ) -> Result<Arc<Tensor<f32>>, ModelError>;
}
