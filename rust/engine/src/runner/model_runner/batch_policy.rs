use std::num::NonZeroUsize;

use super::ActiveRequest;
use crate::runner::kv_pool::KVPool;

/// A selected request and its work for one complete forward; no state is copied.
pub(super) struct ScheduledRequest<'a, 'tok, P: KVPool> {
    pub(super) request: &'a mut ActiveRequest<'tok, P>,
    /// Exclusive token position to compute up to in this forward.
    pub(super) compute_end: usize,
}

pub(super) trait BatchPolicy {
    /// Borrow selected requests only; leave KV progress and request state unchanged.
    /// Each compute_end must be in (ready_len, tokens.len()]. Nonempty active
    /// input must produce work. Runner executes and settles it before the next plan.
    fn plan<'a, 'tok, P: KVPool>(
        &mut self,
        active: &'a mut [ActiveRequest<'tok, P>],
    ) -> Vec<ScheduledRequest<'a, 'tok, P>>;
}

/// Reserve one input token per decode, fill the remainder with prefill chunks,
/// then execute both together. With no prefill budget/work, run decode alone.
pub(super) struct MixedChunkPolicy {
    max_batch_tokens: NonZeroUsize,
}

impl MixedChunkPolicy {
    pub(super) fn new(max_batch_tokens: NonZeroUsize) -> Self {
        Self { max_batch_tokens }
    }
}

impl BatchPolicy for MixedChunkPolicy {
    fn plan<'a, 'tok, P: KVPool>(
        &mut self,
        active: &'a mut [ActiveRequest<'tok, P>],
    ) -> Vec<ScheduledRequest<'a, 'tok, P>> {
        // A one-token prompt/final chunk is still prefill: this turn enters
        // decode only after producing its first real output token.
        let mut decode_budget = active
            .iter()
            .filter(|request| request.generated_tokens > 0)
            .count()
            .min(self.max_batch_tokens.get());
        let mut prefill_budget = self.max_batch_tokens.get() - decode_budget;
        active
            .iter_mut()
            .filter_map(|request| {
                let ready = request.state.kv.ready_len();
                let count = if request.generated_tokens > 0 {
                    // decoding
                    if decode_budget == 0 {
                        return None;
                    }
                    decode_budget -= 1;
                    1
                } else {
                    // prefill
                    assert!(
                        request.state.tokens.len() > ready,
                        "prefill requests must have uncached input tokens"
                    );
                    if prefill_budget == 0 {
                        return None;
                    }
                    let count = (request.state.tokens.len() - ready).min(prefill_budget);
                    prefill_budget -= count;
                    count
                };
                Some(ScheduledRequest {
                    request,
                    compute_end: ready + count,
                })
            })
            .collect()
    }
}
