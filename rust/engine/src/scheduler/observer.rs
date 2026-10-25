use std::{sync::Arc, time::Duration};

use metrics::{Counter, Gauge, Histogram};

use crate::{
    error::EngineError,
    types::{LLMResponse, RequestId},
};

/// Server-side timing ends at Runner completion, independent of consumer polling.
#[derive(Clone, Copy, Debug)]
pub struct RequestStats {
    pub elapsed: Duration,
    pub time_to_first_text: Option<Duration>,
    /// None on cancellation/transport loss before a Runner completion was received.
    pub generated_tokens: Option<usize>,
}

impl RequestStats {
    /// Request output TPS includes queueing, prefill, generation and text decoding.
    /// This is distinct from model-only decode TPS. Zero-duration samples have no rate.
    pub fn tokens_per_second(&self) -> Option<f64> {
        let seconds = self.elapsed.as_secs_f64();
        self.generated_tokens
            .filter(|_| seconds > 0.0)
            .map(|tokens| tokens as f64 / seconds)
    }
}

/// Synchronous, nonblocking notifications; callbacks cannot change scheduling.
/// Called outside pending-map locks. Admission is followed by exactly one finish,
/// including errors/shutdown; dropping a response stream does not finish execution.
/// Callbacks must not panic or perform blocking work.
pub trait SchedulerObserver: Send + Sync {
    fn request_received(&self, _request_id: RequestId) {}

    /// first_text is Some only on the first nonempty delta, never an empty fallback.
    fn text_delta(&self, _response: &LLMResponse, _first_text: Option<Duration>) {}

    fn request_finished(
        &self,
        _request_id: RequestId,
        _stats: &RequestStats,
        _result: &Result<(), EngineError>,
    ) {
    }
}

/// Collections compose observers without adding forwarding state to requests.
impl<T: SchedulerObserver> SchedulerObserver for [T] {
    fn request_received(&self, request_id: RequestId) {
        for observer in self {
            observer.request_received(request_id);
        }
    }

    fn text_delta(&self, response: &LLMResponse, first_text: Option<Duration>) {
        for observer in self {
            observer.text_delta(response, first_text);
        }
    }

    fn request_finished(
        &self,
        request_id: RequestId,
        stats: &RequestStats,
        result: &Result<(), EngineError>,
    ) {
        for observer in self {
            observer.request_finished(request_id, stats, result);
        }
    }
}

impl<T: SchedulerObserver + ?Sized> SchedulerObserver for Arc<T> {
    fn request_received(&self, request_id: RequestId) {
        (**self).request_received(request_id);
    }
    fn text_delta(&self, response: &LLMResponse, first_text: Option<Duration>) {
        (**self).text_delta(response, first_text);
    }
    fn request_finished(
        &self,
        request_id: RequestId,
        stats: &RequestStats,
        result: &Result<(), EngineError>,
    ) {
        (**self).request_finished(request_id, stats, result);
    }
}

/// A completion-only closure composes with collection and Arc observers directly.
impl<F> SchedulerObserver for F
where
    F: Fn(RequestId, &RequestStats, &Result<(), EngineError>) + Send + Sync,
{
    fn request_finished(
        &self,
        id: RequestId,
        stats: &RequestStats,
        result: &Result<(), EngineError>,
    ) {
        self(id, stats, result);
    }
}

/// Install a metrics recorder before constructing this observer. No recorder is
/// installed here; handles are cached and no request/session IDs become labels.
pub struct MetricsObserver {
    requests: Counter,
    failures: Counter,
    in_flight: Gauge,
    success_duration: Histogram,
    error_duration: Histogram,
    first_text: Histogram,
    generated_tokens: Counter,
    output_bytes: Counter,
    output_tps: Histogram,
}

impl Default for MetricsObserver {
    fn default() -> Self {
        Self {
            requests: metrics::counter!("engine_requests_total"),
            failures: metrics::counter!("engine_request_failures_total"),
            in_flight: metrics::gauge!("engine_requests_in_flight"),
            success_duration: metrics::histogram!("engine_request_duration_seconds", "outcome" => "success"),
            error_duration: metrics::histogram!("engine_request_duration_seconds", "outcome" => "error"),
            first_text: metrics::histogram!("engine_time_to_first_text_seconds"),
            generated_tokens: metrics::counter!("engine_generated_tokens_total"),
            output_bytes: metrics::counter!("engine_output_text_bytes_total"),
            output_tps: metrics::histogram!("engine_request_output_tokens_per_second"),
        }
    }
}

impl SchedulerObserver for MetricsObserver {
    fn request_received(&self, _request_id: RequestId) {
        self.requests.increment(1);
        self.in_flight.increment(1.0);
    }

    fn text_delta(&self, response: &LLMResponse, first_text: Option<Duration>) {
        self.output_bytes.increment(response.output.len() as u64);
        if let Some(elapsed) = first_text {
            self.first_text.record(elapsed.as_secs_f64());
        }
    }

    fn request_finished(
        &self,
        _request_id: RequestId,
        stats: &RequestStats,
        result: &Result<(), EngineError>,
    ) {
        self.in_flight.decrement(1.0);
        let duration = if result.is_ok() {
            &self.success_duration
        } else {
            &self.error_duration
        };
        duration.record(stats.elapsed.as_secs_f64());
        if let Some(tokens) = stats.generated_tokens {
            self.generated_tokens.increment(tokens as u64);
        }
        if result.is_err() {
            self.failures.increment(1);
        } else if let Some(tps) = stats.tokens_per_second() {
            self.output_tps.record(tps);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_tps_uses_real_tokens_and_rejects_unknown_or_zero_duration() {
        let mut stats = RequestStats {
            elapsed: Duration::from_secs(2),
            time_to_first_text: None,
            generated_tokens: Some(5),
        };
        assert_eq!(stats.tokens_per_second(), Some(2.5));
        stats.generated_tokens = Some(0);
        assert_eq!(stats.tokens_per_second(), Some(0.0));
        stats.generated_tokens = None;
        assert_eq!(stats.tokens_per_second(), None);
        stats.generated_tokens = Some(5);
        stats.elapsed = Duration::ZERO;
        assert_eq!(stats.tokens_per_second(), None);
    }
}
