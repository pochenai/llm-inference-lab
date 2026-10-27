use std::sync::atomic::{AtomicU64, Ordering};

use metrics::{Counter, Gauge, Histogram};
use serde::Serialize;

/// Logical scratch allocation bytes, excluding allocator padding/reservations,
/// weights, KV, uploads and pinned readbacks. In-use includes GPU/escaped-handle leases.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct WorkspaceMetrics {
    /// One label per model lifetime, shared by its execution/workspace instruments.
    pub model_instance: u64,
    pub in_use_bytes: usize,
    pub peak_in_use_bytes: usize,
    pub idle_bytes: usize,
    pub peak_idle_bytes: usize,
    pub resident_bytes: usize,
    pub peak_resident_bytes: usize,
    pub in_use_workspaces: usize,
    pub idle_workspaces: usize,
    pub peak_forward_bytes: usize,
}

impl WorkspaceMetrics {
    pub(super) fn checkout(&mut self, bytes: usize) {
        self.in_use_bytes += bytes;
        self.in_use_workspaces += 1;
        self.peak_in_use_bytes = self.peak_in_use_bytes.max(self.in_use_bytes);
        self.peak_forward_bytes = self.peak_forward_bytes.max(bytes);
        self.update_peaks();
    }

    pub(super) fn update_peaks(&mut self) {
        self.resident_bytes = self.in_use_bytes + self.idle_bytes;
        self.peak_resident_bytes = self.peak_resident_bytes.max(self.resident_bytes);
        self.peak_idle_bytes = self.peak_idle_bytes.max(self.idle_bytes);
    }
}

/// Handles are registered once, after the application installs its recorder.
/// Instance labels isolate multiple live models; request/session IDs are never labels.
pub(super) struct Instruments {
    prefill: PhaseInstruments,
    decode: PhaseInstruments,
    failed_steps: Counter,
    batch_duration: Histogram,
    batch_size: Histogram,
    pub(super) readback_duration: Histogram,
}

struct PhaseInstruments {
    steps: Counter,
    tokens: Counter,
    step_wall_duration: Histogram,
    host_submit_duration: Histogram,
}

impl PhaseInstruments {
    fn new(instance: &str, phase: &'static str) -> Self {
        Self {
            steps: metrics::counter!("qwen2_steps_total", "model_instance" => instance.to_owned(), "phase" => phase),
            tokens: metrics::counter!("qwen2_input_tokens_total", "model_instance" => instance.to_owned(), "phase" => phase),
            step_wall_duration: metrics::histogram!("qwen2_step_wall_duration_seconds", "model_instance" => instance.to_owned(), "phase" => phase),
            host_submit_duration: metrics::histogram!("qwen2_host_submit_duration_seconds", "model_instance" => instance.to_owned(), "phase" => phase),
        }
    }
}

impl Instruments {
    pub(super) fn new(instance: u64) -> Self {
        let instance = instance.to_string();
        Self {
            prefill: PhaseInstruments::new(&instance, "prefill"),
            decode: PhaseInstruments::new(&instance, "decode"),
            failed_steps: metrics::counter!("qwen2_failed_steps_total", "model_instance" => instance.clone()),
            batch_duration: metrics::histogram!("qwen2_batch_duration_seconds", "model_instance" => instance.clone()),
            batch_size: metrics::histogram!("qwen2_batch_size", "model_instance" => instance.clone()),
            readback_duration: metrics::histogram!("qwen2_readback_wall_duration_seconds", "model_instance" => instance),
        }
    }

    pub(super) fn step(
        &self,
        prefill: bool,
        tokens: usize,
        step_wall_seconds: f64,
        host_submit_seconds: Option<f64>,
    ) {
        // No completed step output is available on failure; do not fabricate a duration.
        let Some(host_submit_seconds) = host_submit_seconds else {
            self.failed_steps.increment(1);
            return;
        };
        let phase = if prefill { &self.prefill } else { &self.decode };
        phase.steps.increment(1);
        phase.tokens.increment(tokens as u64);
        phase.step_wall_duration.record(step_wall_seconds);
        phase.host_submit_duration.record(host_submit_seconds);
    }

    pub(super) fn batch(&self, size: usize, batch_wall_seconds: f64) {
        self.batch_size.record(size as f64);
        self.batch_duration.record(batch_wall_seconds);
    }
}

pub(super) fn next_instance() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

pub(super) struct WorkspaceInstruments {
    in_use_bytes: Gauge,
    peak_in_use_bytes: Gauge,
    idle_bytes: Gauge,
    peak_idle_bytes: Gauge,
    resident_bytes: Gauge,
    peak_resident_bytes: Gauge,
    in_use_workspaces: Gauge,
    idle_workspaces: Gauge,
    peak_forward_bytes: Gauge,
    pub(super) cache_hits: Counter,
    pub(super) cache_misses: Counter,
    pub(super) workspace_allocations: Counter,
    pub(super) checkout_failures: Counter,
    pub(super) evictions: Counter,
    forward_bytes: Histogram,
}

impl WorkspaceInstruments {
    pub(super) fn new(instance: u64) -> Self {
        let instance = instance.to_string();
        Self {
            in_use_bytes: metrics::gauge!("qwen2_workspace_in_use_bytes", "model_instance" => instance.clone()),
            peak_in_use_bytes: metrics::gauge!("qwen2_workspace_peak_in_use_bytes", "model_instance" => instance.clone()),
            idle_bytes: metrics::gauge!("qwen2_workspace_idle_bytes", "model_instance" => instance.clone()),
            peak_idle_bytes: metrics::gauge!("qwen2_workspace_peak_idle_bytes", "model_instance" => instance.clone()),
            resident_bytes: metrics::gauge!("qwen2_workspace_resident_bytes", "model_instance" => instance.clone()),
            peak_resident_bytes: metrics::gauge!("qwen2_workspace_peak_resident_bytes", "model_instance" => instance.clone()),
            in_use_workspaces: metrics::gauge!("qwen2_workspace_in_use_workspaces", "model_instance" => instance.clone()),
            idle_workspaces: metrics::gauge!("qwen2_workspace_idle_workspaces", "model_instance" => instance.clone()),
            peak_forward_bytes: metrics::gauge!("qwen2_workspace_peak_forward_bytes", "model_instance" => instance.clone()),
            cache_hits: metrics::counter!("qwen2_workspace_cache_hits_total", "model_instance" => instance.clone()),
            cache_misses: metrics::counter!("qwen2_workspace_cache_misses_total", "model_instance" => instance.clone()),
            workspace_allocations: metrics::counter!("qwen2_workspace_allocations_total", "model_instance" => instance.clone()),
            checkout_failures: metrics::counter!("qwen2_workspace_checkout_failures_total", "model_instance" => instance.clone()),
            evictions: metrics::counter!("qwen2_workspace_evictions_total", "model_instance" => instance.clone()),
            forward_bytes: metrics::histogram!("qwen2_workspace_forward_bytes", "model_instance" => instance),
        }
    }

    /// Called while holding the pool lock so gauge publication preserves transition order.
    pub(super) fn publish(&self, snapshot: &WorkspaceMetrics) {
        self.in_use_bytes.set(snapshot.in_use_bytes as f64);
        self.peak_in_use_bytes
            .set(snapshot.peak_in_use_bytes as f64);
        self.idle_bytes.set(snapshot.idle_bytes as f64);
        self.peak_idle_bytes.set(snapshot.peak_idle_bytes as f64);
        self.resident_bytes.set(snapshot.resident_bytes as f64);
        self.peak_resident_bytes
            .set(snapshot.peak_resident_bytes as f64);
        self.in_use_workspaces
            .set(snapshot.in_use_workspaces as f64);
        self.idle_workspaces.set(snapshot.idle_workspaces as f64);
        self.peak_forward_bytes
            .set(snapshot.peak_forward_bytes as f64);
    }

    pub(super) fn checkout(&self, bytes: usize) {
        self.forward_bytes.record(bytes as f64);
    }

    /// Peak gauges/counters remain historical; current resource gauges end at zero.
    pub(super) fn close(&self) {
        self.in_use_bytes.set(0.0);
        self.idle_bytes.set(0.0);
        self.resident_bytes.set(0.0);
        self.in_use_workspaces.set(0.0);
        self.idle_workspaces.set(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrent_steps_preserve_counters_and_individual_observations() {
        use metrics_util::debugging::{DebugValue, DebuggingRecorder};
        let recorder = DebuggingRecorder::new();
        let instruments = metrics::with_local_recorder(&recorder, || Instruments::new(99));
        std::thread::scope(|threads| {
            for seconds in [0.1, 0.2] {
                let instruments = &instruments;
                threads.spawn(move || instruments.step(true, 32, seconds, Some(0.01)));
            }
        });
        let values = recorder.snapshotter().snapshot().into_vec();
        let get = |name: &str| {
            &values
                .iter()
                .find(|(key, ..)| {
                    key.key().name() == name
                        && key
                            .key()
                            .labels()
                            .any(|label| label.key() == "phase" && label.value() == "prefill")
                })
                .unwrap()
                .3
        };
        assert_eq!(get("qwen2_steps_total"), &DebugValue::Counter(2));
        assert_eq!(get("qwen2_input_tokens_total"), &DebugValue::Counter(64));
        let DebugValue::Histogram(samples) = get("qwen2_step_wall_duration_seconds") else {
            panic!("histogram")
        };
        let mut samples = samples.clone();
        samples.sort();
        assert_eq!(samples, vec![0.1_f64, 0.2_f64]);
    }

    #[test]
    fn records_counter_and_histogram_observations_without_an_exporter() {
        use metrics_util::debugging::{DebugValue, DebuggingRecorder};
        let recorder = DebuggingRecorder::new();
        // Register locally; cached handles must still work after this scope ends.
        let instruments = metrics::with_local_recorder(&recorder, || Instruments::new(42));
        instruments.step(true, 32, 0.2, Some(0.01));
        instruments.step(true, 16, 0.1, Some(0.02));
        instruments.step(false, 1, 0.03, Some(0.005));
        instruments.step(false, 0, 0.5, None);
        instruments.batch(2, 0.3);
        let (workspace, other) = metrics::with_local_recorder(&recorder, || {
            (WorkspaceInstruments::new(42), WorkspaceInstruments::new(43))
        });
        workspace.publish(&WorkspaceMetrics {
            idle_bytes: 128,
            resident_bytes: 128,
            ..Default::default()
        });
        other.publish(&WorkspaceMetrics {
            idle_bytes: 512,
            resident_bytes: 512,
            ..Default::default()
        });
        workspace.close();
        let values = recorder.snapshotter().snapshot().into_vec();
        let get = |name: &str, phase: Option<&str>| {
            &values
                .iter()
                .find(|(key, ..)| {
                    key.key().name() == name
                        && key
                            .key()
                            .labels()
                            .any(|l| l.key() == "model_instance" && l.value() == "42")
                        && phase.is_none_or(|phase| {
                            key.key()
                                .labels()
                                .any(|l| l.key() == "phase" && l.value() == phase)
                        })
                })
                .unwrap()
                .3
        };
        assert_eq!(
            get("qwen2_steps_total", Some("prefill")),
            &DebugValue::Counter(2)
        );
        assert_eq!(
            get("qwen2_input_tokens_total", Some("prefill")),
            &DebugValue::Counter(48)
        );
        assert_eq!(
            get("qwen2_failed_steps_total", None),
            &DebugValue::Counter(1)
        );
        assert_eq!(
            get("qwen2_step_wall_duration_seconds", Some("prefill")),
            &DebugValue::Histogram(vec![0.2.into(), 0.1.into()])
        );
        assert_eq!(
            get("qwen2_step_wall_duration_seconds", Some("decode")),
            &DebugValue::Histogram(vec![0.03.into()])
        );
        assert_eq!(
            get("qwen2_host_submit_duration_seconds", Some("prefill")),
            &DebugValue::Histogram(vec![0.01.into(), 0.02.into()])
        );
        assert_eq!(
            get("qwen2_host_submit_duration_seconds", Some("decode")),
            &DebugValue::Histogram(vec![0.005.into()])
        );
        assert_eq!(
            get("qwen2_batch_size", None),
            &DebugValue::Histogram(vec![2.0.into()])
        );
        assert_eq!(
            get("qwen2_workspace_idle_bytes", None),
            &DebugValue::Gauge(0.0.into())
        );
        let (_, _, _, other_idle) = values
            .iter()
            .find(|(key, ..)| {
                key.key().name() == "qwen2_workspace_idle_bytes"
                    && key
                        .key()
                        .labels()
                        .any(|label| label.key() == "model_instance" && label.value() == "43")
            })
            .unwrap();
        assert_eq!(other_idle, &DebugValue::Gauge(512.0.into()));
    }
}
