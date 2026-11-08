use std::{sync::Arc, task::Poll};

use cuda_async::{
    device_operation::{value, with_context, DeviceOp, ExecutionContext, GraphNode},
    scheduling_policies::SchedulingPolicy,
};
use futures::{
    future::{poll_fn, BoxFuture, Shared},
    FutureExt,
};
use tokio::sync::oneshot;

use super::{ModelError, NextToken};

/// One GPU submission's shared completion; awaiting it does not commit KV or copy tokens.
pub(crate) type GpuWorkCompletionFuture = Shared<BoxFuture<'static, Result<(), ModelError>>>;

/// Submission handle: output available after enqueue, paired with its GPU completion.
/// Models default to token predictions; the submission helper can return CPU handoff state.
pub struct SubmittedGpuWorkWithCompletion<T = Vec<Result<NextToken, ModelError>>> {
    pub(crate) output: T,
    pub(crate) gpu_completion: GpuWorkCompletionFuture,
    /// Producing stream identity supplied by CausalModel::enqueue_append_batch.
    pub(crate) compute_stream_id: Option<usize>,
}

impl<T> SubmittedGpuWorkWithCompletion<T> {
    /// Wrap output whose work is already complete; no GPU wait remains.
    pub fn from_completed_output(output: T) -> Self {
        Self {
            output,
            gpu_completion: futures::future::ready(Ok(())).boxed().shared(),
            compute_stream_id: None,
        }
    }

    /// Submit owned GPU work and return a handle containing output and completion.
    /// The first poll registers completion before CPU state is handed back to the caller.
    pub(super) async fn submit(
        policy: &Arc<dyn SchedulingPolicy>,
        submit: impl FnOnce(&ExecutionContext) -> T + Send + 'static,
    ) -> Result<Self, ModelError>
    where
        T: Send + 'static,
    {
        let (tx, mut rx) = oneshot::channel();
        let op = with_context(move |exec| {
            let _ = tx.send(submit(exec));
            value(())
        });
        let mut future = op
            .schedule(policy)?
            .map(|result| result.map_err(ModelError::from))
            .boxed();
        let first = poll_fn(|cx| Poll::Ready(future.as_mut().poll(cx))).await;
        let completion = match first {
            Poll::Pending => future.shared(),
            Poll::Ready(Ok(())) => futures::future::ready(Ok(())).boxed().shared(),
            // Never publish feedback from a submission already known to have failed.
            Poll::Ready(Err(error)) => return Err(error),
        };
        let output = rx
            .try_recv()
            .map_err(|_| ModelError::Execution("GPU submission did not execute".into()))?;
        Ok(Self {
            output,
            gpu_completion: completion,
            compute_stream_id: None,
        })
    }
}

/// Share kernel definitions between ordinary submission and checked scope recording.
/// Both modes borrow scratch; neither creates replacement tensor aliases.
pub(super) enum GpuKernelRecorder<'a> {
    Eager(&'a ExecutionContext),
    Capture(&'a cuda_async::cuda_graph::Scope),
}

impl GpuKernelRecorder<'_> {
    /// Eager callers must retain storage and prevent reuse until GPU completion.
    pub(super) unsafe fn record<N: GraphNode>(
        &self,
        op: N,
    ) -> Result<<N as DeviceOp>::Output, ModelError> {
        Ok(match self {
            Self::Eager(exec) => op.execute(exec)?,
            Self::Capture(scope) => scope.record(op)?,
        })
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::runner::{
        kv_pool::{KVContext, KVPool},
        model::{CausalModel, ModelInput, SimpleMLPEngine},
    };
    use cuda_async::scheduling_policies::SingleStream;
    use cutile::cuda_core::{Context, Stream};
    use std::{
        sync::{
            atomic::{AtomicBool, Ordering},
            mpsc,
        },
        time::Duration,
    };

    /// Hold later stream work until dropped; expiration is a test-harness failure.
    pub(in crate::runner::model) struct Gate {
        release: mpsc::Sender<()>,
        completed: Arc<AtomicBool>,
        pub(in crate::runner::model) expired: Arc<AtomicBool>,
    }

    impl Gate {
        pub(in crate::runner::model) fn arm(stream: &Arc<Stream>) -> anyhow::Result<Self> {
            let (release, wait) = mpsc::channel();
            let completed = Arc::new(AtomicBool::new(false));
            let expired = Arc::new(AtomicBool::new(false));
            let done = completed.clone();
            let failed = expired.clone();
            // Callback only waits/signals CPU state; it never calls CUDA APIs.
            stream.launch_host_function(move || {
                if wait.recv_timeout(Duration::from_secs(10)).is_ok() {
                    done.store(true, Ordering::Release);
                } else {
                    failed.store(true, Ordering::Release);
                }
            })?;
            Ok(Self {
                release,
                completed,
                expired,
            })
        }

        fn assert_held(&self) {
            assert!(!self.expired.load(Ordering::Acquire), "GPU gate expired");
            assert!(
                !self.completed.load(Ordering::Acquire),
                "GPU gate released early"
            );
        }
    }

    impl Drop for Gate {
        fn drop(&mut self) {
            // Cleanup must also unblock the stream during assertion unwinding.
            // A disconnected receiver is reported by the expired flag in the test.
            let _ = self.release.send(());
        }
    }

    /// Fail if a retained owner is destroyed before its queued GPU work completes.
    struct AssertGpuCompletedBeforeOwnerDrop {
        gpu_done: Arc<AtomicBool>,
        released: mpsc::Sender<()>,
    }

    impl Drop for AssertGpuCompletedBeforeOwnerDrop {
        fn drop(&mut self) {
            assert!(
                self.gpu_done.load(Ordering::Acquire),
                "owner released before GPU work completed"
            );
            let _ = self.released.send(());
        }
    }

    #[tokio::test]
    async fn selected_contexts_restore_before_completion_even_on_callback_error(
    ) -> anyhow::Result<()> {
        use crate::runner::kv_pool::{NaiveKVPool, NaiveKVPoolConfig};
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let device = Context::new(0)?;
        let policy: Arc<dyn SchedulingPolicy> = Arc::new(SingleStream::new(&device)?);
        let pool = NaiveKVPool::new(
            NaiveKVPoolConfig {
                max_rows: 2,
                num_slots: 4,
                max_seq_len: 4,
                head_dim: 4,
                ..Default::default()
            },
            &policy,
        )
        .await?;
        let initial = pool.metrics();
        // Initialize completion machinery before deliberately blocking the stream.
        SubmittedGpuWorkWithCompletion::submit(&policy, |_| ())
            .await?
            .gpu_completion
            .await?;
        for fail in [false, true] {
            let mut untouched = KVContext::new(pool.clone())?;
            let mut selected = KVContext::new(pool.clone())?;
            let submitted;
            let gate = Gate::arm(&policy.next_stream()?)?;
            let expired = gate.expired.clone();
            submitted = SimpleMLPEngine::enqueue_append_batch(
                &mut [
                    ModelInput {
                        token_ids: (&[3u32]).into(),
                        kv: &mut untouched,
                    },
                    ModelInput {
                        token_ids: (&[9u32]).into(),
                        kv: &mut selected,
                    },
                ],
                std::iter::once(1),
                &policy,
                move |inputs, _exec| -> Result<(), ModelError> {
                    assert_eq!(inputs.len(), 1);
                    assert_eq!(inputs[0].token_ids.host()?, &[9]);
                    assert_eq!(inputs[0].kv.pending_range(), Some(0..1));
                    if fail {
                        return Err(ModelError::Execution("injected callback failure".into()));
                    }
                    Ok(())
                },
            )
            .await?;
            gate.assert_held();
            assert_eq!(submitted.output.is_err(), fail);
            assert!(untouched.has_owned_pool_state() && selected.has_owned_pool_state());
            assert_eq!(untouched.pending_range(), None);
            assert_eq!(selected.pending_range(), Some(0..1));
            assert_eq!(selected.ready_len(), 0);
            drop(gate);
            submitted.gpu_completion.await?;
            assert!(!expired.load(Ordering::Acquire), "GPU gate expired");
            drop((untouched, selected));
            assert_eq!(pool.metrics(), initial);
        }
        Ok(())
    }

    #[tokio::test]
    async fn owned_handoff_and_completion_do_not_wait_for_the_next_submission() -> anyhow::Result<()>
    {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let device = Context::new(0)?;
        let policy: Arc<dyn SchedulingPolicy> = Arc::new(SingleStream::new(&device)?);
        let stream = policy.next_stream()?;
        // Initialize the library's completion machinery before blocking the stream.
        SubmittedGpuWorkWithCompletion::submit(&policy, |_| ())
            .await?
            .gpu_completion
            .await?;
        // Gates drop before these futures if an assertion unwinds.
        let mut completions = Vec::with_capacity(2);
        let first_gate = Gate::arm(&stream)?;
        let first_expired = first_gate.expired.clone();
        let first_owner = Arc::new(());
        let second_owner = Arc::new(());
        let owner = first_owner.clone();
        let SubmittedGpuWorkWithCompletion {
            output: state,
            gpu_completion: completion,
            ..
        } = SubmittedGpuWorkWithCompletion::submit(&policy, move |exec| {
            exec.retain(owner).unwrap();
            vec![42u32]
        })
        .await?;
        completions.push(completion);
        assert_eq!(state, [42]);
        first_gate.assert_held();
        assert_eq!(Arc::strong_count(&first_owner), 2);

        let second_gate = Gate::arm(&stream)?;
        let second_expired = second_gate.expired.clone();
        let owner = second_owner.clone();
        let submitted = SubmittedGpuWorkWithCompletion::submit(&policy, move |exec| {
            exec.retain(owner).unwrap()
        })
        .await?;
        completions.push(submitted.gpu_completion);
        second_gate.assert_held();
        drop(first_gate);
        tokio::time::timeout(Duration::from_secs(2), completions[0].clone()).await??;
        assert!(!first_expired.load(Ordering::Acquire));
        assert_eq!(Arc::strong_count(&first_owner), 1);
        assert_eq!(Arc::strong_count(&second_owner), 2);
        second_gate.assert_held();
        assert!(!stream.query()?);
        drop(second_gate);
        completions[1].clone().await?;
        assert!(!second_expired.load(Ordering::Acquire));
        assert_eq!(Arc::strong_count(&second_owner), 1);
        Ok(())
    }

    #[tokio::test]
    async fn partial_submission_and_cancelled_completion_keep_owners_until_safe(
    ) -> anyhow::Result<()> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let device = Context::new(0)?;
        let policy: Arc<dyn SchedulingPolicy> = Arc::new(SingleStream::new(&device)?);
        let owner = Arc::new(());
        let retained = owner.clone();
        let submitted = SubmittedGpuWorkWithCompletion::submit(&policy, move |exec| {
            exec.retain(retained).unwrap();
            Err::<(), _>(ModelError::Execution(
                "failure after resource registration".into(),
            ))
        })
        .await?;
        assert!(submitted.output.is_err());
        submitted.gpu_completion.await?;
        assert_eq!(Arc::strong_count(&owner), 1);

        let stream = policy.next_stream()?;
        let cancelled;
        let gate = Gate::arm(&stream)?;
        let expired = gate.expired.clone();
        let gpu_done = Arc::new(AtomicBool::new(false));
        let (released, released_rx) = mpsc::channel();
        let checked_owner = Arc::new(AssertGpuCompletedBeforeOwnerDrop {
            gpu_done: gpu_done.clone(),
            released,
        });
        let weak_owner = Arc::downgrade(&checked_owner);
        let done = gpu_done.clone();
        cancelled = SubmittedGpuWorkWithCompletion::submit(&policy, move |exec| {
            exec.retain(checked_owner).unwrap();
            // This queued callback models work that still uses the retained owner.
            exec.get_cuda_stream()
                .launch_host_function(move || {
                    done.store(true, Ordering::Release);
                })
                .unwrap();
        })
        .await?;
        gate.assert_held();
        assert!(!gpu_done.load(Ordering::Acquire));
        assert!(weak_owner.upgrade().is_some());

        let (started, started_rx) = mpsc::channel();
        let (finished, finished_rx) = mpsc::channel();
        let done = gpu_done.clone();
        let canceller = std::thread::spawn(move || {
            assert!(
                !done.load(Ordering::Acquire),
                "cancellation must begin in flight"
            );
            started.send(()).unwrap();
            drop(cancelled);
            finished.send(()).unwrap();
        });
        started_rx.recv_timeout(Duration::from_secs(2))?;
        // cuda-async 0.5 defers in-flight reclamation instead of draining on drop.
        finished_rx.recv_timeout(Duration::from_secs(2))?;
        gate.assert_held();
        assert!(
            weak_owner.upgrade().is_some(),
            "cancel freed an in-flight owner"
        );
        drop(gate);
        canceller
            .join()
            .expect("cancellation must not release owners early");
        released_rx.recv_timeout(Duration::from_secs(2))?;
        assert!(!expired.load(Ordering::Acquire), "GPU gate expired");
        assert!(gpu_done.load(Ordering::Acquire));
        assert!(weak_owner.upgrade().is_none());
        Ok(())
    }
}
