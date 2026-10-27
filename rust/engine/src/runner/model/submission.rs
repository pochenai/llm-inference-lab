use std::{sync::Arc, task::Poll};

use cuda_async::{
    device_operation::{value, with_context, DeviceOp, ExecutionContext},
    scheduling_policies::SchedulingPolicy,
};
use futures::{
    future::{poll_fn, BoxFuture, Shared},
    FutureExt,
};
use tokio::sync::oneshot;

use crate::runner::kv_pool::{KVContext, KVPool};

use super::{ModelError, ModelInput, ModelTokens, NextToken};

/// One GPU submission's shared completion; awaiting it does not commit KV or copy tokens.
pub(crate) type GpuWorkCompletionFuture = Shared<BoxFuture<'static, Result<(), ModelError>>>;

/// Submission handle: output available after enqueue, paired with its GPU completion.
/// Models default to token predictions; the submission helper can return CPU handoff state.
pub struct SubmittedGpuWorkWithCompletion<T = Vec<Result<NextToken, ModelError>>> {
    pub(crate) output: T,
    pub(crate) gpu_completion: GpuWorkCompletionFuture,
}

impl<T> SubmittedGpuWorkWithCompletion<T> {
    /// Wrap output whose work is already complete; no GPU wait remains.
    pub fn from_completed_output(output: T) -> Self {
        Self {
            output,
            gpu_completion: futures::future::ready(Ok(())).boxed().shared(),
        }
    }
}

/// Submit owned GPU work and return a handle containing output and completion.
/// The first poll registers completion before CPU state is handed back to the caller.
async fn submit_and_return_gpu_handle<T: Send + 'static>(
    policy: &Arc<dyn SchedulingPolicy>,
    submit: impl FnOnce(&ExecutionContext) -> T + Send + 'static,
) -> Result<SubmittedGpuWorkWithCompletion<T>, ModelError> {
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
    Ok(SubmittedGpuWorkWithCompletion {
        output,
        gpu_completion: completion,
    })
}

/// Private handoff; only the batch helper can extract output and restore its KV owners.
struct ModelInputsAfterGpuSubmission<P: KVPool, T> {
    inputs: Vec<(usize, ModelTokens<'static>, KVContext<P>)>,
    output: T,
}

/// Submit selected inputs and restore their KV owners before exposing output to the model.
/// The callback only borrows inputs; T must be owned. Selected indices must be unique and valid.
/// Outer errors/cancellation invalidate moved contexts; callback errors still restore ownership.
pub(super) async fn submit_model_batch_and_restore_kv_contexts<
    P: KVPool + 'static,
    T: Send + 'static,
>(
    batch: &mut [ModelInput<'_, P>],
    selected: impl Iterator<Item = usize> + Send,
    policy: &Arc<dyn SchedulingPolicy>,
    submit: impl FnOnce(&mut [ModelInput<'_, P>], &ExecutionContext) -> T + Send + 'static,
) -> Result<SubmittedGpuWorkWithCompletion<T>, ModelError> {
    // Own tokens and KV contexts for the 'static submission closure;
    // retain original batch indices so each context returns to its request.
    let mut owned: Vec<_> = selected
        .map(|index| {
            let input = &mut batch[index];
            let tokens = match &input.token_ids {
                ModelTokens::BorrowedHost(tokens) => ModelTokens::SharedHost {
                    tokens: Arc::new(tokens.to_vec()),
                    end: tokens.len(),
                },
                ModelTokens::SharedHost { tokens, end } => ModelTokens::SharedHost {
                    tokens: tokens.clone(),
                    end: *end,
                },
                ModelTokens::Device(token) => ModelTokens::Device(token.clone()),
            };
            (index, tokens, input.kv.take_context_for_gpu_submission())
        })
        .collect();
    // This await returns owned CPU state and a GPU completion handle;
    // GPU work may still be running when the handoff becomes available.
    let submitted = submit_and_return_gpu_handle(policy, move |exec| {
        let mut inputs: Vec<_> = owned
            .iter_mut()
            .map(|(_, tokens, kv)| ModelInput {
                token_ids: tokens.clone(),
                kv,
            })
            .collect();
        let output = submit(&mut inputs, exec);
        // End temporary borrows before handing the owned contexts back.
        drop(inputs);
        ModelInputsAfterGpuSubmission {
            inputs: owned,
            output,
        }
    })
    .await?;
    // restore kv context
    // Ownership returns now; gpu_completion still retains in-flight GPU resources.
    for (index, _, kv) in submitted.output.inputs {
        *batch[index].kv = kv;
    }
    Ok(SubmittedGpuWorkWithCompletion {
        output: submitted.output.output,
        gpu_completion: submitted.gpu_completion,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuda_async::scheduling_policies::SingleStream;
    use cutile::cuda_core::{Device, Stream};
    use std::{
        sync::{
            atomic::{AtomicBool, Ordering},
            mpsc,
        },
        time::Duration,
    };

    /// Hold later stream work until dropped; expiration is a test-harness failure.
    struct Gate {
        release: mpsc::Sender<()>,
        completed: Arc<AtomicBool>,
        expired: Arc<AtomicBool>,
    }

    impl Gate {
        fn arm(stream: &Arc<Stream>) -> anyhow::Result<Self> {
            let (release, wait) = mpsc::channel();
            let completed = Arc::new(AtomicBool::new(false));
            let expired = Arc::new(AtomicBool::new(false));
            let done = completed.clone();
            let failed = expired.clone();
            // Callback only waits/signals CPU state; it never calls CUDA APIs.
            unsafe {
                stream.launch_host_function(move || {
                    if wait.recv_timeout(Duration::from_secs(10)).is_ok() {
                        done.store(true, Ordering::Release);
                    } else {
                        failed.store(true, Ordering::Release);
                    }
                })?;
            }
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
    struct AssertGpuCompletedBeforeOwnerDrop(Arc<AtomicBool>);

    impl Drop for AssertGpuCompletedBeforeOwnerDrop {
        fn drop(&mut self) {
            assert!(
                self.0.load(Ordering::Acquire),
                "owner released before GPU work completed"
            );
        }
    }

    #[tokio::test]
    async fn selected_contexts_restore_before_completion_even_on_callback_error(
    ) -> anyhow::Result<()> {
        use crate::runner::kv_pool::{NaiveKVPool, NaiveKVPoolConfig};
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let device = Device::new(0)?;
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
        submit_and_return_gpu_handle(&policy, |_| ())
            .await?
            .gpu_completion
            .await?;
        for fail in [false, true] {
            let mut untouched = KVContext::new(pool.clone())?;
            let mut selected = KVContext::new(pool.clone())?;
            let submitted;
            let gate = Gate::arm(&policy.next_stream()?)?;
            let expired = gate.expired.clone();
            submitted = submit_model_batch_and_restore_kv_contexts(
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
                move |inputs, exec| -> Result<(), ModelError> {
                    assert_eq!(inputs.len(), 1);
                    assert_eq!(inputs[0].token_ids.host()?, &[9]);
                    KVContext::prepare_queued_append_batch(&mut [(&mut *inputs[0].kv, 1)], exec)?;
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
        let device = Device::new(0)?;
        let policy: Arc<dyn SchedulingPolicy> = Arc::new(SingleStream::new(&device)?);
        let stream = policy.next_stream()?;
        // Initialize the library's completion machinery before blocking the stream.
        submit_and_return_gpu_handle(&policy, |_| ())
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
        } = submit_and_return_gpu_handle(&policy, move |exec| {
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
        let submitted =
            submit_and_return_gpu_handle(&policy, move |exec| exec.retain(owner).unwrap()).await?;
        completions.push(submitted.gpu_completion);
        second_gate.assert_held();
        drop(first_gate);
        tokio::time::timeout(Duration::from_secs(2), completions[0].clone()).await??;
        assert!(!first_expired.load(Ordering::Acquire));
        assert_eq!(Arc::strong_count(&first_owner), 1);
        assert_eq!(Arc::strong_count(&second_owner), 2);
        second_gate.assert_held();
        assert!(!unsafe { stream.query()? });
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
        let device = Device::new(0)?;
        let policy: Arc<dyn SchedulingPolicy> = Arc::new(SingleStream::new(&device)?);
        let owner = Arc::new(());
        let retained = owner.clone();
        let submitted = submit_and_return_gpu_handle(&policy, move |exec| {
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
        let checked_owner = Arc::new(AssertGpuCompletedBeforeOwnerDrop(gpu_done.clone()));
        let weak_owner = Arc::downgrade(&checked_owner);
        let done = gpu_done.clone();
        cancelled = submit_and_return_gpu_handle(&policy, move |exec| {
            exec.retain(checked_owner).unwrap();
            // This queued callback models work that still uses the retained owner.
            unsafe {
                exec.get_cuda_stream()
                    .launch_host_function(move || {
                        done.store(true, Ordering::Release);
                    })
                    .unwrap();
            }
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
        // Give drop a chance to run while the test still owns the closed gate.
        // Both a blocking drain and safe deferred cleanup must keep the owner alive.
        if let Err(mpsc::RecvTimeoutError::Disconnected) =
            finished_rx.recv_timeout(Duration::from_millis(50))
        {
            panic!("cancellation panicked before the GPU gate was released");
        }
        gate.assert_held();
        assert!(
            weak_owner.upgrade().is_some(),
            "cancel freed an in-flight owner"
        );
        drop(gate);
        canceller
            .join()
            .expect("cancellation must not release owners early");
        assert!(!expired.load(Ordering::Acquire), "GPU gate expired");
        assert!(gpu_done.load(Ordering::Acquire));
        assert!(weak_owner.upgrade().is_none());
        Ok(())
    }
}
