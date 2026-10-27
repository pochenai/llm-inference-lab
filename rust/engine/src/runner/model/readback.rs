use std::{
    borrow::Borrow,
    sync::{Arc, Mutex},
    time::Instant,
};

use cuda_async::{
    device_operation::{value, with_context, DeviceOp, ExecutionContext},
    error::DeviceError,
    scheduling_policies::{SchedulingPolicy, SingleStream},
};
use cutile::{
    cuda_core::{memcpy_dtoh_async, CudaContext, DType, Device, DeviceCopy, PinnedHostBuffer},
    tensor::Tensor,
};

use metrics::Histogram;

use super::{submission::GpuWorkCompletionFuture, GpuTokenPredictionHandle, ModelError};

const MAX_CACHED_READBACKS: usize = 16;

pub(super) type Readback<T> = Arc<PinnedHostBuffer<T>>;
/// Own pinned buffers and enforce reuse only when no GPU copy or CPU reader holds them.
pub(super) struct ReadbackCache<T: DeviceCopy> {
    buffers: Mutex<Vec<Readback<T>>>,
}

impl<T: DeviceCopy> Default for ReadbackCache<T> {
    fn default() -> Self {
        Self {
            buffers: Mutex::new(Vec::new()),
        }
    }
}

/// Reusable sampled-token D2H cache, independent stream, and readback instrumentation.
/// GPU predictions share one lazy CPU copy; compute-stream consumers never wait for it.
pub(super) struct GpuTokenReadback {
    cache: Arc<ReadbackCache<i32>>,
    policy: Arc<dyn SchedulingPolicy>,
    duration: Histogram,
}

impl GpuTokenReadback {
    pub(super) fn new(device: &Arc<Device>, duration: Histogram) -> Result<Self, DeviceError> {
        Ok(Self {
            cache: Arc::default(),
            policy: Arc::new(SingleStream::new(device)?),
            duration,
        })
    }

    /// Enqueue on the caller's stream; host storage is readable only after completion.
    pub(super) unsafe fn enqueue(
        &self,
        values: Tensor<i32>,
        exec: &ExecutionContext,
    ) -> Result<Readback<i32>, DeviceError> {
        self.cache.enqueue(values, exec)
    }

    /// Return GPU prediction handles immediately; CPU reads await the producer then D2H.
    /// Positions and producer stream identify the next forward's logical input/dependency.
    pub(super) fn predictions(
        &self,
        values: Tensor<i32>,
        positions: Vec<usize>,
        vocab_size: usize,
        completion: GpuWorkCompletionFuture,
        producer_stream_id: Option<usize>,
    ) -> impl Iterator<Item = GpuTokenPredictionHandle> + 'static {
        let values = Arc::new(values);
        let source = values.clone();
        let cache = self.cache.clone();
        let policy = self.policy.clone();
        let duration = self.duration.clone();
        let host = async move {
            completion.await?;
            let started = Instant::now();
            let result: Result<Readback<i32>, ModelError> = async {
                let output = with_context(|exec| {
                    // SAFETY: producer completed; D2H retains both source and destination.
                    value(unsafe { cache.enqueue(source, exec) })
                })
                .schedule(&policy)?
                .await??;
                Ok(output)
            }
            .await;
            let readback_wall_seconds = started.elapsed().as_secs_f64();
            duration.record(readback_wall_seconds);
            tracing::debug!(target: "engine::model", event = "readback_completed",
                readback_wall_seconds, success = result.is_ok());
            result
        };
        GpuTokenPredictionHandle::batch(values, host, vocab_size, positions.into_iter()).map(
            move |mut token| {
                token.producer_stream_id = producer_stream_id;
                token
            },
        )
    }
}

impl<T: DeviceCopy + DType> ReadbackCache<T> {
    /// Enqueue D2H without waiting; read the returned buffer only after outer completion.
    /// Cache-only ownership proves a buffer has neither an in-flight copy nor a reader.
    /// Source and destination stay retained by exec even if the consumer is cancelled.
    pub(super) unsafe fn enqueue(
        &self,
        tensor: impl Borrow<Tensor<T>> + Send + 'static,
        exec: &ExecutionContext,
    ) -> Result<Readback<T>, DeviceError> {
        let count = tensor.borrow().size();
        let source = tensor.borrow().device_pointer().cu_deviceptr();
        let output = {
            let mut idle = self.buffers.lock().unwrap();
            // Reuse only a matching length/device with exclusive cache ownership; GPU/CPU users prevent reuse.
            if let Some(buffer) = idle.iter_mut().find_map(|entry| {
                let buffer = Arc::get_mut(entry)?;
                (buffer.len() == count && buffer.context().ordinal() == exec.get_device_id())
                    .then(|| entry.clone())
            }) {
                buffer
            } else {
                let buffer = Arc::new(PinnedHostBuffer::zeroed(
                    &CudaContext::new(exec.get_device_id())?,
                    count,
                )?);
                // This limit counts cached entries, including idle buffers, not active users.
                if idle.len() < MAX_CACHED_READBACKS {
                    // cache it
                    idle.push(buffer.clone());
                } else if let Some(entry) =
                    idle.iter_mut().find(|entry| Arc::strong_count(entry) == 1)
                {
                    // only cache hold it, which means it's not in use.
                    *entry = buffer.clone();
                }
                // If every entry still has a GPU/CPU user, use this buffer without caching it.
                buffer
            }
        };
        exec.retain(tensor)?;
        exec.retain(output.clone())?;
        memcpy_dtoh_async(
            output.as_ptr().cast_mut(),
            source,
            count,
            exec.get_cuda_stream(),
        )?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuda_async::{
        device_context::global_policy,
        device_operation::{value, with_context, DeviceOp},
    };
    use cutile::api;

    #[tokio::test]
    async fn full_cache_replaces_idle_shapes_and_keeps_live_readers() -> Result<(), DeviceError> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = global_policy(0)?;
        let cache = ReadbackCache::default();
        let new_count = MAX_CACHED_READBACKS * 2;
        let mut previous = None;
        for count in (1..=MAX_CACHED_READBACKS).chain([new_count, new_count]) {
            let output = with_context(|exec| {
                value((|| -> Result<_, DeviceError> {
                    // SAFETY: upload and D2H share exec; host reads follow completion.
                    unsafe {
                        let tensor =
                            api::copy_host_vec_to_device(&Arc::new(vec![count as i32; count]))
                                .execute(exec)?;
                        cache.enqueue(tensor, exec)
                    }
                })())
            })
            .schedule(&policy)?
            .await??;
            assert_eq!(output.as_slice(), vec![count as i32; count]);
            if count == new_count {
                let entries = cache.buffers.lock().unwrap();
                assert_eq!(entries.len(), MAX_CACHED_READBACKS);
                assert!(
                    entries.iter().any(|entry| Arc::ptr_eq(entry, &output)),
                    "a new length must enter the full cache"
                );
                let pointer = output.as_ptr();
                if let Some(previous) = previous {
                    assert_eq!(pointer, previous);
                }
                previous = Some(pointer);
            }
        }
        // Keep every cached buffer borrowed by a CPU reader while another copy runs.
        let readers = cache.buffers.lock().unwrap().clone();
        let overflow = with_context(|exec| {
            value((|| -> Result<_, DeviceError> {
                // SAFETY: exec owns the copy; existing readers must remain untouched.
                unsafe {
                    let tensor =
                        api::copy_host_vec_to_device(&Arc::new(vec![99i32; new_count + 1]))
                            .execute(exec)?;
                    let output = cache.enqueue(tensor, exec)?;
                    let entries = cache.buffers.lock().unwrap();
                    assert_eq!(entries.len(), MAX_CACHED_READBACKS);
                    assert!(readers
                        .iter()
                        .zip(entries.iter())
                        .all(|(a, b)| Arc::ptr_eq(a, b)));
                    assert!(!entries.iter().any(|entry| Arc::ptr_eq(entry, &output)));
                    Ok(output)
                }
            })())
        })
        .schedule(&policy)?
        .await??;
        assert_eq!(overflow.as_slice(), vec![99; new_count + 1]);
        for reader in &readers {
            assert!(reader
                .as_slice()
                .iter()
                .all(|&value| value == reader.len() as i32));
        }
        Ok(())
    }

    #[tokio::test]
    async fn pinned_readback_waits_for_completion_and_consumer_before_reuse(
    ) -> Result<(), DeviceError> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = global_policy(0)?;
        let cache = ReadbackCache::default();
        let mut previous = None;
        for expected in [151645i32, 17] {
            let output = with_context(|exec| {
                value((|| -> Result<_, DeviceError> {
                    // SAFETY: writes and D2H share exec; resources are retained.
                    unsafe {
                        let tensor = api::copy_host_vec_to_device(&Arc::new(vec![expected]))
                            .execute(exec)?;
                        let output = cache.enqueue(tensor, exec)?;
                        assert!(cache
                            .buffers
                            .lock()
                            .unwrap()
                            .iter()
                            .all(|buffer| Arc::strong_count(buffer) > 1));
                        Ok(output)
                    }
                })())
            })
            .schedule(&policy)?
            .await??;
            assert!(cache
                .buffers
                .lock()
                .unwrap()
                .iter()
                .all(|buffer| Arc::strong_count(buffer) > 1));
            // SAFETY: the outer await includes the asynchronous D2H copy.
            let actual = output.as_slice();
            assert_eq!(actual, [expected]);
            let pointer = actual.as_ptr();
            if let Some(previous) = previous {
                assert_eq!(pointer, previous);
            }
            previous = Some(pointer);
            drop(output);
            assert_eq!(Arc::strong_count(&cache.buffers.lock().unwrap()[0]), 1);
        }

        let result = with_context(|exec| {
            value((|| -> Result<(), DeviceError> {
                // SAFETY: even an error after enqueue leaves ownership with exec.
                unsafe {
                    let tensor =
                        api::copy_host_vec_to_device(&Arc::new(vec![42i32])).execute(exec)?;
                    drop(cache.enqueue(tensor, exec)?);
                }
                assert!(cache
                    .buffers
                    .lock()
                    .unwrap()
                    .iter()
                    .all(|buffer| Arc::strong_count(buffer) > 1));
                Err(DeviceError::Internal("injected failure after D2H".into()))
            })())
        })
        .schedule(&policy)?
        .await?;
        assert!(result.is_err());
        assert_eq!(Arc::strong_count(&cache.buffers.lock().unwrap()[0]), 1);

        let (first, second) = with_context(|exec| {
            value((|| -> Result<_, DeviceError> {
                // SAFETY: both D2H copies and their sources belong to exec.
                unsafe {
                    let a = api::copy_host_vec_to_device(&Arc::new(vec![11i32])).execute(exec)?;
                    let first = cache.enqueue(a, exec)?;
                    let b = api::copy_host_vec_to_device(&Arc::new(vec![22i32])).execute(exec)?;
                    let second = cache.enqueue(b, exec)?;
                    assert!(!Arc::ptr_eq(&first, &second));
                    Ok((first, second))
                }
            })())
        })
        .schedule(&policy)?
        .await??;
        assert_eq!(first.as_slice(), [11]);
        assert_eq!(second.as_slice(), [22]);
        Ok(())
    }
}
