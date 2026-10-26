use std::{
    borrow::Borrow,
    sync::{Arc, Mutex},
};

use cuda_async::{device_operation::ExecutionContext, error::DeviceError};
use cutile::{
    cuda_core::{memcpy_dtoh_async, CudaContext, DType, DeviceCopy, PinnedHostBuffer},
    tensor::Tensor,
};

const MAX_CACHED_READBACKS: usize = 16;

pub(super) type Readback<T> = Arc<PinnedHostBuffer<T>>;
pub(super) type ReadbackCache<T> = Mutex<Vec<Readback<T>>>;

/// Enqueue D2H without waiting; read the returned buffer only after outer completion.
/// Cache-only ownership proves a buffer has neither an in-flight copy nor a reader.
/// Source and destination stay retained by exec even if the consumer is cancelled.
pub(super) unsafe fn enqueue_readback<T: DeviceCopy + DType>(
    tensor: impl Borrow<Tensor<T>> + Send + 'static,
    cache: &ReadbackCache<T>,
    exec: &ExecutionContext,
) -> Result<Readback<T>, DeviceError> {
    let count = tensor.borrow().size();
    let source = tensor.borrow().device_pointer().cu_deviceptr();
    let output = {
        let mut idle = cache.lock().unwrap();
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
            } else if let Some(entry) = idle.iter_mut().find(|entry| Arc::strong_count(entry) == 1)
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
                        enqueue_readback(tensor, &cache, exec)
                    }
                })())
            })
            .schedule(&policy)?
            .await??;
            assert_eq!(output.as_slice(), vec![count as i32; count]);
            if count == new_count {
                let entries = cache.lock().unwrap();
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
        let readers = cache.lock().unwrap().clone();
        let overflow = with_context(|exec| {
            value((|| -> Result<_, DeviceError> {
                // SAFETY: exec owns the copy; existing readers must remain untouched.
                unsafe {
                    let tensor =
                        api::copy_host_vec_to_device(&Arc::new(vec![99i32; new_count + 1]))
                            .execute(exec)?;
                    let output = enqueue_readback(tensor, &cache, exec)?;
                    let entries = cache.lock().unwrap();
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
                        let output = enqueue_readback(tensor, &cache, exec)?;
                        assert!(cache
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
            assert_eq!(Arc::strong_count(&cache.lock().unwrap()[0]), 1);
        }

        let result = with_context(|exec| {
            value((|| -> Result<(), DeviceError> {
                // SAFETY: even an error after enqueue leaves ownership with exec.
                unsafe {
                    let tensor =
                        api::copy_host_vec_to_device(&Arc::new(vec![42i32])).execute(exec)?;
                    drop(enqueue_readback(tensor, &cache, exec)?);
                }
                assert!(cache
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
        assert_eq!(Arc::strong_count(&cache.lock().unwrap()[0]), 1);

        let (first, second) = with_context(|exec| {
            value((|| -> Result<_, DeviceError> {
                // SAFETY: both D2H copies and their sources belong to exec.
                unsafe {
                    let a = api::copy_host_vec_to_device(&Arc::new(vec![11i32])).execute(exec)?;
                    let first = enqueue_readback(a, &cache, exec)?;
                    let b = api::copy_host_vec_to_device(&Arc::new(vec![22i32])).execute(exec)?;
                    let second = enqueue_readback(b, &cache, exec)?;
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
