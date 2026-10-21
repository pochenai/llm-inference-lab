use std::{
    borrow::Borrow,
    sync::{Arc, Mutex},
};

use cuda_async::{device_operation::ExecutionContext, error::DeviceError};
use cutile::{
    cuda_core::{memcpy_dtoh_async, CudaContext, DType, DeviceCopy, PinnedHostBuffer},
    tensor::Tensor,
};

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
            // Keep only a small cache; overflow buffers live through their GPU owners.
            if idle.len() < 4 {
                idle.push(buffer.clone());
            }
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
