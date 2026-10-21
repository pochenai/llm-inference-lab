use std::sync::{Arc, Mutex};

use cuda_async::{
    device_buffer::DeviceAllocation,
    device_operation::{DeviceOp, ExecutionContext},
    error::DeviceError,
};
use cutile::{
    api,
    cuda_core::{memcpy_htod_async, DType},
    tensor::Tensor,
};

const MAX_IDLE_BUFFERS: usize = 4;

/// Reusable integer upload destinations, exclusively leased per submission.
#[derive(Default)]
pub(super) struct UploadPool<T: DType = i32> {
    idle: Mutex<Vec<Arc<Tensor<T>>>>,
    host_idle: Mutex<Vec<Arc<Vec<T>>>>,
}

impl<T: DType> UploadPool<T> {
    /// Copy a borrowed slice into reusable owned staging memory. Only cache-exclusive
    /// buffers can be refilled; exec retains each upload source through completion.
    /// Cache hits reuse the staging Vec and its Arc owner; capacity grows only as needed.
    pub unsafe fn upload_slice(
        self: &Arc<Self>,
        data: &[T],
        exec: &ExecutionContext,
    ) -> Result<Arc<Tensor<T>>, DeviceError> {
        let source = {
            let mut idle = self.host_idle.lock().unwrap();
            if let Some(source) = idle.iter_mut().find_map(|entry| {
                let buffer = Arc::get_mut(entry)?;
                buffer.clear();
                buffer.extend_from_slice(data);
                Some(entry.clone())
            }) {
                source
            } else {
                let source = Arc::new(data.to_vec());
                if idle.len() < MAX_IDLE_BUFFERS {
                    idle.push(source.clone());
                }
                source
            }
        };
        self.upload(source, &[data.len()], exec)
    }

    /// Caller owns exec and must establish GPU completion. All consumers must run
    /// on exec until it finishes; source/destination owners survive cancellation.
    pub unsafe fn upload(
        self: &Arc<Self>,
        data: Arc<impl AsRef<[T]> + Send + Sync + 'static>,
        shape: &[usize],
        exec: &ExecutionContext,
    ) -> Result<Arc<Tensor<T>>, DeviceError> {
        let values: &[T] = data.as_ref().as_ref();
        let elements = shape.iter().try_fold(1usize, |n, &dim| {
            n.checked_mul(dim)
                .filter(|&n| dim > 0 && n <= i32::MAX as usize)
        });
        if elements != Some(values.len()) || values.is_empty() {
            return Err(DeviceError::Internal("invalid integer upload shape".into()));
        }
        let cached = {
            let mut idle = self.idle.lock().unwrap();
            idle.iter()
                .position(|tensor| {
                    tensor.size() == values.len() && tensor.device_id() == exec.get_device_id()
                })
                .map(|index| idle.swap_remove(index))
        };
        let tensor = match cached {
            Some(tensor) => tensor,
            // Initialize once on a miss. Upload overwrites every element each use.
            None => Arc::new(api::zeros::<T>(&[values.len()]).execute(exec)?),
        };
        let lease = Arc::new(UploadLease {
            tensor: Some(tensor),
            pool: self.clone(),
        });
        exec.retain(lease.clone())?;
        exec.retain(data.clone())?;
        memcpy_htod_async(
            lease.device_ptr(),
            values.as_ptr(),
            values.len(),
            exec.get_cuda_stream(),
        )?;
        let mut stride = 1i32;
        let mut strides = vec![1; shape.len()];
        for (index, &dim) in shape.iter().enumerate().rev() {
            strides[index] = stride;
            stride *= dim as i32;
        }
        // SAFETY: there is one tensor handle per checkout and all GPU accesses are
        // ordered on exec. Holding this owner also prevents reuse after exec finishes
        // if a tensor or derived handle escapes the submission.
        Ok(Arc::new(Tensor::from_foreign(
            lease,
            shape.iter().map(|&dim| dim as i32).collect(),
            strides,
        )))
    }
}

struct UploadLease<T: DType> {
    tensor: Option<Arc<Tensor<T>>>,
    pool: Arc<UploadPool<T>>,
}

// SAFETY: the original tensor owns a fixed allocation with stable pointer, byte
// length, and device ordinal for this lease's lifetime.
unsafe impl<T: DType> DeviceAllocation for UploadLease<T> {
    fn device_ptr(&self) -> u64 {
        self.tensor
            .as_ref()
            .unwrap()
            .device_pointer()
            .cu_deviceptr()
    }
    fn len_bytes(&self) -> usize {
        self.tensor.as_ref().unwrap().num_bytes()
    }
    fn device_id(&self) -> usize {
        self.tensor.as_ref().unwrap().device_id()
    }
}

impl<T: DType> Drop for UploadLease<T> {
    fn drop(&mut self) {
        let mut idle = self.pool.idle.lock().unwrap();
        if idle.len() == MAX_IDLE_BUFFERS {
            idle.remove(0);
        }
        idle.push(self.tensor.take().unwrap());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuda_async::{
        device_context::global_policy,
        device_operation::{value, with_context},
    };
    use cutile::tensor::ToHostVec;

    #[tokio::test]
    async fn borrowed_u32_upload_reuses_owned_staging_after_completion() -> Result<(), DeviceError>
    {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = global_policy(0)?;
        let pool = Arc::new(UploadPool::<u32>::default());
        let mut host_pointer = None;
        let mut device_pointer = None;
        for mut input in [[1u32, 151935, 3, 4], [151645, 8, 7, 6]] {
            let expected = input;
            let (actual, pointer) = with_context(|exec| {
                value((|| -> Result<_, DeviceError> {
                    // SAFETY: upload leases both buffers until outer completion.
                    let tensor = unsafe { pool.upload_slice(&input, exec)? };
                    assert!(pool
                        .host_idle
                        .lock()
                        .unwrap()
                        .iter()
                        .all(|buffer| Arc::strong_count(buffer) > 1));
                    // Source ownership is independent of the caller after upload.
                    input.fill(0);
                    let pointer = tensor.device_pointer().cu_deviceptr();
                    let actual = unsafe { tensor.to_host_vec().execute(exec)? };
                    Ok((actual, pointer))
                })())
            })
            .schedule(&policy)?
            .await??;
            assert_eq!(actual, expected);
            let idle = pool.host_idle.lock().unwrap();
            assert_eq!(idle.len(), 1);
            let source = idle[0].as_ptr();
            if let Some(previous) = host_pointer {
                assert_eq!(source, previous);
            }
            if let Some(previous) = device_pointer {
                assert_eq!(pointer, previous);
            }
            host_pointer = Some(source);
            device_pointer = Some(pointer);
        }
        Ok(())
    }

    #[tokio::test]
    async fn uploads_reuse_storage_with_new_contents_and_shapes() -> Result<(), DeviceError> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = global_policy(0)?;
        let pool = Arc::new(UploadPool::default());
        let mut previous = None;
        for (shape, data) in [
            (vec![2, 4], vec![1, 2, 3, 4, 5, 6, 7, 8]),
            (vec![8], vec![-8, -7, -6, -5, -4, -3, -2, -1]),
        ] {
            let expected = data.clone();
            let (actual, pointer) = with_context(|exec| {
                value((|| -> Result<_, DeviceError> {
                    // SAFETY: upload/readback use one exec and outer completion is awaited.
                    let tensor = unsafe { pool.upload(Arc::new(data), &shape, exec)? };
                    assert_eq!(
                        tensor.shape(),
                        shape.iter().map(|&n| n as i32).collect::<Vec<_>>()
                    );
                    let pointer = tensor.device_pointer().cu_deviceptr();
                    let actual = unsafe { tensor.to_host_vec().execute(exec)? };
                    assert!(pool.idle.lock().unwrap().is_empty());
                    Ok((actual, pointer))
                })())
            })
            .schedule(&policy)?
            .await??;
            assert_eq!(actual, expected);
            if let Some(previous) = previous {
                assert_eq!(pointer, previous);
            }
            previous = Some(pointer);
            assert_eq!(pool.idle.lock().unwrap().len(), 1);
        }
        Ok(())
    }

    #[tokio::test]
    async fn live_uploads_are_isolated_and_failure_returns_buffers() -> Result<(), DeviceError> {
        let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
        let policy = global_policy(0)?;
        let pool = Arc::new(UploadPool::default());
        let (a, b) = with_context(|exec| {
            value(unsafe {
                (|| -> Result<_, DeviceError> {
                    let a = pool.upload(Arc::new(vec![1, 2, 3, 4]), &[4], exec)?;
                    let b = pool.upload(Arc::new(vec![9, 8, 7, 6]), &[4], exec)?;
                    assert_ne!(
                        a.device_pointer().cu_deviceptr(),
                        b.device_pointer().cu_deviceptr()
                    );
                    Ok((a, b))
                })()
            })
        })
        .schedule(&policy)?
        .await??;
        assert!(pool.idle.lock().unwrap().is_empty());
        assert_eq!(a.to_host_vec().schedule(&policy)?.await?, [1, 2, 3, 4]);
        assert_eq!(b.to_host_vec().schedule(&policy)?.await?, [9, 8, 7, 6]);
        assert_eq!(pool.idle.lock().unwrap().len(), 2);
        let result = with_context(|exec| {
            value(unsafe {
                (|| -> Result<(), DeviceError> {
                    assert!(pool.upload(Arc::new(vec![1, 2]), &[3], exec).is_err());
                    let tensor = pool.upload(Arc::new(vec![5, 6, 7, 8]), &[2, 2], exec)?;
                    drop(tensor);
                    assert_eq!(pool.idle.lock().unwrap().len(), 1);
                    Err(DeviceError::Internal(
                        "injected failure after upload".into(),
                    ))
                })()
            })
        })
        .schedule(&policy)?
        .await?;
        assert!(result.is_err());
        assert_eq!(pool.idle.lock().unwrap().len(), 2);
        Ok(())
    }
}
