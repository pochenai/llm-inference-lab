use std::{ffi::CStr, marker::PhantomData};

/// A thread-local NVTX range. Keep it inside synchronous code, never across await.
pub(crate) struct ScopedRange(PhantomData<*mut ()>);

impl ScopedRange {
    pub(crate) fn new(name: &CStr) -> Self {
        nvtx_sys::range_push_ascii(name);
        // The marker prevents moving the guard to another thread before pop.
        Self(PhantomData)
    }
}

impl Drop for ScopedRange {
    fn drop(&mut self) {
        nvtx_sys::range_pop();
    }
}
