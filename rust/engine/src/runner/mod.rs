pub mod kv_pool;
pub mod loader;
pub mod model;
pub mod model_runner;
mod upload;

// Keep GPU test runtimes from overlapping; individual tests still exercise concurrent batches.
#[cfg(test)]
pub(crate) static GPU_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
