use std::sync::Arc;

use cutile::{cuda_core::PinnedHostBuffer, tensor::Tensor};
use futures::{future::BoxFuture, FutureExt};

use super::ModelError;

type HostTokens =
    futures::future::Shared<BoxFuture<'static, Result<Arc<PinnedHostBuffer<i32>>, ModelError>>>;

/// Address/index of a GPU token prediction whose value may still be pending.
/// Shared readback yields its CPU ID after producer and D2H completion.
#[derive(Clone)]
pub struct GpuTokenPredictionHandle {
    pub(crate) values: Arc<Tensor<i32>>,
    pub(crate) index: usize,
    /// Absolute position of this prediction, known without reading its GPU value.
    pub(crate) position: usize,
    /// CUDA stream identity: queued GPU consumers must use this same stream.
    /// None means producer computation already completed; no stream restriction remains.
    pub(crate) producer_stream_id: Option<usize>,
    host: HostTokens,
    pub(crate) vocab_size: usize,
}

impl GpuTokenPredictionHandle {
    /// Create handles sharing batch storage and one deferred CPU readback future.
    /// Pending GPU values require stream ordering; CPU reads await compute and D2H.
    pub(crate) fn batch(
        values: Arc<Tensor<i32>>,
        host: impl std::future::Future<Output = Result<Arc<PinnedHostBuffer<i32>>, ModelError>>
            + Send
            + 'static,
        vocab_size: usize,
        positions: impl ExactSizeIterator<Item = usize>,
    ) -> impl Iterator<Item = Self> {
        assert_eq!(values.size(), positions.len());
        let host = host.boxed().shared();
        positions.enumerate().map(move |(index, position)| Self {
            values: values.clone(),
            index,
            position,
            producer_stream_id: None,
            host: host.clone(),
            vocab_size,
        })
    }

    /// Await the independent D2H submission before touching pinned host storage.
    pub async fn read(&self) -> Result<u32, ModelError> {
        let values = self.host.clone().await?;
        let token = values.as_slice()[self.index];
        if token < 0 || token as usize >= self.vocab_size {
            return Err(ModelError::Execution(
                "non-finite logits or invalid argmax".into(),
            ));
        }
        Ok(token as u32)
    }
}

/// CPU-only models can return a ready token; GPU models defer host observation.
#[derive(Clone)]
pub enum NextToken {
    Host(u32),
    Device(GpuTokenPredictionHandle),
}

impl NextToken {
    pub async fn read(self) -> Result<u32, ModelError> {
        match self {
            Self::Host(token) => Ok(token),
            Self::Device(token) => token.read().await,
        }
    }
}

/// Borrowed/shared CPU history or one GPU prediction; no dummy CPU token is needed.
#[derive(Clone)]
pub enum ModelTokens<'a> {
    /// Borrowed CPU prefix; its slice length is this forward's input boundary.
    BorrowedHost(&'a [u32]),
    /// Shared history prefix; owned submission tasks release this Arc after enqueue.
    SharedHost {
        tokens: Arc<Vec<u32>>,
        end: usize,
    },
    Device(GpuTokenPredictionHandle),
}

impl<'a, T: AsRef<[u32]> + ?Sized> From<&'a T> for ModelTokens<'a> {
    fn from(tokens: &'a T) -> Self {
        Self::BorrowedHost(tokens.as_ref())
    }
}

impl ModelTokens<'_> {
    pub(crate) fn host(&self) -> Result<&[u32], ModelError> {
        match self {
            Self::BorrowedHost(tokens) => Ok(tokens),
            Self::SharedHost { tokens, end } => tokens.get(..*end).ok_or_else(|| {
                ModelError::InvalidInput("shared token prefix exceeds history".into())
            }),
            Self::Device(_) => Err(ModelError::InvalidInput(
                "model requires host token input".into(),
            )),
        }
    }
}
