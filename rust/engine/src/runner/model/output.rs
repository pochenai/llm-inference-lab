use std::sync::Arc;

use cutile::{cuda_core::PinnedHostBuffer, tensor::Tensor};
use futures::{future::BoxFuture, FutureExt};

use super::ModelError;

type HostTokens =
    futures::future::Shared<BoxFuture<'static, Result<Arc<PinnedHostBuffer<i32>>, ModelError>>>;

/// One sampled token. GPU feedback and CPU readback share owned batch storage.
#[derive(Clone)]
pub struct DeviceToken {
    pub(crate) values: Arc<Tensor<i32>>,
    pub(crate) index: usize,
    host: HostTokens,
    pub(crate) vocab_size: usize,
}

impl DeviceToken {
    /// The producer must have completed before these handles enter another forward.
    /// One type-erased readback future is shared by the whole packed batch.
    pub(crate) fn batch(
        values: Arc<Tensor<i32>>,
        host: impl std::future::Future<Output = Result<Arc<PinnedHostBuffer<i32>>, ModelError>>
            + Send
            + 'static,
        vocab_size: usize,
    ) -> impl Iterator<Item = Self> {
        let host = host.boxed().shared();
        (0..values.size()).map(move |index| Self {
            values: values.clone(),
            index,
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
pub enum NextToken {
    Host(u32),
    Device(DeviceToken),
}

impl NextToken {
    pub async fn read(self) -> Result<u32, ModelError> {
        match self {
            Self::Host(token) => Ok(token),
            Self::Device(token) => token.read().await,
        }
    }
}

/// A host history prefix or one sampled GPU token; no dummy CPU history is needed.
#[derive(Clone)]
pub enum ModelTokens<'a> {
    Host(&'a [u32]),
    Device(DeviceToken),
}

impl<'a, T: AsRef<[u32]> + ?Sized> From<&'a T> for ModelTokens<'a> {
    fn from(tokens: &'a T) -> Self {
        Self::Host(tokens.as_ref())
    }
}

impl ModelTokens<'_> {
    pub(crate) fn host(&self) -> Result<&[u32], ModelError> {
        match self {
            Self::Host(tokens) => Ok(tokens),
            Self::Device(_) => Err(ModelError::InvalidInput(
                "model requires host token input".into(),
            )),
        }
    }
}
