use std::future::Future;

use crate::{error::EngineError, runner::types::ModelOutput, types::BatchIn};

use tokenizers::{Encoding, Tokenizer};

pub mod simple_mlp;

pub trait Model: Send + Sync {
    fn tokenizer(&self) -> &Tokenizer;

    fn generate(
        &self,
        batch: BatchIn<Encoding>,
    ) -> impl Future<Output = Result<Vec<ModelOutput>, EngineError>> + Send;
}
