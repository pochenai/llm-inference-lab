use crate::error::EngineError;
use crate::{
    runner::model::Model,
    types::{BatchIn, BatchReply, LLMResponse, SingleRequest},
};
use tokenizers::Encoding;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

pub struct ModelRunner<M> {
    batch_req_rx: mpsc::UnboundedReceiver<BatchIn>,
    batch_reply_tx: mpsc::UnboundedSender<BatchReply>,
    model: M,
}

impl<M: Model> ModelRunner<M> {
    pub fn new(
        batch_req_rx: mpsc::UnboundedReceiver<BatchIn>,
        batch_reply_tx: mpsc::UnboundedSender<BatchReply>,
        model: M,
    ) -> Self {
        Self {
            batch_req_rx,
            batch_reply_tx,
            model,
        }
    }

    async fn execute_batch(&self, batch: BatchIn) -> BatchReply {
        let request_ids = batch.iter().map(|item| item.request_id).collect();
        let inputs: Result<BatchIn<Encoding>, EngineError> = batch
            .into_iter()
            .map(|item| {
                let request = self
                    .model
                    .tokenizer()
                    .encode(item.request.as_str(), true)
                    .map_err(|error| EngineError::InvalidInput(error.to_string()))?;
                Ok(SingleRequest {
                    request_id: item.request_id,
                    request,
                })
            })
            .collect();
        let result = match inputs {
            Ok(inputs) => self.model.generate(inputs).await,
            Err(error) => Err(error),
        };
        let result = result.and_then(|outputs| {
            outputs
                .into_iter()
                .map(|item| {
                    let output = self
                        .model
                        .tokenizer()
                        .decode(&item.token_ids, true)
                        .map_err(|error| EngineError::Model(error.to_string()))?;
                    Ok(LLMResponse {
                        request_id: item.request_id,
                        output,
                    })
                })
                .collect()
        });
        BatchReply {
            request_ids,
            result,
        }
    }

    pub async fn run(mut self, cancel_token: CancellationToken) {
        loop {
            let batch = tokio::select! {
                biased;
                _ = cancel_token.cancelled() => break,
                batch = self.batch_req_rx.recv() => match batch {
                    Some(batch) => batch,
                    None => break,
                },
            };
            println!("model runner: batch size = {}", batch.len());
            let output = self.execute_batch(batch).await;
            if self.batch_reply_tx.send(output).is_err() {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::model::simple_mlp::SimpleMLPEngine;

    // Exercise encoding and batch results directly, without a scheduler or run loop.
    #[tokio::test]
    async fn execute_batch_encodes_strings_and_returns_input_errors() -> Result<(), EngineError> {
        let model: SimpleMLPEngine = SimpleMLPEngine::new(0).await?;
        let (_request_tx, request_rx) = mpsc::unbounded_channel();
        let (reply_tx, _reply_rx) = mpsc::unbounded_channel();
        let runner = ModelRunner::new(request_rx, reply_tx, model);
        let batch = |request_id, request| {
            vec![SingleRequest {
                request_id,
                request,
            }]
        };

        // Known tokens still fail model validation when there are fewer than 16.
        let invalid = runner.execute_batch(batch(10, "0 1 2".into())).await;
        assert_eq!(invalid.request_ids, vec![10]);
        assert!(matches!(invalid.result, Err(EngineError::InvalidInput(_))));

        // Unknown words encode as [UNK]; the MLP rejects those IDs.
        let unknown = runner
            .execute_batch(batch(20, vec!["unknown"; 16].join(" ")))
            .await;
        assert_eq!(unknown.request_ids, vec![20]);
        assert!(matches!(unknown.result, Err(EngineError::InvalidInput(_))));

        // Use a single-request result as the reference for the nonzero input.
        let numeric_input = (0..16)
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        let reference = runner
            .execute_batch(batch(40, numeric_input.clone()))
            .await
            .result?;

        // Two distinct inputs share one batch and retain their own request IDs.
        // Zero IDs produce "0"; the other output must match its single-request result.
        let valid = runner
            .execute_batch(vec![
                SingleRequest {
                    request_id: 30,
                    request: vec!["0"; 16].join(" "),
                },
                SingleRequest {
                    request_id: 31,
                    request: numeric_input,
                },
            ])
            .await;
        assert_eq!(valid.request_ids, vec![30, 31]);
        let output = valid.result?;
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].request_id, 30);
        assert_eq!(output[0].output, "0");
        assert_eq!(output[1].request_id, 31);
        assert_eq!(output[1].output, reference[0].output);
        Ok(())
    }
}
