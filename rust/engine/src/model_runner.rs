use crate::common::{BatchIn, BatchOut, LLMResponse};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

pub struct ModelRunner {
    batch_req_rx: mpsc::UnboundedReceiver<BatchIn>,
    batch_reply_tx: mpsc::UnboundedSender<BatchOut>,
}

impl ModelRunner {
    pub fn new(
        batch_req_rx: mpsc::UnboundedReceiver<BatchIn>,
        batch_reply_tx: mpsc::UnboundedSender<BatchOut>,
    ) -> Self {
        Self {
            batch_req_rx,
            batch_reply_tx,
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
            let output = batch
                .into_iter()
                .map(|item| {
                    let _prompt = item.request.prompt;
                    LLMResponse {
                        request_id: item.request_id,
                        value: 42,
                    }
                })
                .collect();
            if self.batch_reply_tx.send(output).is_err() {
                break;
            }
        }
    }
}
