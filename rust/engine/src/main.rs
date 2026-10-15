use engine::{
    runner::{
        model::simple_mlp::{SimpleMLPEngine, INPUT_DIM},
        model_runner::ModelRunner,
    },
    scheduler::Scheduler,
};
use tokio::sync::mpsc;
use tokio_util::{sync::CancellationToken, task::TaskTracker};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = SimpleMLPEngine::new(0).await?;
    let cancel_token = CancellationToken::new();
    let (batch_req_tx, batch_req_rx) = mpsc::unbounded_channel();
    let (batch_reply_tx, batch_reply_rx) = mpsc::unbounded_channel();
    let scheduler = Scheduler::new(cancel_token.child_token(), batch_req_tx, batch_reply_rx);
    let runner_tasks = TaskTracker::new();
    runner_tasks.spawn(
        ModelRunner::new(batch_req_rx, batch_reply_tx, model).run(cancel_token.child_token()),
    );
    runner_tasks.close();

    let mut callers = Vec::new();
    for i in 0..6 {
        let scheduler = scheduler.clone();
        callers.push(tokio::spawn(async move {
            let response = scheduler
                .execute(
                    (0..INPUT_DIM)
                        .map(|j| (i * INPUT_DIM + j).to_string())
                        .collect::<Vec<_>>()
                        .join(" "),
                )
                .await;
            println!("caller {i}: {response:?}");
            response
        }));
    }
    // Always stop both components, including when a caller receives a model error.
    let mut result = Ok(());
    for caller in callers {
        match caller.await {
            Ok(Ok(_)) => {}
            Ok(Err(error)) => result = Err(error.into()),
            Err(error) => result = Err(error.into()),
        }
    }
    cancel_token.cancel();
    scheduler.wait_stopped().await;
    runner_tasks.wait().await;
    result
}
