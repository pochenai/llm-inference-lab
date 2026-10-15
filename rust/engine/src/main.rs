use engine::{common::Promp, model_runner::ModelRunner, scheduler::Scheduler};
use tokio::sync::mpsc;
use tokio_util::{sync::CancellationToken, task::TaskTracker};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cancel_token = CancellationToken::new();
    let (batch_req_tx, batch_req_rx) = mpsc::unbounded_channel();
    let (batch_reply_tx, batch_reply_rx) = mpsc::unbounded_channel();

    let scheduler = Scheduler::new(cancel_token.child_token(), batch_req_tx, batch_reply_rx);
    let runner_tasks = TaskTracker::new();
    runner_tasks
        .spawn(ModelRunner::new(batch_req_rx, batch_reply_tx).run(cancel_token.child_token()));
    runner_tasks.close();

    let mut callers = Vec::new();
    for i in 0..6 {
        let scheduler = scheduler.clone();
        callers.push(tokio::spawn(async move {
            let response = scheduler
                .execute(Promp {
                    prompt: format!("Hello from caller {i}"),
                })
                .await?;
            println!("caller {i}: {response:?}");
            Ok::<_, engine::error::EngineError>(())
        }));
    }
    for caller in callers {
        caller.await??;
    }

    cancel_token.cancel();
    scheduler.wait_stopped().await;
    runner_tasks.wait().await;
    Ok(())
}
