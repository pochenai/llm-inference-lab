use engine::{
    runner::{model::INPUT_DIM, model_runner::RunnerBuilder},
    scheduler::Scheduler,
    types::SessionRequest,
};
use tokio::sync::mpsc;
use tokio_util::{sync::CancellationToken, task::TaskTracker};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cancel_token = CancellationToken::new();
    let (command_tx, command_rx) = mpsc::unbounded_channel();
    let (batch_reply_tx, batch_reply_rx) = mpsc::unbounded_channel();
    let scheduler = Scheduler::new(cancel_token.child_token(), command_tx, batch_reply_rx);
    let runner_tasks = TaskTracker::new();
    let runner = RunnerBuilder::default()
        .build(command_rx, batch_reply_tx)
        .await?;
    runner_tasks.spawn(runner.run(cancel_token.child_token()));
    runner_tasks.close();

    let mut callers = Vec::new();
    for i in 0..6 {
        let scheduler = scheduler.clone();
        callers.push(tokio::spawn(async move {
            let response = scheduler
                .execute(SessionRequest::new(
                    None,
                    (0..INPUT_DIM)
                        .map(|j| (i * INPUT_DIM + j).to_string())
                        .collect::<Vec<_>>()
                        .join(" "),
                ))
                .await;
            println!("caller {i}: {response:?}");
            response
        }));
    }
    let session_scheduler = scheduler.clone();
    callers.push(tokio::spawn(async move {
        let session_id = session_scheduler.open_session().await?;
        println!("opened session {session_id}");
        let first = session_scheduler
            .execute(SessionRequest::new(Some(session_id), "1 2 3".into()))
            .await?;
        println!("session {session_id}, first turn: {first:?}");
        // Send only the new prompt delta; Runner retains the previous tokens and KV.
        let second = session_scheduler
            .execute(SessionRequest::new(Some(session_id), "4 5".into()))
            .await?;
        println!("session {session_id}, second turn: {second:?}");
        session_scheduler.close_session(session_id).await?;
        println!("closed session {session_id}");
        Ok(second)
    }));
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
