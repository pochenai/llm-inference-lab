use super::Qwen3_5Engine;
use anyhow::{ensure, Result};
use cuda_async::device_operation::{value, with_context, DeviceOp};
use std::{path::PathBuf, sync::Arc};

#[tokio::test]
#[ignore = "generate the independent fixture with rust/engine/examples/qwen3_5_reference.py first"]
async fn reference_chunking_and_state_validation() -> Result<()> {
    let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
    let model_dir = PathBuf::from(std::env::var("QWEN35_TEST_MODEL")?);
    let reference_path = std::env::var_os("QWEN35_TEST_REFERENCE")
        .map(PathBuf::from)
        .unwrap_or_else(|| model_dir.join("reference.f32"));
    let reference = std::fs::read(reference_path)?;
    let expected: Vec<f32> = reference
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let policy = cuda_async::device_context::global_policy(0)?;
    let engine = Qwen3_5Engine::<cutile::core::bf16>::load(&model_dir, &policy).await?;
    let ids = [1, 7, 9, 3, 12];
    let mut full = engine.new_context(16).await?;
    let (token, logits) = engine.step_with_logits(&ids, &mut full).await?;
    ensure!(logits.len() == expected.len(), "reference shape mismatch");
    let max_error = logits
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rms = (logits
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / logits.len() as f32)
        .sqrt();
    let expected_token = expected
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32;
    eprintln!("reference max={max_error}, rms={rms}, token={token}, expected={expected_token}");
    let reference_rms =
        (expected.iter().map(|v| v * v).sum::<f32>() / expected.len() as f32).sqrt();
    let tiny = engine.config().text_config.hidden_size == 128;
    // Long BF16 MoE paths amplify rounding through discrete expert selection.
    // Keep the reduced operator fixture strict and report the real-model budget.
    let within_budget = if tiny {
        max_error < 0.005 && rms < 0.0005
    } else {
        max_error < 1.0 && rms / reference_rms < 0.05
    };
    ensure!(
        within_budget,
        "independent logits differ: relative RMS={}",
        rms / reference_rms
    );
    assert_eq!(token, expected_token);
    for chunk in [1, 2, 3, 4] {
        let mut incremental = engine.new_context(16).await?;
        let mut result = None;
        for end in (chunk..ids.len())
            .step_by(chunk)
            .chain(std::iter::once(ids.len()))
        {
            result = Some(
                engine
                    .step_with_logits(&ids[..end], &mut incremental)
                    .await?,
            );
        }
        let (cached_token, cached) = result.unwrap();
        let error = cached
            .iter()
            .zip(&logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("chunk={chunk} max={error}, token={cached_token}");
        ensure!(error < 0.02, "chunk={chunk} error={error}");
        assert_eq!(cached_token, token);
        assert_eq!(incremental.ready_len(), ids.len());
        assert!(incremental.is_valid());
        assert!(engine
            .step_with_logits(&[1, 7, 8, 3, 12, 9], &mut incremental)
            .await
            .is_err());
        assert!(engine
            .step_with_logits(&ids, &mut incremental)
            .await
            .is_err());
        assert!(incremental.is_valid());
    }
    assert!(engine.new_context(0).await.is_err());
    let mut invalid = engine.new_context(4).await?;
    assert!(engine.step_with_logits(&ids, &mut invalid).await.is_err());
    assert!(engine
        .step_with_logits(&[u32::MAX], &mut invalid)
        .await
        .is_err());
    assert_eq!(invalid.ready_len(), 0);
    Ok(())
}

#[tokio::test]
async fn router_ties_use_lowest_indices_and_normalize_selected_scores() -> Result<()> {
    use super::kernels;
    use cutile::{
        api,
        core::bf16,
        tensor::{PartitionMut, Reshape, ToHostVec},
        tile_kernel::TileKernel,
        DType,
    };
    let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
    let policy = cuda_async::device_context::global_policy(0)?;
    let values = Arc::new(
        [2.0f32, 2.0, 2.0, 1.0, -80.0, -80.0, -80.0, -80.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect::<Vec<_>>(),
    );
    let logits = api::copy_host_vec_to_device(&values)
        .schedule(&policy)?
        .await?
        .reshape(&[2, 4])?;
    let mut ids = api::zeros::<i32>(&[2, 2]).schedule(&policy)?.await?;
    let mut scores = api::zeros::<bf16>(&[2, 2]).schedule(&policy)?.await?;
    kernels::route(
        (&mut ids).partition([1, 2]),
        (&mut scores).partition([1, 2]),
        &logits,
    )
    .generics(vec![bf16::DTYPE.to_string(), "4".into(), "2".into()])
    .schedule(&policy)?
    .await?;
    assert_eq!(ids.to_host_vec().schedule(&policy)?.await?, [0, 1, 0, 1]);
    assert_eq!(
        scores.to_host_vec().schedule(&policy)?.await?,
        vec![bf16::from_f32(0.5); 4]
    );
    Ok(())
}

#[tokio::test]
async fn convolution_preserves_three_distinct_history_planes() -> Result<()> {
    use super::kernels;
    use cutile::{
        api,
        core::bf16,
        tensor::{Reshape, ToHostVec},
        tile_kernel::TileKernel,
        DType,
    };
    let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
    let policy = cuda_async::device_context::global_policy(0)?;
    let weights = Arc::new(vec![bf16::from_f32(1.0); 4 * 128]);
    let weights = api::copy_host_vec_to_device(&weights)
        .schedule(&policy)?
        .await?
        .reshape(&[4, 128])?;
    for chunk in [1usize, 2, 5] {
        let state = api::zeros::<bf16>(&[3, 128]).schedule(&policy)?.await?;
        let mut start = 0;
        while start < 5 {
            let end = (start + chunk).min(5);
            let host = Arc::new(
                (start..end)
                    .flat_map(|token| std::iter::repeat_n(bf16::from_f32((token + 1) as f32), 128))
                    .collect::<Vec<_>>(),
            );
            let input = api::copy_host_vec_to_device(&host)
                .schedule(&policy)?
                .await?
                .reshape(&[end - start, 128])?;
            let output = api::zeros::<bf16>(&[end - start, 128])
                .schedule(&policy)?
                .await?;
            with_context(|exec| {
                value(unsafe {
                    (|| -> Result<()> {
                        exec.retain(state.into_shared_alias())?;
                        exec.retain(output.into_shared_alias())?;
                        kernels::conv(
                            output.device_pointer(),
                            state.device_pointer(),
                            &input,
                            &weights,
                        )
                        .generics(vec![bf16::DTYPE.to_string(), "128".into()])
                        .grid((1, 1, 1))
                        .execute(exec)?;
                        Ok(())
                    })()
                })
            })
            .schedule(&policy)?
            .await??;
            let actual = output.to_host_vec().schedule(&policy)?.await?;
            for (row, token) in (start..end).enumerate() {
                let sum: f32 = ((token + 1).saturating_sub(3)..=token + 1)
                    .filter(|v| *v > 0)
                    .map(|v| v as f32)
                    .sum();
                let expected = bf16::from_f32(sum / (1.0 + (-sum).exp()));
                assert!(actual[row * 128..(row + 1) * 128]
                    .iter()
                    .all(|v| *v == expected));
            }
            let state_values = unsafe { state.into_shared_alias() };
            let actual = state_values.to_host_vec().schedule(&policy)?.await?;
            for plane in 0..3 {
                let value = (end as i32 - 2 + plane as i32).max(0) as f32;
                assert!(
                    actual[plane * 128..(plane + 1) * 128]
                        .iter()
                        .all(|v| *v == bf16::from_f32(value)),
                    "chunk={chunk}, end={end}, plane={plane}"
                );
            }
            start = end;
        }
    }
    Ok(())
}

#[tokio::test]
#[ignore = "requires QWEN35_TEST_MODEL (the independent tiny fixture)"]
async fn pooled_state_and_runner_match_standalone() -> Result<()> {
    check_pooled_runner::<cutile::core::bf16>().await
}

#[tokio::test]
#[ignore = "requires QWEN35_TEST_MODEL (the independent tiny fixture)"]
async fn fp32_pooled_state_and_runner_match_standalone() -> Result<()> {
    check_pooled_runner::<f32>().await
}

async fn check_pooled_runner<
    T: crate::runner::loader::WeightElement + cutile::core::ElementType,
>() -> Result<()> {
    use crate::{
        runner::{
            kv_pool::{KVContext, KVPool},
            model::{CausalModel, ModelInput, Qwen3_5StatePool},
            model_runner::RunnerBuilder,
            PrefixCacheConfig,
        },
        scheduler::Scheduler,
        types::{SamplingConfig, SessionRequest},
    };
    use std::num::NonZeroUsize;
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    let _gpu = crate::runner::GPU_TEST_LOCK.lock().await;
    let policy = cuda_async::device_context::global_policy(0)?;
    let model =
        Qwen3_5Engine::<T>::load(&PathBuf::from(std::env::var("QWEN35_TEST_MODEL")?), &policy)
            .await?;
    let mut config = model.kv_pool_config(32)?;
    config.max_rows = 2;
    config.num_slots = 12;
    let pool = Qwen3_5StatePool::new(&model, config, &policy).await?;
    let initial = pool.metrics();
    let recurrent_bytes = model.config().text_config.state_bytes::<T>(0).unwrap();
    let token_bytes = model.config().text_config.state_bytes::<T>(1).unwrap() - recurrent_bytes;
    {
        let mut a = KVContext::new(pool.clone())?;
        let mut b = KVContext::new(pool.clone())?;
        assert!(KVContext::new(pool.clone()).is_err());
        // Opening a session reserves a mapping row, not its maximum token history.
        assert_eq!(pool.metrics().layers.assigned_bytes, 0);
        let ids = [1, 7, 9, 3, 12];
        let mut reference = model.new_context(32).await?;
        for (end, first) in [(2, true), (2, false), (5, true)] {
            let input = if first { &mut a } else { &mut b };
            let expected = if end == 2 && input.ready_len() == 0 {
                let mut fresh = model.new_context(32).await?;
                model.step_with_logits(&ids[..end], &mut fresh).await?.0
            } else {
                model.step_with_logits(&ids[..end], &mut reference).await?.0
            };
            let actual = model
                .next_tokens(
                    &mut [ModelInput {
                        token_ids: (&ids[..end]).into(),
                        kv: input,
                    }],
                    policy.clone(),
                )
                .await;
            assert_eq!(actual[0].as_ref().unwrap(), &expected);
            assert_eq!(input.ready_len(), end);
            assert!(!input.is_pending());
        }
        // A's later slots are separated from its first slots by B's allocation.
        assert_eq!(
            pool.metrics().layers.assigned_bytes,
            7 * token_bytes + 2 * recurrent_bytes
        );
        let mut extended = ids.to_vec();
        let expected_first = model
            .step_with_logits(&[1, 7, 9, 3, 12, 8], &mut reference)
            .await?
            .0;
        extended.push(8);
        let bad = [1, 8, 9];
        let results = model
            .next_tokens(
                &mut [
                    ModelInput {
                        token_ids: extended.as_slice().into(),
                        kv: &mut a,
                    },
                    ModelInput {
                        token_ids: bad.as_slice().into(),
                        kv: &mut b,
                    },
                ],
                policy.clone(),
            )
            .await;
        assert_eq!(*results[0].as_ref().unwrap(), expected_first);
        assert!(results[1].is_err());
        assert_eq!(b.ready_len(), 2);
        assert_eq!(
            pool.metrics().layers.assigned_bytes,
            8 * token_bytes + 2 * recurrent_bytes
        );
        // Five more tokens exceed the four remaining slots: no partial logical commit.
        extended.extend_from_slice(&[3, 3, 3, 3, 3]);
        assert!(model
            .next_tokens(
                &mut [ModelInput {
                    token_ids: extended.as_slice().into(),
                    kv: &mut a
                }],
                policy.clone()
            )
            .await[0]
            .is_err());
        assert!(!a.has_owned_pool_state());
        assert!(model
            .next_tokens(
                &mut [ModelInput {
                    token_ids: extended.as_slice().into(),
                    kv: &mut a,
                }],
                policy.clone()
            )
            .await[0]
            .is_err());
    }
    assert_eq!(pool.metrics(), initial);
    assert!(pool.can_append(12, 2));

    // Low-level pool calls reject another pool's state before initialization or reservation.
    {
        let other = Qwen3_5StatePool::new(&model, config, &policy).await?;
        let other_initial = other.metrics();
        let mut raw = pool.alloc_state()?;
        let wrong_pool = with_context(|exec| value(other.prepare_append(&mut raw, 0..1, exec)))
            .schedule(&policy)?
            .await?;
        assert!(wrong_pool.is_err());
        pool.release_state(raw);
        assert_eq!(other.metrics(), other_initial);
        assert_eq!(pool.metrics(), initial);
    }

    // No view without a pending append, and no backend commit without a begun update.
    {
        let mut incomplete = KVContext::new(pool.clone())?;
        let missing = with_context(|exec| value(incomplete.execution_view(&[1], exec).map(drop)))
            .schedule(&policy)?
            .await?;
        assert!(missing.is_err());
        with_context(|exec| value(incomplete.prepare_append(1, exec)))
            .schedule(&policy)?
            .await??;
        assert!(incomplete.commit_append().is_err());
        assert_eq!(incomplete.ready_len(), 0);
        assert!(incomplete.is_pending());
    }
    assert_eq!(pool.metrics(), initial);
    // Dropping a view after a failed submission cannot publish or roll back mutable state.
    {
        let mut failed = KVContext::new(pool.clone())?;
        let failure = with_context(|exec| {
            value((|| -> Result<()> {
                failed.prepare_append(1, exec)?;
                let view = failed.execution_view(&[1], exec)?;
                drop(view);
                ensure!(
                    failed.ready_len() == 0 && failed.is_pending(),
                    "view published progress"
                );
                anyhow::bail!("injected submission failure");
            })())
        })
        .schedule(&policy)?
        .await?;
        assert!(failure.is_err());
        assert!(model
            .next_tokens(
                &mut [ModelInput {
                    token_ids: [1, 7].as_slice().into(),
                    kv: &mut failed
                }],
                policy.clone()
            )
            .await[0]
            .is_err());
        assert!(!failed.has_owned_pool_state());
    }
    assert_eq!(pool.metrics(), initial);

    let prompt = "hello world again";
    let delta = " next";
    let mut history = model
        .tokenizer()
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?
        .get_ids()
        .to_vec();
    let mut reference = model.new_context(32).await?;
    let mut expected = Vec::with_capacity(2);
    for _ in 0..2 {
        let next = model.step_with_logits(&history, &mut reference).await?.0;
        if model.is_eos(next) {
            break;
        }
        expected.push(next);
        history.push(next);
    }
    let expected_text = model
        .tokenizer()
        .decode(&expected, true)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    history.extend_from_slice(
        model
            .tokenizer()
            .encode(delta, false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .get_ids(),
    );
    let next = model.step_with_logits(&history, &mut reference).await?.0;
    let delta_text = if model.is_eos(next) {
        String::new()
    } else {
        model
            .tokenizer()
            .decode(&[next], true)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
    };
    drop(reference);
    let cancel = CancellationToken::new();
    let (commands, command_rx) = mpsc::unbounded_channel();
    let (replies, reply_rx) = mpsc::unbounded_channel();
    let scheduler = Scheduler::new(cancel.child_token(), commands, reply_rx);
    let owned_pool = pool.clone();
    let runner_policy = policy.clone();
    let runner = RunnerBuilder::default()
        .with_max_batch_tokens(NonZeroUsize::new(2).unwrap())
        .with_overlap(false)
        .with_model_factory(move || async move { Ok(model) })
        .with_pool_factory(move |_| async move { Ok(owned_pool) })
        .with_policy_factory(move || Ok(runner_policy))
        .build(command_rx, replies)
        .await?;
    let task = tokio::spawn(runner.run(cancel.child_token()));
    let checked: Result<()> = async {
        let session = scheduler.open_session().await?;
        let mut request = SessionRequest::new(Some(session), prompt.to_owned());
        request.sampling = Some(SamplingConfig {
            max_new_tokens: Some(2),
            ..Default::default()
        });
        assert_eq!(
            scheduler.execute(request.clone()).await?.output,
            expected_text
        );
        let mut continuation = SessionRequest::new(Some(session), delta.to_owned());
        continuation.sampling = Some(SamplingConfig {
            max_new_tokens: Some(1),
            ..Default::default()
        });
        assert_eq!(scheduler.execute(continuation).await?.output, delta_text);
        scheduler.close_session(session).await?;
        assert_eq!(pool.metrics(), initial);
        request.session_id = None;
        for _ in 0..2 {
            assert_eq!(
                scheduler.execute(request.clone()).await?.output,
                expected_text
            );
        }
        Ok(())
    }
    .await;
    cancel.cancel();
    scheduler.wait_stopped().await;
    task.await?;
    checked?;
    assert_eq!(pool.metrics(), initial);
    assert!(pool.can_append(12, 2));
    // Unsupported prefix snapshots must be rejected before starting a Runner.
    let prefix_model =
        Qwen3_5Engine::<T>::load(&PathBuf::from(std::env::var("QWEN35_TEST_MODEL")?), &policy)
            .await?;
    let (_commands, command_rx) = mpsc::unbounded_channel();
    let (replies, _reply_rx) = mpsc::unbounded_channel();
    let owned_pool = Qwen3_5StatePool::new(&prefix_model, config, &policy).await?;
    let prefix_policy = policy.clone();
    let result = RunnerBuilder::default()
        .with_prefix_cache(PrefixCacheConfig::default())
        .with_model_factory(move || async move { Ok(prefix_model) })
        .with_pool_factory(move |_| async move { Ok(owned_pool) })
        .with_policy_factory(move || Ok(prefix_policy))
        .build(command_rx, replies)
        .await;
    assert!(matches!(
        result,
        Err(crate::error::EngineError::InvalidInput(_))
    ));

    Ok(())
}
