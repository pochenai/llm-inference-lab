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
    let engine = Qwen3_5Engine::load(&model_dir, &policy).await?;
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
