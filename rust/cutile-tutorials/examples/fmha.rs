#![allow(unused_variables)]

use cuda_async::prelude::DeviceOp;
use cuda_core::Device;
use cutile::api::{randn, zeros};
use cutile::error::Error;
use cutile::tensor::{IntoPartition, Tensor, ToHostVec};
use cutile::tile_kernel::TileKernel;
use std::sync::Arc;

/*
Attention(Q, K, V) = softmax(Q @ K^T / √d) @ V

Algorithm:
For each Q tile (row block of the output):
    Initialize: max = -∞, sum = 0, output = 0

    For each K,V tile (streaming through the sequence):
        1. Compute attention scores: scores = Q_tile @ K_tile^T
        2. Scale: scores = scores / √d
        3. Update running max: new_max = max(current_max, max(scores))
        4. Compute correction factor: α = exp(old_max - new_max)
        5. Rescale previous results: output *= α, sum *= α
        6. Compute new exponentials: P = exp(scores - new_max)
        7. Update sum: sum += sum(P)
        8. Accumulate: output += P @ V_tile
        9. Update max: current_max = new_max

    Normalize: output = output / sum
 */

#[cutile::module]
mod my_module {
    use cutile::core::*;
    #[cutile::entry()]
    fn fmha<const BM: i32, const BN: i32, const D: i32>(
        out: &mut Tensor<f32, { [1, BM, D] }>,
        q: &Tensor<f32, { [-1, -1, -1, -1] }>, // (B, H, M, D)
        k: &Tensor<f32, { [-1, -1, -1, -1] }>, // (B, H, N, D)
        v: &Tensor<f32, { [-1, -1, -1, -1] }>, // (B, H, N, D)
        qk_scale: f32,
    ) {
        let pid0 = program_id(0);
        let h = q.shape()[1];
        let batch_idx = pid0 / h;
        let head_idx = pid0 % h;
        let q_m_idx = program_id(1);
        let two: Tile<f32, { [] }> = constant(2.0f32, shape![]);
        let log2: f32 = tile_to_scalar(log(two));
        let q_scale: f32 = qk_scale;
        let q_scale = q_scale.broadcast(shape![BM, BN]);

        let part_q = q.partition(shape![1, 1, BM, D]);
        let part_k = k.partition(shape![1, 1, BN, D]);
        let part_v = v.partition(shape![1, 1, BN, D]);

        let q_tile = part_q.load([batch_idx, head_idx, q_m_idx, 0]);
        let q_tile = q_tile.reshape(shape![BM, D]);

        let mut max_i = constant(f32::NEG_INFINITY, shape![BM, 1]);
        let mut sum_i = constant(0.0, shape![BM, 1]);
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, shape![BM, D]);

        for k_col in 0..k.shape()[2] / BN {
            let k_tile = part_k
                .load([batch_idx, head_idx, k_col, 0])
                .reshape(shape![BN, D]);
            let k_tile = k_tile.transpose();
            let mut scores: Tile<f32, { [BM, BN] }> = constant(f32::ZERO, shape![BM, BN]);
            // 1. Compute attention scores: scores = Q_tile @ K_tile^T
            scores = mma(q_tile, k_tile, scores);
            // 2. Scale: scores = scores / √d
            scores = scores * q_scale;
            // 3. Update running max: new_max = max(current_max, max(scores))
            let new_max: Tile<f32, { [BM] }> = reduce_max(scores, 1);
            let new_max = new_max.reshape(shape![BM, 1]);
            let new_max = max_tile(max_i, new_max);
            // 4. Compute correction factor: α = exp(old_max - new_max)
            let alpha: Tile<f32, { [BM, 1] }> = exp(max_i - new_max);
            // 5. Rescale previous results: output *= α, sum *= α
            acc = acc * broadcast(alpha, shape![BM, D]);
            sum_i = sum_i * alpha;
            // 6. Compute new exponentials: P = exp(scores - new_max)
            let p: Tile<f32, { [BM, BN] }> = exp(scores - new_max.broadcast(shape![BM, BN]));
            // 7. Update sum: sum += sum(P)
            let sum_ij: Tile<f32, { [BM, 1] }> = reduce_sum(p, 1);
            let sum_ij = sum_ij.reshape(shape![BM, 1]);
            sum_i = sum_i + sum_ij;
            // 8. Accumulate: output += P @ V_tile
            let v_tile = part_v
                .load([batch_idx, head_idx, k_col, 0])
                .reshape(shape![BN, D]);
            acc = mma(p, v_tile, acc);

            // 9. Update max: current_max = new_max
            max_i = new_max;
        }
        // Normalize: output = output / sum
        acc = acc / broadcast(sum_i, shape![BM, D]);
        out.store(acc.reshape(shape![1, BM, D]));
    }
}

use my_module::fmha;

fn index4(
    batch: usize,
    head: usize,
    seq: usize,
    dim: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> usize {
    (((batch * heads + head) * seq_len + seq) * head_dim) + dim
}

fn fmha_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch_size: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch_size * heads * seq_len * head_dim];

    for batch in 0..batch_size {
        for head in 0..heads {
            for query_idx in 0..seq_len {
                let mut scores = vec![0.0f32; seq_len];
                for key_idx in 0..seq_len {
                    let mut dot = 0.0f32;
                    for dim in 0..head_dim {
                        dot += q[index4(batch, head, query_idx, dim, heads, seq_len, head_dim)]
                            * k[index4(batch, head, key_idx, dim, heads, seq_len, head_dim)];
                    }
                    scores[key_idx] = dot * scale;
                }

                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut denominator = 0.0f32;
                for score in &mut scores {
                    *score = (*score - max_score).exp();
                    denominator += *score;
                }

                for dim in 0..head_dim {
                    let mut weighted_sum = 0.0f32;
                    for key_idx in 0..seq_len {
                        weighted_sum += scores[key_idx]
                            * v[index4(batch, head, key_idx, dim, heads, seq_len, head_dim)];
                    }
                    output[index4(batch, head, query_idx, dim, heads, seq_len, head_dim)] =
                        weighted_sum / denominator;
                }
            }
        }
    }

    output
}

fn main() -> Result<(), Error> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    let (batch, heads, seq_len, head_dim) = (2, 4, 128, 64);
    let (bm, bn) = (64, 32);

    let seed = 42u64;
    let q: Arc<Tensor<f32>> = randn(0., 1., [batch, heads, seq_len, head_dim], Some(seed))
        .sync_on(&stream)?
        .into();
    let k: Arc<Tensor<f32>> = randn(0., 1., [batch, heads, seq_len, head_dim], Some(seed + 1))
        .sync_on(&stream)?
        .into();
    let v: Arc<Tensor<f32>> = randn(0., 1., [batch, heads, seq_len, head_dim], Some(seed + 2))
        .sync_on(&stream)?
        .into();

    let out = zeros(&[batch * heads, seq_len, head_dim])
        .sync_on(&stream)?
        .partition([1, bm, head_dim]);

    let qk_scale = 1.0 / f32::sqrt(head_dim as f32);
    let generics = vec![bm.to_string(), bn.to_string(), head_dim.to_string()];

    let q_ref = q.clone();
    let k_ref = k.clone();
    let v_ref = v.clone();
    let (out, _, _, _, _) = fmha(out, q, k, v, qk_scale)
        .generics(generics)
        .sync_on(&stream)?;

    let out_host: Vec<f32> = out.unpartition().to_host_vec().sync_on(&stream)?;
    let q_host = q_ref.to_host_vec().sync_on(&stream)?;
    let k_host = k_ref.to_host_vec().sync_on(&stream)?;
    let v_host = v_ref.to_host_vec().sync_on(&stream)?;
    let expected = fmha_reference(
        &q_host, &k_host, &v_host, batch, heads, seq_len, head_dim, qk_scale,
    );

    let mut max_abs_error = 0.0f32;
    let mut squared_error_sum = 0.0f64;
    for (&actual, &reference) in out_host.iter().zip(&expected) {
        let error = (actual - reference).abs();
        max_abs_error = max_abs_error.max(error);
        squared_error_sum += f64::from(error) * f64::from(error);
    }
    let rms_error = (squared_error_sum / expected.len() as f64).sqrt();
    println!("Output length: {}", out_host.len());
    println!("Against standard softmax(QK^T * scale) @ V reference:");
    println!("  max absolute error: {max_abs_error:.6e}");
    println!("  RMS error:          {rms_error:.6e}");

    Ok(())
}
