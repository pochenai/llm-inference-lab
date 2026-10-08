use cuda_async::prelude::DeviceOp;
use cuda_core::Device;
use cutile::api::{randn, zeros};
use cutile::error::Error;
use cutile::tensor::{IntoPartition, Tensor, ToHostVec};
use cutile::tile_kernel::TileKernel;
use std::sync::Arc;

// Causal masking has two cases. A future KV tile is skipped entirely:
//
// | query \\ key | 3 | 4 | 5 |
// |-------------|---|---|---|
// | 0           | — | — | — |
// | 1           | — | — | — |
// | 2           | — | — | — |
//
// In the boundary tile, MMA computes the full score tile first, then the
// upper-triangular future positions are masked:
//
// | query \\ key | 0 | 1 | 2 |
// |-------------|---|---|---|
// | 0           | ✓ | × | × |
// | 1           | ✓ | ✓ | × |
// | 2           | ✓ | ✓ | ✓ |
#[cutile::module]
mod fmha_causal_module {
    use cutile::core::*;

    #[cutile::entry(print_ir = false)]
    fn fmha_causal<const BM: i32, const BN: i32, const D: i32>(
        out: &mut Tensor<f32, { [1, BM, D] }>,
        q: &Tensor<f32, { [-1, -1, -1, -1] }>, // (B, H, M, D)
        k: &Tensor<f32, { [-1, -1, -1, -1] }>, // (B, H, N, D)
        v: &Tensor<f32, { [-1, -1, -1, -1] }>, // (B, H, N, D)
        qk_scale: f32,
    ) {
        let pid0 = program_id(0);
        let heads = q.shape()[1];
        let batch_idx = pid0 / heads;
        let head_idx = pid0 % heads;
        let q_tile_idx = program_id(1);

        let q_part = q.partition(shape![1, 1, BM, D]);
        let k_part = k.partition(shape![1, 1, BN, D]);
        let v_part = v.partition(shape![1, 1, BN, D]);

        let q_tile = q_part
            .load([batch_idx, head_idx, q_tile_idx, 0i32])
            .reshape(shape![BM, D]);

        let q_tile_start = q_tile_idx * BM;
        let q_tile_end = min(q_tile_start + BM, q.shape()[2]);
        // Skip future kv tiles
        let num_kv_tiles = ceil_div(q_tile_end, BN);

        let q_rows = broadcast_scalar(q_tile_start, shape![BM]) + iota(shape![BM]);
        let q_rows: Tile<i32, { [BM, 1] }> = q_rows.reshape(shape![BM, 1]);
        let q_rows: Tile<i32, { [BM, BN] }> = q_rows.broadcast(shape![BM, BN]);

        let mut running_max: Tile<f32, { [BM, 1] }> = constant(f32::NEG_INFINITY, shape![BM, 1]);
        let mut running_sum: Tile<f32, { [BM, 1] }> = constant(0.0f32, shape![BM, 1]);
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, shape![BM, D]);
        let qk_scale: Tile<f32, { [BM, BN] }> = qk_scale.broadcast(shape![BM, BN]);
        let mask_true: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        let mask_false: Tile<f32, { [BM, BN] }> = constant(f32::NEG_INFINITY, shape![BM, BN]);

        for kv_tile_idx in 0i32..num_kv_tiles {
            let k_tile = k_part
                .load([batch_idx, head_idx, kv_tile_idx, 0i32])
                .reshape(shape![BN, D]);
            let k_tile_t = k_tile.transpose();
            let mut scores: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
            scores = mma(q_tile, k_tile_t, scores);
            scores = scores * qk_scale;

            let k_cols = broadcast_scalar(kv_tile_idx * BN, shape![BN]) + iota(shape![BN]);
            let k_cols: Tile<i32, { [1, BN] }> = k_cols.reshape(shape![1, BN]);
            let k_cols: Tile<i32, { [BM, BN] }> = k_cols.broadcast(shape![BM, BN]);
            // Causal mask in boundry tile: 1 if q_row >= k_col, else 0
            let causal_mask = ge_tile(q_rows, k_cols);
            scores = scores + select(causal_mask, mask_true, mask_false);

            let tile_max: Tile<f32, { [BM] }> = reduce_max(scores, 1);
            let tile_max: Tile<f32, { [BM, 1] }> = tile_max.reshape(shape![BM, 1]);
            let next_max = max_tile(running_max, tile_max);
            let alpha = exp(running_max - next_max);

            let probabilities = exp(scores - next_max.broadcast(shape![BM, BN]));
            let tile_sum: Tile<f32, { [BM] }> = reduce_sum(probabilities, 1);
            let tile_sum: Tile<f32, { [BM, 1] }> = tile_sum.reshape(shape![BM, 1]);
            running_sum = running_sum * alpha + tile_sum;
            acc = acc * alpha.broadcast(shape![BM, D]);

            let v_tile = v_part
                .load([batch_idx, head_idx, kv_tile_idx, 0i32])
                .reshape(shape![BN, D]);
            acc = mma(probabilities, v_tile, acc);
            running_max = next_max;
        }

        let result = true_div(acc, running_sum.broadcast(shape![BM, D]));
        out.store(result.reshape(shape![1, BM, D]));
    }
}

use fmha_causal_module::fmha_causal;

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

fn causal_fmha_reference(
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
                let mut scores = vec![0.0f32; query_idx + 1];
                for key_idx in 0..=query_idx {
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
                    for key_idx in 0..=query_idx {
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

    let (batch, heads, seq_len, head_dim) = (2usize, 4usize, 128usize, 64usize);
    let (bm, bn) = (64usize, 32usize);
    let qk_scale = 1.0 / (head_dim as f32).sqrt();

    let q: Arc<Tensor<f32>> = randn(0.0, 1.0, [batch, heads, seq_len, head_dim], Some(42))
        .sync_on(&stream)?
        .into();
    let k: Arc<Tensor<f32>> = randn(0.0, 1.0, [batch, heads, seq_len, head_dim], Some(43))
        .sync_on(&stream)?
        .into();
    let v: Arc<Tensor<f32>> = randn(0.0, 1.0, [batch, heads, seq_len, head_dim], Some(44))
        .sync_on(&stream)?
        .into();

    let out = zeros(&[batch * heads, seq_len, head_dim])
        .sync_on(&stream)?
        .partition([1, bm, head_dim]);
    let generics = vec![bm.to_string(), bn.to_string(), head_dim.to_string()];

    let q_ref = q.clone();
    let k_ref = k.clone();
    let v_ref = v.clone();
    let (out, _, _, _, _) = fmha_causal(out, q, k, v, qk_scale)
        .generics(generics)
        .sync_on(&stream)?;

    let output = out.unpartition().to_host_vec().sync_on(&stream)?;
    let q_host = q_ref.to_host_vec().sync_on(&stream)?;
    let k_host = k_ref.to_host_vec().sync_on(&stream)?;
    let v_host = v_ref.to_host_vec().sync_on(&stream)?;
    let expected = causal_fmha_reference(
        &q_host, &k_host, &v_host, batch, heads, seq_len, head_dim, qk_scale,
    );

    let mut max_abs_error = 0.0f32;
    let mut squared_error_sum = 0.0f64;
    for (&actual, &reference) in output.iter().zip(&expected) {
        let error = (actual - reference).abs();
        max_abs_error = max_abs_error.max(error);
        squared_error_sum += f64::from(error) * f64::from(error);
    }
    let rms_error = (squared_error_sum / expected.len() as f64).sqrt();
    println!("Output length: {}", output.len());
    println!("Against causal softmax(QK^T * scale) @ V reference:");
    println!("  max absolute error: {max_abs_error:.6e}");
    println!("  RMS error:          {rms_error:.6e}");
    Ok(())
}
