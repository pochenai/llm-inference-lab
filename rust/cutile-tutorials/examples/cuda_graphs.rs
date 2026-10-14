//! Repeated GEMM execution using CUDA Graph replay.
//!
//! Capture the same fixed-buffer GEMM used by `cuda_graphs_eager.rs`, then
//! replay it repeatedly. Capture and warmup are excluded from the timing.

use cuda_async::prelude::DeviceOp;
use cuda_core::Device;
use cutile::api::{full, zeros};
use cutile::error::Error;
use cutile::tensor::{IntoPartition, Tensor, ToHostVec};
use cutile::tile_kernel::TileKernel;
use std::time::Instant;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn gemm<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        c: &mut Tensor<f32, { [BM, BN] }>,
        a: &Tensor<f32, { [-1, -1] }>,
        b: &Tensor<f32, { [-1, -1] }>,
    ) {
        let row = program_id(0);
        let col = program_id(1);
        let part_a = a.partition(shape![BM, BK]);
        let part_b = b.partition(shape![BK, BN]);
        let mut tile_c: Tile<f32, { [BM, BN] }> = constant(0.0, shape![BM, BN]);
        for idx in 0..(K / BK) {
            tile_c = mma(part_a.load([row, idx]), part_b.load([idx, col]), tile_c);
        }
        c.store(tile_c);
    }
}

use kernels::gemm;

fn main() -> Result<(), Error> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let (m, n, k) = (512usize, 512usize, 512usize);
    let (bm, bn, bk) = (16usize, 16usize, 8usize);
    let iterations = 100usize;

    let a: Tensor<f32> = full(2.0, &[m, k]).sync_on(&stream)?.into();
    let b: Tensor<f32> = full(3.0, &[k, n]).sync_on(&stream)?.into();
    let c = zeros(&[m, n]).sync_on(&stream)?.partition([bm, bn]);
    let generics = vec![
        bm.to_string(),
        bn.to_string(),
        bk.to_string(),
        k.to_string(),
    ];

    // Capture one GEMM launch. Its buffer addresses remain fixed for replay.
    let mut graph = gemm(c, a, b).generics(generics).graph_on(stream.clone())?;
    let (c, _a, _b) = graph.take_output().expect("capture returned GEMM buffers");
    graph.launch().sync_on(&stream)?; // Warmup replay before timing.

    let start = Instant::now();
    for _ in 0..iterations {
        graph.launch().sync_on(&stream)?;
    }
    let elapsed = start.elapsed();
    let output: Vec<f32> = c.unpartition().to_host_vec().sync_on(&stream)?;
    println!("graph c[0] = {} (expected {})", output[0], 6.0 * k as f32);
    println!(
        "graph: {iterations} replays in {:.3} ms ({:.3} us/replay)",
        elapsed.as_secs_f64() * 1e3,
        elapsed.as_secs_f64() * 1e6 / iterations as f64
    );
    Ok(())
}
