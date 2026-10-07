#![allow(unused_variables)]

use cuda_async::prelude::DeviceOp;
use cuda_core::Device;
use cutile::api::{full, zeros};
use cutile::error::Error;
use cutile::tensor::{IntoPartition, Tensor, ToHostVec};
use cutile::DType;
use std::time::Instant;

#[cutile::module]
mod my_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn gemm<E: ElementType, const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        c: &mut Tensor<E, { [BM, BN] }>,
        a: &Tensor<E, { [-1, -1] }>,
        b: &Tensor<E, { [-1, -1] }>,
    ) {
        let row = program_id(0);
        let col = program_id(1);
        let part_x = a.partition(shape![BM, BK]);
        let part_y = b.partition(shape![BK, BN]);
        let mut tile_c: Tile<E, { [BM, BN] }> = constant(E::ZERO, shape![BM, BN]);

        for idx in 0..(K / BK) {
            let tile_a = part_x.load([row, idx]);
            let tile_b = part_y.load([idx, col]);
            // c = a * b + c (a * b is matrix multiplication)
            tile_c = mma(tile_a, tile_b, tile_c);
        }
        c.store(tile_c);
    }
}

use cutile::tile_kernel::TileKernel;
use my_module::gemm;

fn main() -> Result<(), Error> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    let d = 1024;
    let tile_d = d / 8;
    let (bm, bn, bk) = (tile_d, tile_d, tile_d);
    let (m, n, k) = (d, d, d);

    let a: Tensor<f32> = full(2.0, &[m, k]).sync_on(&stream)?.into();
    let b: Tensor<f32> = full(3.0, &[k, n]).sync_on(&stream)?.into();

    let c = zeros(&[m, n]).sync_on(&stream)?.partition([bm, bn]);

    let generics = vec![
        f32::DTYPE.as_str().to_string(),
        bm.to_string(),
        bn.to_string(),
        bk.to_string(),
        k.to_string(),
    ];

    let warmup_steps = 1;
    let runs = 2;
    let mut c = c;
    let mut a = a;
    let mut b = b;

    for _ in 0..warmup_steps {
        let (next_c, next_a, next_b) = gemm(c, a, b).generics(generics.clone()).sync_on(&stream)?;
        c = next_c;
        a = next_a;
        b = next_b;
    }

    let start = Instant::now();
    for _ in 0..runs {
        let (next_c, next_a, next_b) = gemm(c, a, b).generics(generics.clone()).sync_on(&stream)?;
        c = next_c;
        a = next_a;
        b = next_b;
    }
    let average_time = start.elapsed() / runs;

    let c_host: Vec<f32> = c.unpartition().to_host_vec().sync_on(&stream)?;

    println!("c[0] = {:?} (expected: 6.0 * 64.0 = 384.0)", c_host[0]);
    println!(
        "average time over {runs} runs (after {warmup_steps} warmup steps): {:.3} ms/run",
        average_time.as_secs_f64() * 1_000.0
    );
    Ok(())
}
