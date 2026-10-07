#![allow(unused_variables)]

use cuda_async::prelude::DeviceOp;
use cuda_core::Device;
use cutile::api::{full, zeros};
use cutile::bench::{do_bench, BenchOptions};
use cutile::error::Error;
use cutile::tensor::{IntoPartition, Tensor, ToHostVec};
use cutile::tile_kernel::TileKernel;
use cutile::DType;

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

    #[cutile::entry(
        unchecked_accesses = true,
        optimization_hints = (
            sm_120 = (max_divisibility = 8,), // num_cta_in_cga会导致性能下降很严重
        )
    )]
    unsafe fn gemm_unsafe<T: ElementType, const BM: i32, const BN: i32, const BK: i32>(
        z: &mut Tensor<T, { [BM, BN] }>,
        x: &Tensor<T, { [-1, -1] }>,
        y: &Tensor<T, { [-1, -1] }>,
        k: i32,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_y = y.partition(shape![BK, BN]);
        let pid_m = program_id(0);
        let pid_n = program_id(1);
        let mut tile_z: Tile<T, { [BM, BN] }> = constant(T::ZERO, shape![BM, BN]);
        for i in 0i32..(k / BK) {
            let tile_x = part_x.load([pid_m, i]);
            let tile_y = part_y.load([i, pid_n]);
            tile_z = mma(tile_x, tile_y, tile_z);
        }
        z.store(tile_z);
    }
}

use my_module::*;

fn main() -> Result<(), Error> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    let d = 2048;
    let (bm, bn, bk) = (16, 16, 8);
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

    let mut c = Some(c);
    let mut a = Some(a);
    let mut b = Some(b);

    let measurement = do_bench(&stream, &BenchOptions::default(), |bench_stream| {
        let (next_c, next_a, next_b) = gemm(
            c.take().expect("benchmark output partition is available"),
            a.take().expect("benchmark input A is available"),
            b.take().expect("benchmark input B is available"),
        )
        .generics(generics.clone())
        .sync_on(bench_stream)?;
        c = Some(next_c);
        a = Some(next_a);
        b = Some(next_b);
        Ok(())
    })?;

    let c = c.expect("benchmark returned the output partition");
    let c_host: Vec<f32> = c.unpartition().to_host_vec().sync_on(&stream)?;

    println!(
        "c[0] = {:?} (expected: 6.0 * {k} = {})",
        c_host[0],
        6.0 * k as f32
    );
    println!(
        "gemm cuTile bench: median {:.3} ms, mean {:.3} ms, {} reps",
        measurement.median_ms(),
        measurement.mean_ms(),
        measurement.reps()
    );

    let mut unsafe_c = Some(zeros(&[m, n]).sync_on(&stream)?.partition([bm, bn]));
    let mut unsafe_a = a;
    let mut unsafe_b = b;
    let unsafe_generics = vec![
        f32::DTYPE.as_str().to_string(),
        bm.to_string(),
        bn.to_string(),
        bk.to_string(),
    ];
    let unsafe_measurement = do_bench(&stream, &BenchOptions::default(), |bench_stream| {
        let launch = unsafe {
            gemm_unsafe(
                unsafe_c
                    .take()
                    .expect("unsafe benchmark output partition is available"),
                unsafe_a
                    .take()
                    .expect("unsafe benchmark input A is available"),
                unsafe_b
                    .take()
                    .expect("unsafe benchmark input B is available"),
                k as i32,
            )
            .generics(unsafe_generics.clone())
        };
        let (next_c, next_a, next_b, _next_k) = launch.sync_on(bench_stream)?;
        unsafe_c = Some(next_c);
        unsafe_a = Some(next_a);
        unsafe_b = Some(next_b);
        Ok(())
    })?;

    let unsafe_c = unsafe_c.expect("unsafe benchmark returned the output partition");
    let unsafe_c_host: Vec<f32> = unsafe_c.unpartition().to_host_vec().sync_on(&stream)?;
    println!(
        "gemm_unsafe c[0] = {:?} (expected: 6.0 * {k} = {})",
        unsafe_c_host[0],
        6.0 * k as f32
    );
    println!(
        "gemm_unsafe cuTile bench: median {:.3} ms, mean {:.3} ms, {} reps",
        unsafe_measurement.median_ms(),
        unsafe_measurement.mean_ms(),
        unsafe_measurement.reps()
    );
    Ok(())
}
