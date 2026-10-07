use cuda_async::prelude::DeviceOp;
use cuda_core::Device;
use cutile::api::{full, zeros};
use cutile::bench::{do_bench, BenchOptions};
use cutile::error::Error;
use cutile::tensor::{IntoPartition, Tensor, ToHostVec};
use cutile::tile_kernel::TileKernel;
use cutile::DType;

#[cutile::module]
mod persistent_module {
    use cutile::core::*;

    #[cutile::entry(
        // optimization_hints = (
        //     sm_120 = (num_cta_in_cga = 2,),
        //     sm_100 = (num_cta_in_cga = 2,),
        // ),
        deny_in_kernel_checks = true,
    )]
    fn gemm_persistent<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const MAP_SHAPE: [i32; 2],
    >(
        mut z: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<T, { [-1, -1] }>,
        y: &Tensor<T, { [-1, -1] }>,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_y = y.partition(shape![BK, BN]);

        // Each mapped output index owns one output tile. The index stream
        // traverses the output grid while each tile accumulates across K.
        for out_idx in z.iter_indices() {
            let (bid_m, bid_n) = out_idx.components();
            let mut tile_z: Tile<T, { [BM, BN] }> = constant(T::ZERO, shape![BM, BN]);

            for k_tile in 0i32..num_tiles(&part_x, 1) {
                let tile_x = part_x.load([bid_m, k_tile]);
                let tile_y = part_y.load([k_tile, bid_n]);
                tile_z = mma(tile_x, tile_y, tile_z);
            }

            z.store(tile_z, out_idx);
        }
    }
}

use persistent_module::gemm_persistent;

fn main() -> Result<(), Error> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    let d = 2048;
    let (bm, bn, bk) = (16, 16, 8);
    let (m, n, k) = (d, d, d);
    let num_tile_blocks = (m / bm * (n / bn)) as u32;

    let x: Tensor<f32> = full(2.0, &[m, k]).sync_on(&stream)?.into();
    let y: Tensor<f32> = full(3.0, &[k, n]).sync_on(&stream)?.into();
    let z = zeros::<f32>(&[m, n]).sync_on(&stream)?;

    let generics = vec![
        f32::DTYPE.as_str().to_string(),
        bm.to_string(),
        bn.to_string(),
        bk.to_string(),
        "4".to_string(),
        "1".to_string(),
    ];

    let mut z = Some(z);
    let mut x = Some(x);
    let mut y = Some(y);

    let measurement = do_bench(&stream, &BenchOptions::default(), |bench_stream| {
        let mapped_z = z
            .take()
            .expect("benchmark output tensor is available")
            .partition([bm, bn])
            .map([4, 1], num_tile_blocks);
        let (next_z, next_x, next_y) = gemm_persistent(
            mapped_z,
            x.take().expect("benchmark input X is available"),
            y.take().expect("benchmark input Y is available"),
        )
        .generics(generics.clone())
        .sync_on(bench_stream)?;
        z = Some(next_z.unpartition());
        x = Some(next_x);
        y = Some(next_y);
        Ok(())
    })?;
    let z = z.expect("benchmark returned the output tensor");
    let z_host: Vec<f32> = z.to_host_vec().sync_on(&stream)?;

    println!(
        "z[0] = {:?} (expected: 2.0 * 3.0 * {k} = {})",
        z_host[0],
        2.0 * 3.0 * k as f32
    );
    println!(
        "cuTile bench: median {:.3} ms, mean {:.3} ms, {} reps",
        measurement.median_ms(),
        measurement.mean_ms(),
        measurement.reps()
    );
    Ok(())
}
