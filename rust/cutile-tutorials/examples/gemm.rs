#![allow(unused_variables)]

use cuda_async::prelude::DeviceOp;
use cuda_core::Device;
use cutile::api::{full, zeros};
use cutile::error::Error;
use cutile::tensor::{IntoPartition, Tensor, ToHostVec};
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
        let mut tile_c = load_tile_mut(c);

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

    let (bm, bn, bk) = (16usize, 16usize, 8usize);
    let (m, n, k) = (64usize, 64usize, 64usize);

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
    let (c, a, b) = gemm(c, a, b).generics(generics).sync_on(&stream)?;
    let a = a.to_host_vec().sync_on(&stream)?;
    println!("a[0] = {:?} (expected: 2.0)", a[0]);
    let c_host: Vec<f32> = c.unpartition().to_host_vec().sync_on(&stream)?;

    println!("c[0] = {:?} (expected: 6.0 * 64.0 = 384.0)", c_host[0]);
    Ok(())
}
