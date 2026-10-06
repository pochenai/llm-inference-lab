#![allow(unused_variables)]

use cuda_async::prelude::DeviceOp;
use cuda_core::Device;
use cutile::api::{ones, zeros};
use cutile::error::Error;
use cutile::tensor::{IntoPartition, Tensor, ToHostVec};

#[cutile::module]
mod vec_add_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn add<const S: [i32; 2]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1, -1] }>,
        y: &Tensor<f32, { [-1, -1] }>,
    ) {
        let tile_x = x.load_like(z);
        let tile_y = y.load_like(z);
        let one = 1.0f32.broadcast(z.shape());
        z.store(tile_x + tile_y + one);
    }
}

use vec_add_module::add;

fn main() -> Result<(), Error> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    let x: Tensor<f32> = ones(&[32, 32]).sync_on(&stream)?.into();
    let y: Tensor<f32> = ones(&[32, 32]).sync_on(&stream)?.into();

    let z = zeros(&[32, 32]).sync_on(&stream)?.partition([4, 4]);

    let (z, x, _y) = add(z, x, y).sync_on(&stream)?;
    let x = x.to_host_vec().sync_on(&stream)?;
    println!("x[0] = {:?} (expected: 1.0)", x[0]);
    let z_host: Vec<f32> = z.unpartition().to_host_vec().sync_on(&stream)?;

    println!("z[0] = {:?} (expected: 3.0)", z_host[0]);
    Ok(())
}
