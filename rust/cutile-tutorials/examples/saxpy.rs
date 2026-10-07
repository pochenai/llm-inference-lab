use cuda_async::prelude::DeviceOp;
use cuda_core::Device;
use cutile::api::arange;
use cutile::error::Error;
use cutile::tensor::{IntoPartition, Reshape, Tensor, ToHostVec};

#[cutile::module]
mod saxpy_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn saxpy<const S: [i32; 2]>(y: &mut Tensor<f32, S>, a: f32, x: &Tensor<f32, { [-1, -1] }>) {
        let tile_x = x.load_like(y);
        let tile_y = y.load();
        let tile_a = a.broadcast(y.shape());
        // y = a*x + y (a*x is element-wise multiplication)
        y.store(tile_a * tile_x + tile_y);
    }
}

use saxpy_module::saxpy;

fn main() -> Result<(), Error> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    let input: Tensor<f32> = arange(32).sync_on(&stream)?.into();
    let x: Tensor<f32> = input.dup().sync_on(&stream)?.reshape(&[4, 8])?;
    let y = input
        .dup()
        .sync_on(&stream)?
        .reshape(&[4, 8])?
        .partition([2, 2]);

    let (y, _, _) = saxpy(y, 2.0, &x).sync_on(&stream)?;
    let x = x.to_host_vec().sync_on(&stream)?;
    println!("x[0] = {:?} (expected: 0.0)", x[0]);
    let y_host: Vec<f32> = y.unpartition().to_host_vec().sync_on(&stream)?;

    println!("y[3] = {:?} (expected: 3*2 + 3 = 9.0)", y_host[3]);
    Ok(())
}
