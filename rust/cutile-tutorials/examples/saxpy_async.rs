use std::fmt::Display;
use std::ops::{Add, Mul};

use cuda_async::device_operation::{value, Unzippable3, Zippable};
use cuda_async::zip;
use cutile::api::arange;
use cutile::error::Error;
use cutile::tensor::{IntoPartition, Reshape, ToHostVec, Unpartition};
use cutile::tile_kernel::ToHostVecOp;
use cutile::{core::ElementType, DType};

#[cutile::module]
mod saxpy_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn saxpy<T: ElementType, const S: [i32; 2]>(
        y: &mut Tensor<T, S>,
        a: T,
        x: &Tensor<T, { [-1, -1] }>,
    ) {
        let tile_x = x.load_like(y);
        let tile_y = y.load();
        let tile_a = a.broadcast(y.shape());
        // y = a*x + y (a*x is element-wise multiplication)
        y.store(tile_a * tile_x + tile_y);
    }
}

use saxpy_module::saxpy;

async fn execute<
    T: DType + ElementType + Display + PartialEq + Mul<Output = T> + Add<Output = T>,
>(
    size: usize,
) -> Result<(), Error> {
    assert!(
        size >= 32 && size % 32 == 0,
        "size must be a positive multiple of 32"
    );
    let shape = [4, 8];

    let (input, x, y) = zip!(arange::<T>(size), arange::<T>(size), arange::<T>(size)).await?;
    let input = input.reshape(&shape)?;
    let x = x.reshape(&shape)?;
    let y = y.reshape(&shape)?;
    let a = value(T::one() + T::one());

    let y_host = saxpy(y.partition([2, 4]), a, x)
        .first()
        .unpartition()
        .to_host_vec()
        .await?;

    println!(
        "input[3] = {:?} (expected: 3)",
        input.to_host_vec().await?[3]
    );
    println!("y[3] = {:?} (expected: 3*2 + 3 = 9)", y_host[3]);

    Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Error> {
    cuda_async::device_context::set_default_device(0);
    execute::<f32>(2usize.pow(5)).await?;
    execute::<f64>(2usize.pow(5)).await?;
    Ok(())
}
