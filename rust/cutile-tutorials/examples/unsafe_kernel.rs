use cutile::api::{ones, zeros};
use cutile::error::{tensor_error_result, Error};
use cutile::tensor::{Tensor, ToHostVec};
use cutile::tile_kernel::TileKernel;
use cutile::{core::ElementType, DType};

#[cutile::module]
mod my_module {

    use cutile::core::*;

    unsafe fn get_tensor<T: ElementType>(ptr: *mut T, len: i32) -> Tensor<T, { [-1] }> {
        let shape: Shape<{ [-1] }> = Shape::<{ [-1] }> { dims: &[len] };
        let strides: Array<{ [-1] }> = Array::<{ [-1] }> { dims: &[1i32] };
        let ptr_tile: PointerTile<*mut T, { [] }> = pointer_to_tile(ptr);
        let tensor = unsafe { make_tensor_view(ptr_tile, shape, strides, new_token_unordered()) };
        tensor
    }

    #[cutile::entry()]
    unsafe fn add_ptr<T: ElementType>(z_ptr: *mut T, x_ptr: *mut T, y_ptr: *mut T, len: i32) {
        let mut z_tensor: Tensor<T, { [-1] }> = unsafe { get_tensor(z_ptr, len) };
        let x_tensor: Tensor<T, { [-1] }> = unsafe { get_tensor(x_ptr, len) };
        let y_tensor: Tensor<T, { [-1] }> = unsafe { get_tensor(y_ptr, len) };
        let pid = program_id(0);
        let tile_shape = shape![4i32];
        let tile_x = x_tensor.partition(tile_shape).load([pid]);
        let tile_y = y_tensor.partition(tile_shape).load([pid]);
        z_tensor
            .partition_mut(tile_shape)
            .store(tile_x + tile_y, [pid]);
    }
}

use my_module::add_ptr;

const TILE_SIZE: usize = 4;

/// Adds `x` and `y` into `z` after validating the raw-pointer kernel's assumptions.
async fn add<T: DType + ElementType>(
    z: &mut Tensor<T>,
    x: &Tensor<T>,
    y: &Tensor<T>,
) -> Result<(), Error> {
    let len = z.size();
    if x.size() != len || y.size() != len {
        return tensor_error_result("z, x, and y must have the same number of elements");
    }
    if len == 0 || len % TILE_SIZE != 0 {
        return tensor_error_result("tensor length must be a non-zero multiple of the tile size");
    }
    if !z.is_contiguous() || !x.is_contiguous() || !y.is_contiguous() {
        return tensor_error_result("z, x, and y must be contiguous tensors");
    }
    if z.shape().len() != 1 || x.shape().len() != 1 || y.shape().len() != 1 {
        return tensor_error_result("z, x, and y must be one-dimensional tensors");
    }
    if z.device_id() != x.device_id() || z.device_id() != y.device_id() {
        return tensor_error_result("z, x, and y must be on the same device");
    }
    let len_i32 = match i32::try_from(len) {
        Ok(len) => len,
        Err(_) => return tensor_error_result("tensor length exceeds the kernel's i32 limit"),
    };
    let grid_x = match u32::try_from(len / TILE_SIZE) {
        Ok(grid_x) => grid_x,
        Err(_) => return tensor_error_result("kernel grid exceeds the u32 limit"),
    };

    // The references remain live until the awaited launch completes, keeping all
    // allocations alive while the kernel uses their raw device pointers.
    unsafe {
        add_ptr(
            z.device_pointer(),
            x.device_pointer(),
            y.device_pointer(),
            len_i32,
        )
    }
    .grid((grid_x, 1, 1))
    .await?;

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), cutile::error::Error> {
    let len = 2usize.pow(5);

    // Initialize tensors.
    let mut z: Tensor<f32> = zeros(&[len]).await?;
    let x: Tensor<f32> = ones(&[len]).await?;
    let y: Tensor<f32> = ones(&[len]).await?;

    // Unsafe baseline: this launches the kernel without the host-side checks
    // performed by `add`. Keep it commented out to compare the two call sites.
    // let len_i32 = len as i32;
    // let grid_x = (len / TILE_SIZE) as u32;
    // unsafe {
    //     add_ptr(
    //         z.device_pointer(),
    //         x.device_pointer(),
    //         y.device_pointer(),
    //         len_i32,
    //     )
    // }
    // .grid((grid_x, 1, 1))
    // .await?;

    add(&mut z, &x, &y).await?;

    let x_host = x.to_host_vec().await?;
    let y_host = y.to_host_vec().await?;
    let z_host = z.to_host_vec().await?;
    for i in 0..z_host.len() {
        let x_i: f32 = x_host[i];
        let y_i: f32 = y_host[i];
        let answer = x_i + y_i;
        println!("{} + {} = {}", x_i, y_i, z_host[i]);
        assert_eq!(answer, z_host[i], "{} != {} ?", answer, z_host[i]);
    }
    Ok(())
}
