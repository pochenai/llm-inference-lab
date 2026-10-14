//! Compare direct powers, Horner, explicit FMA over a vector of x values.
//!
//! Run from `rust/` with `cargo run -p cutile-tutorials --example polynomial_eval`.

use cuda_async::prelude::DeviceOp;
use cuda_core::Device;
use cutile::api::{copy_host_vec_to_device, linspace, zeros};
use cutile::bench::{do_bench, BenchOptions};
use cutile::error::Error;
use cutile::tensor::{IntoPartition, Reshape, Tensor, ToHostVec};
use cutile::tile_kernel::TileKernel;
use std::sync::Arc;

const N: usize = 1 << 20;
const TILE: usize = 16;
const COEFFICIENT_COUNT: usize = 16;

#[cutile::module]
mod polynomial_module {
    use cutile::core::*;

    // Direct-power form: P(x) = sum_{i=0}^{COEFFICIENT_COUNT-1} c_i * x^i.
    // `power` starts at x^0 and is multiplied by x once per loop iteration.
    #[cutile::entry()]
    fn naive<const B: i32, const COEFFICIENT_COUNT: i32>(
        z: &mut Tensor<f32, { [B, 1] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        coeffs: &Tensor<f32, { [-1, -1] }>,
    ) {
        let x = x.load_like(z);
        let coefficient_tile = coeffs
            .partition(shape![COEFFICIENT_COUNT, 1])
            .load([0i32, 0i32]);
        let coefficient_column = scalar_to_tile(0i32);
        let mut power = constant(1.0f32, z.shape());
        let mut result = constant(0.0f32, z.shape());

        for i in 0..COEFFICIENT_COUNT {
            let coefficient_cell: Tile<f32, { [1, 1] }> =
                extract(coefficient_tile, [scalar_to_tile(i), coefficient_column]);
            let coefficient = coefficient_cell.broadcast(z.shape());
            result = result + coefficient * power;
            power = power * x;
        }
        z.store(result);
    }

    // Horner form: P(x) = c_0 + x * (c_1 + x * (... + x * c_{n-1})).
    // The loop starts with c_{n-1} and folds coefficients down to c_0.
    #[cutile::entry()]
    fn horner<const B: i32, const COEFFICIENT_COUNT: i32>(
        z: &mut Tensor<f32, { [B, 1] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        coeffs: &Tensor<f32, { [-1, -1] }>,
    ) {
        let x = x.load_like(z);
        let coefficient_tile = coeffs
            .partition(shape![COEFFICIENT_COUNT, 1])
            .load([0i32, 0i32]);
        let coefficient_column = scalar_to_tile(0i32);
        let result_cell: Tile<f32, { [1, 1] }> = extract(
            coefficient_tile,
            [scalar_to_tile(COEFFICIENT_COUNT - 1), coefficient_column],
        );
        let mut result = result_cell.broadcast(z.shape());

        for i in 1..COEFFICIENT_COUNT {
            let coefficient_cell: Tile<f32, { [1, 1] }> = extract(
                coefficient_tile,
                [
                    scalar_to_tile(COEFFICIENT_COUNT - 1 - i),
                    coefficient_column,
                ],
            );
            let coefficient = coefficient_cell.broadcast(z.shape());
            result = result * x + coefficient;
        }
        z.store(result);
    }

    // FMA Horner form: a <- fma(x, a, c_i), for i = n-2 down to 0,
    // starting with a = c_{n-1}; thus each step computes a*x + c_i with one rounding.
    #[cutile::entry()]
    fn horner_fma<const B: i32, const COEFFICIENT_COUNT: i32>(
        z: &mut Tensor<f32, { [B, 1] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        coeffs: &Tensor<f32, { [-1, -1] }>,
    ) {
        let x = x.load_like(z);
        let coefficient_tile = coeffs
            .partition(shape![COEFFICIENT_COUNT, 1])
            .load([0i32, 0i32]);
        let coefficient_column = scalar_to_tile(0i32);
        let result_cell: Tile<f32, { [1, 1] }> = extract(
            coefficient_tile,
            [scalar_to_tile(COEFFICIENT_COUNT - 1), coefficient_column],
        );
        let mut result = result_cell.broadcast(z.shape());

        for i in 1..COEFFICIENT_COUNT {
            let coefficient_cell: Tile<f32, { [1, 1] }> = extract(
                coefficient_tile,
                [
                    scalar_to_tile(COEFFICIENT_COUNT - 1 - i),
                    coefficient_column,
                ],
            );
            let coefficient = coefficient_cell.broadcast(z.shape());
            result = fma(x, result, coefficient, rounding::NearestEven, ftz::Disabled);
        }
        z.store(result);
    }
}

use polynomial_module::{horner, horner_fma, naive};

macro_rules! benchmark_kernel {
    ($kernel:ident, $name:literal, $stream:ident, $input_tensor:ident, $coeff_tensor:ident) => {{
        let mut z = Some(zeros(&[N, 1]).sync_on(&$stream)?.partition([TILE, 1]));
        let mut input = Some($input_tensor.dup().sync_on(&$stream)?);
        let mut coefficient_input = Some($coeff_tensor.dup().sync_on(&$stream)?);
        let measurement = do_bench(&$stream, &BenchOptions::default(), |bench_stream| {
            let (next_z, next_x, next_coeffs) = $kernel(
                z.take().expect("benchmark output is available"),
                input.take().expect("benchmark input is available"),
                coefficient_input
                    .take()
                    .expect("benchmark coefficients are available"),
            )
            .generics(vec![
                TILE.to_string(),
                (COEFFICIENT_COUNT as i32).to_string(),
            ])
            .sync_on(bench_stream)?;
            z = Some(next_z);
            input = Some(next_x);
            coefficient_input = Some(next_coeffs);
            Ok(())
        })?;
        let result: Vec<f32> = z
            .expect("benchmark returned the output partition")
            .unpartition()
            .to_host_vec()
            .sync_on(&$stream)?;
        println!(
            "{}: y[0]={:.6}, y[N/2]={:.6}, y[N-1]={:.6}",
            $name,
            result[0],
            result[N / 2],
            result[N - 1]
        );
        println!(
            "{} CUDA-event bench: median {:.3} ms, mean {:.3} ms, {} reps",
            $name,
            measurement.median_ms(),
            measurement.mean_ms(),
            measurement.reps()
        );
    }};
}

fn main() -> Result<(), Error> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let x_1d: Tensor<f32> = linspace(-1.0, 1.0, N).sync_on(&stream)?.into();
    let x = x_1d.reshape(&[N, 1])?;

    // Generate reproducible pseudo-random coefficients on the CPU.
    assert!(
        COEFFICIENT_COUNT > 0,
        "polynomial needs at least one coefficient"
    );
    let host_coefficients: Arc<Vec<f32>> =
        Arc::new(make_coefficients(COEFFICIENT_COUNT, 0x5eed_cafe));
    let coefficients_1d: Tensor<f32> =
        copy_host_vec_to_device(&host_coefficients).sync_on(&stream)?;
    let coeffs = coefficients_1d.reshape(&[COEFFICIENT_COUNT, 1])?;

    benchmark_kernel!(naive, "naive", stream, x, coeffs);
    benchmark_kernel!(horner, "horner", stream, x, coeffs);
    benchmark_kernel!(horner_fma, "horner_fma", stream, x, coeffs);
    println!("Degree: {}; all times are CUDA-event kernel timings. Input creation and host copies are outside the measurement.", COEFFICIENT_COUNT - 1);
    Ok(())
}

fn make_coefficients(count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / 16_777_215.0;
            (2.0 * unit - 1.0) * 0.02
        })
        .collect()
}
