//! Compare direct powers, Horner, explicit FMA over a vector of x values.
//!
//! Run from `rust/` with `cargo run --release -p cutile-tutorials --example polynomial_eval`.
//! Historical coefficient-access comparisons are recorded in ../experiments/polynomial/RESULTS.md.

mod support;

use cuda_async::prelude::DeviceOp;
use cuda_core::Device;
use cutile::api::{copy_host_vec_to_device, linspace, zeros};
use cutile::bench::{do_bench_paired, BenchOptions};
use cutile::error::Error;
use cutile::tensor::{IntoPartition, Reshape, Tensor, ToHostVec};
use cutile::tile_kernel::TileKernel;
use std::sync::Arc;
use support::{Args, TimedGraph};

const DEFAULT_TILE: usize = 1024;

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
        let coefficients = coeffs.partition(shape![1, 1]);
        let mut power = constant(1.0f32, z.shape());
        let mut result = constant(0.0f32, z.shape());
        for i in 0..COEFFICIENT_COUNT {
            let c = coefficients.load([i, 0i32]).broadcast(z.shape());
            result = result + c * power;
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
        let coefficients = coeffs.partition(shape![1, 1]);
        let mut result = coefficients
            .load([COEFFICIENT_COUNT - 1, 0i32])
            .broadcast(z.shape());
        if COEFFICIENT_COUNT > 1 {
            for i in 1..COEFFICIENT_COUNT {
                let coefficient = coefficients
                    .load([COEFFICIENT_COUNT - 1 - i, 0i32])
                    .broadcast(z.shape());
                result = result * x + coefficient;
            }
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
        let coefficients = coeffs.partition(shape![1, 1]);
        let mut result = coefficients
            .load([COEFFICIENT_COUNT - 1, 0i32])
            .broadcast(z.shape());
        if COEFFICIENT_COUNT > 1 {
            for i in 1..COEFFICIENT_COUNT {
                let coefficient = coefficients
                    .load([COEFFICIENT_COUNT - 1 - i, 0i32])
                    .broadcast(z.shape());
                result = fma(x, result, coefficient, rounding::NearestEven, ftz::Disabled);
            }
        }
        z.store(result);
    }
}

use polynomial_module::{horner, horner_fma, naive};

fn main() -> Result<(), Error> {
    let args = Args::parse_with_tile(DEFAULT_TILE);
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let x: Tensor<f32> = linspace(-1.0, 1.0, args.n).sync_on(&stream)?;
    let x = x.reshape(&[args.n, 1])?;
    let host_coefficients = Arc::new(make_coefficients(args.coefficients, 0x5eed_cafe));
    let coeffs = copy_host_vec_to_device(&host_coefficients)
        .sync_on(&stream)?
        .reshape(&[args.coefficients, 1])?;
    let generics = vec![args.tile.to_string(), args.coefficients.to_string()];
    let options = args.options();
    let output = || -> Result<_, Error> {
        let z: Tensor<f32> = zeros(&[args.n, 1]).sync_on(&stream)?;
        Ok(Some(z.partition([args.tile, 1])))
    };
    let mut zn = output()?;
    let mut zh = output()?;
    let mut zf = output()?;
    macro_rules! capture {
        ($kernel:ident, $z:ident) => {
            TimedGraph::capture(&stream, args.batch, |scope| {
                let (next, _, _) = scope.record(
                    $kernel($z.take().unwrap(), &x, &coeffs)
                        .generics(generics.clone())
                        .compile_options(options.clone()),
                )?;
                $z = Some(next);
                Ok(())
            })?
        };
    }
    let gn = capture!(naive, zn);
    let gh = capture!(horner, zh);
    let gf = capture!(horner_fma, zf);
    if args.profile {
        gn.sample_us(&stream)?;
    } else {
        let (sn, sh) = support::paired_samples(&stream, &gn, &gh, args.reps, args.cold)?;
        println!(
            "naive vs Horner GPU graph interval: {:.3} vs {:.3} us, speedup {:.3}x",
            sn.median(),
            sh.median(),
            sn.median() / sh.median()
        );
    }
    let (sh, sf) = support::paired_samples(&stream, &gh, &gf, args.reps, args.cold)?;
    support::print_pair("horner", &args, &sh, &sf);
    if args.eager {
        // The outer event interval can include GPU idle time while Rust prepares a launch.
        let bench_options = BenchOptions {
            clear_l2: args.cold,
            min_reps: args.reps,
            max_reps: args.reps,
            ..BenchOptions::default()
        };
        let (a, b) = do_bench_paired(
            &stream,
            &bench_options,
            |s| {
                let (next, _, _) = horner(zh.take().unwrap(), &x, &coeffs)
                    .generics(generics.clone())
                    .compile_options(options.clone())
                    .sync_on(s)?;
                zh = Some(next);
                Ok(())
            },
            |s| {
                let (next, _, _) = horner_fma(zf.take().unwrap(), &x, &coeffs)
                    .generics(generics.clone())
                    .compile_options(options.clone())
                    .sync_on(s)?;
                zf = Some(next);
                Ok(())
            },
        )?;
        println!(
            "EAGER,event_interval_horner_us={:.3},event_interval_fma_us={:.3},speedup={:.3}",
            a.median_ms() * 1e3,
            b.median_ms() * 1e3,
            a.median_ms() / b.median_ms()
        );
    }
    let xv: Vec<f32> = x.dup().sync_on(&stream)?.to_host_vec().sync_on(&stream)?;
    for (name, z) in [("naive", zn), ("horner", zh), ("horner_fma", zf)] {
        let values: Vec<f32> = z.unwrap().unpartition().to_host_vec().sync_on(&stream)?;
        let mut max_error = 0.0f64;
        for (&v, &x) in values.iter().zip(&xv) {
            let reference = host_coefficients
                .iter()
                .rev()
                .fold(0.0f64, |p, &c| p * x as f64 + c as f64);
            max_error = max_error.max((v as f64 - reference).abs());
            if !v.is_finite() || (v as f64 - reference).abs() > 2e-5 + reference.abs() * 2e-5 {
                return Err(cutile::error::tensor_error(
                    "polynomial disagrees with CPU f64 reference",
                ));
            }
        }
        println!(
            "{name}: y[0]={:.6}, y[N/2]={:.6}, y[N-1]={:.6}, max_abs_error={max_error:.9e}",
            values[0],
            values[args.n / 2],
            values[args.n - 1]
        );
    }
    if let Some(dir) = &args.inspect_dir {
        for kernel in ["naive", "horner", "horner_fma"] {
            support::inspect(
                polynomial_module::__module_ast_self,
                "polynomial_module",
                kernel,
                generics.clone(),
                &[("z", &[1, 1]), ("x", &[1, 1]), ("coeffs", &[1, 1])],
                options.clone(),
                dir,
                &format!("final_b{}_c{}", args.tile, args.coefficients),
            )?;
        }
    }
    println!("Degree: {}; tile: {}; cache: {}; graph batch: {}. Coefficient preparation and CPU launch time are outside the graph-internal event interval.", args.coefficients-1,args.tile,if args.cold {"cold"}else{"warm"},args.batch);
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
