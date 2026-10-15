//! Compare separate multiply/add instructions with explicit FMA, and determine when
//! fewer arithmetic instructions improve kernel time. This is a synthetic control
//! experiment for polynomial evaluation; its speedup is not a prediction for Horner.
//!
//! Inputs and paired comparisons
//! -----------------------------
//! Both implementations receive the same f32 vector x (default N=1,048,576, values
//! in [-0.9, 0.9]), runtime alpha=0.99931, and runtime bias=0.000127.
//! --chains 1 compares mul_add with fused; --chains 4 compares mul_add_ilp with
//! fused_ilp. Within each pair, only separate multiply/add versus explicit FMA
//! changes. Separate operations round twice; FMA rounds once, so outputs can differ.
//! Saved SASS from the recorded runs confirms FMUL/FADD versus FFMA instructions.
//!
//! Experimental design
//! -------------------
//! 1. Single vector transform: --iterations 1 --chains 1
//!    Each element computes y[i] = alpha*x[i] + bias. One f32 input load and one
//!    f32 output store accompany only one multiply/add pair. This measures whether
//!    FMA helps when memory traffic dominates; the observed benefit is negligible.
//!
//! 2. One dependent recurrence: --iterations 512 --chains 1
//!    Start with r=x[i], repeat r=alpha*r+bias 512 times, then store r.
//!    Intermediate results stay in registers. More work per input load increases
//!    arithmetic intensity, but each step depends on the previous result.
//!
//! 3. Four independent recurrences: --iterations 512 --chains 4
//!    Start with r[k]=x[i]+0.1*k for k=0..3. Repeat r[k]=alpha*r[k]+bias
//!    512 times for each chain, then store their sum. This performs 2048 multiply/add
//!    pairs per element with the same main input/output traffic as case 1.
//!    Four independent chains expose instruction-level parallelism (ILP), allowing
//!    the arithmetic pipelines to stay busy while each chain waits for its result.
//!    This configuration exposes the throughput benefit of FMA most clearly.
//!
//! --iterations is the recurrence length INSIDE each kernel, not the number of
//! kernel launches. --chains is the number of independent accumulators per input
//! element, not the number of GPU threads. Both configurations process a vector
//! in parallel. Four chains do four times the recurrence work of one chain, so
//! compare separate versus fused within each configuration, not raw times across it.
//!
//! Recorded results (microseconds; N=1,048,576, tile=256, warm cache, 100 paired samples):
//!   iterations  chains  separate  FMA     speedup
//!   1           1         5.56      5.48  1.015x
//!   512         1        92.52     68.32  1.354x
//!   512         4       359.94    183.48  1.962x
//! The last case reached 1.988x with cold cache. These are measured results from
//! the recorded setup, not universal ratios. One chain can still benefit from FMA;
//! the near-2x result requires enough arithmetic work and independent instructions.
//! See ../experiments/polynomial/RESULTS.md for hardware, profiling, and more cases.
//!
//! Run from rust/:
//!   cargo run --release -p cutile-tutorials --example fma_vs_mul_add -- --iterations 1 --chains 1
//!   cargo run --release -p cutile-tutorials --example fma_vs_mul_add -- --iterations 512 --chains 1
//!   cargo run --release -p cutile-tutorials --example fma_vs_mul_add -- --iterations 512 --chains 4
//! Timing uses CUDA events inside graphs and paired A/B sampling, excluding input
//! creation and host copies. --reps controls sample count; --batch controls kernel
//! launches per graph (default 16). --cache cold flushes L2 before each sample and
//! sets batch=1. Reported times are per kernel; --tile defaults to 256.

mod support;

use cuda_async::prelude::DeviceOp;
use cuda_core::Device;
use cutile::api::{linspace, zeros};
use cutile::error::{tensor_error, Error};
use cutile::tensor::{IntoPartition, Reshape, Tensor, ToHostVec};
use cutile::tile_kernel::TileKernel;
use support::{Args, TimedGraph};

#[cutile::module]
mod arithmetic {
    use cutile::core::*;

    // r_{j+1}(x) = alpha*r_j(x) + bias, r_0(x)=x; separate rounding for mul and add.
    #[cutile::entry()]
    fn mul_add<const B: i32, const STEPS: i32>(
        z: &mut Tensor<f32, { [B, 1] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        alpha: f32,
        bias: f32,
    ) {
        let mut r = x.load_like(z);
        let a = alpha.broadcast(z.shape());
        let b = bias.broadcast(z.shape());
        for _ in 0..STEPS {
            r = r * a + b;
        }
        z.store(r);
    }

    // r_{j+1}(x) = fma(alpha, r_j(x), bias), r_0(x)=x; one rounding per step.
    #[cutile::entry()]
    fn fused<const B: i32, const STEPS: i32>(
        z: &mut Tensor<f32, { [B, 1] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        alpha: f32,
        bias: f32,
    ) {
        let mut r = x.load_like(z);
        let a = alpha.broadcast(z.shape());
        let b = bias.broadcast(z.shape());
        for _ in 0..STEPS {
            r = fma(r, a, b, rounding::NearestEven, ftz::Disabled);
        }
        z.store(r);
    }

    // r_{k,j+1}=alpha*r_{k,j}+bias, r_{k,0}=x+0.1*k, k=0..3; z=sum_k r_{k,STEPS}.
    #[cutile::entry()]
    fn mul_add_ilp<const B: i32, const STEPS: i32>(
        z: &mut Tensor<f32, { [B, 1] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        alpha: f32,
        bias: f32,
    ) {
        let x = x.load_like(z);
        let mut r0 = x;
        let mut r1 = x + 0.1f32.broadcast(z.shape());
        let mut r2 = x + 0.2f32.broadcast(z.shape());
        let mut r3 = x + 0.3f32.broadcast(z.shape());
        let a = alpha.broadcast(z.shape());
        let b = bias.broadcast(z.shape());
        for _ in 0..STEPS {
            r0 = r0 * a + b;
            r1 = r1 * a + b;
            r2 = r2 * a + b;
            r3 = r3 * a + b;
        }
        z.store((r0 + r1) + (r2 + r3));
    }

    // r_{k,j+1}=fma(alpha,r_{k,j},bias), r_{k,0}=x+0.1*k; z=sum_k r_{k,STEPS}.
    // Independent chains expose instruction-level parallelism.
    #[cutile::entry()]
    fn fused_ilp<const B: i32, const STEPS: i32>(
        z: &mut Tensor<f32, { [B, 1] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        alpha: f32,
        bias: f32,
    ) {
        let x = x.load_like(z);
        let mut r0 = x;
        let mut r1 = x + 0.1f32.broadcast(z.shape());
        let mut r2 = x + 0.2f32.broadcast(z.shape());
        let mut r3 = x + 0.3f32.broadcast(z.shape());
        let a = alpha.broadcast(z.shape());
        let b = bias.broadcast(z.shape());
        for _ in 0..STEPS {
            r0 = fma(r0, a, b, rounding::NearestEven, ftz::Disabled);
            r1 = fma(r1, a, b, rounding::NearestEven, ftz::Disabled);
            r2 = fma(r2, a, b, rounding::NearestEven, ftz::Disabled);
            r3 = fma(r3, a, b, rounding::NearestEven, ftz::Disabled);
        }
        z.store((r0 + r1) + (r2 + r3));
    }
}

fn main() -> Result<(), Error> {
    let args = Args::parse();
    assert!(args.chains == 1 || args.chains == 4);
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let input: Tensor<f32> = linspace(-0.9, 0.9, args.n).sync_on(&stream)?;
    let x = input.reshape(&[args.n, 1])?;
    let za: Tensor<f32> = zeros(&[args.n, 1]).sync_on(&stream)?;
    let zb: Tensor<f32> = zeros(&[args.n, 1]).sync_on(&stream)?;
    let mut za = Some(za.partition([args.tile, 1]));
    let mut zb = Some(zb.partition([args.tile, 1]));
    let alpha = 0.99931f32;
    let bias = 0.000127f32;
    let generics = vec![args.tile.to_string(), args.iterations.to_string()];
    let options = args.options();

    macro_rules! compare {
        ($a:ident, $b:ident) => {{
            let ga = TimedGraph::capture(&stream, args.batch, |scope| {
                let (next_z, _, _, _) = scope.record(arithmetic::$a(za.take().unwrap(), &x, alpha, bias)
                    .generics(generics.clone()).compile_options(options.clone()))?;
                za = Some(next_z);
                Ok(())
            })?;
            let gb = TimedGraph::capture(&stream, args.batch, |scope| {
                let (next_z, _, _, _) = scope.record(arithmetic::$b(zb.take().unwrap(), &x, alpha, bias)
                    .generics(generics.clone()).compile_options(options.clone()))?;
                zb = Some(next_z);
                Ok(())
            })?;
            let (sa, sb) = support::paired_samples(&stream, &ga, &gb, args.reps, args.cold)?;
            support::print_pair("arithmetic", &args, &sa, &sb);
            if let Some(dir) = &args.inspect_dir {
                for kernel in [stringify!($a), stringify!($b)] {
                    support::inspect(arithmetic::__module_ast_self, "arithmetic", kernel,
                        generics.clone(), &[("z", &[1, 1]), ("x", &[1, 1])],
                        options.clone(), dir, &format!("b{}_s{}_c{}_w{:?}", args.tile, args.iterations, args.chains, args.warps))?;
                }
            }
        }};
    }
    if args.chains == 1 {
        compare!(mul_add, fused);
    } else {
        compare!(mul_add_ilp, fused_ilp);
    }
    let xv: Vec<f32> = x.to_host_vec().sync_on(&stream)?;
    let av: Vec<f32> = za.unwrap().unpartition().to_host_vec().sync_on(&stream)?;
    let bv: Vec<f32> = zb.unwrap().unpartition().to_host_vec().sync_on(&stream)?;
    let mut error_a = 0.0f64;
    let mut error_b = 0.0f64;
    for i in (0..args.n).step_by((args.n / 4096).max(1)) {
        let mut reference = 0.0;
        for chain in 0..args.chains {
            let mut v = xv[i] as f64 + (chain as f32 * 0.1) as f64;
            for _ in 0..args.iterations {
                v = v * alpha as f64 + bias as f64;
            }
            reference += v;
        }
        error_a = error_a.max((av[i] as f64 - reference).abs());
        error_b = error_b.max((bv[i] as f64 - reference).abs());
    }
    let difference = av
        .iter()
        .zip(&bv)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("CHECK,max_abs_error_mul_add={error_a:.9e},max_abs_error_fma={error_b:.9e},max_abs_difference={difference:.9e}");
    if !av.iter().chain(&bv).all(|v| v.is_finite()) || error_a > 2e-3 || error_b > 2e-3 {
        return Err(tensor_error(
            "arithmetic output disagrees with the CPU f64 reference",
        ));
    }
    Ok(())
}
