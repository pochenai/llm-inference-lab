//! Shared CLI, GPU timing, statistics, and assembly export for the polynomial and FMA examples.

#![allow(dead_code)]

use cuda_async::cuda_graph::{CudaGraph, Scope};
use cuda_async::error::DeviceError;
use cuda_async::prelude::DeviceOp;
use cuda_core::{Device, DriverError, IntoResult, Stream};
use cutile::api::zeros;
use cutile::error::{tensor_error, Error};
use cutile::tensor::Tensor;
use cutile::tile_kernel::CompileOptions;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

/// Workload and measurement settings shared by the two standalone examples.
pub struct Args {
    /// Number of input elements; must be positive and divisible by `tile`.
    pub n: usize,
    /// Output elements per tile; must be a power of two.
    pub tile: usize,
    /// Polynomial coefficient count; degree is `coefficients - 1`.
    pub coefficients: usize,
    /// Recurrence steps inside each arithmetic kernel, independent of benchmark repetitions.
    pub iterations: usize,
    /// Independent accumulators per input element in the arithmetic example (1 or 4).
    pub chains: usize,
    /// Optional compiler hint for worker warps per CTA; emitted launch metadata is authoritative.
    pub warps: Option<i32>,
    /// Number of timed samples collected for each kernel in a comparison.
    pub reps: usize,
    /// Kernel launches captured in each graph; timing is divided by this count.
    pub batch: usize,
    /// Flush L2 before each timed sample and use one kernel per graph.
    pub cold: bool,
    /// Optional destination directory for IR, cubin, SASS, PTX text, and resource reports.
    pub inspect_dir: Option<String>,
    /// Use one kernel per graph for profiling; the polynomial example also skips naive timing.
    pub profile: bool,
    /// Print the additional outer-event launch timing in the polynomial example.
    pub eager: bool,
}

impl Args {
    /// Parse process arguments with the arithmetic example's default tile size of 256.
    pub fn parse() -> Self {
        Self::parse_with_tile(256)
    }

    /// Parse shared flags with a caller-selected default tile and validate kernel dimensions.
    /// `--help` prints usage and exits; malformed or unsupported arguments panic.
    /// Cold-cache and profiling modes override `--batch` to 1.
    pub fn parse_with_tile(default_tile: usize) -> Self {
        let mut args = Self {
            n: 1 << 20,
            tile: default_tile,
            coefficients: 32,
            iterations: 128,
            chains: 1,
            warps: None,
            reps: 100,
            batch: 16,
            cold: false,
            inspect_dir: None,
            profile: false,
            eager: false,
        };
        let mut cli = std::env::args().skip(1);
        while let Some(flag) = cli.next() {
            if flag == "--profile" {
                args.profile = true;
                continue;
            }
            if flag == "--eager" {
                args.eager = true;
                continue;
            }
            if flag == "--help" {
                println!("Options: --n N --tile B --coefficients C --iterations S --chains 1|4 --warps W --reps R --batch K --cache warm|cold --inspect-dir DIR --profile --eager");
                std::process::exit(0);
            }
            let value = cli.next().expect("each option needs a value");
            match flag.as_str() {
                "--n" => args.n = value.parse().unwrap(),
                "--tile" => args.tile = value.parse().unwrap(),
                "--coefficients" => args.coefficients = value.parse().unwrap(),
                "--iterations" => args.iterations = value.parse().unwrap(),
                "--chains" => args.chains = value.parse().unwrap(),
                "--warps" => args.warps = Some(value.parse().unwrap()),
                "--reps" => args.reps = value.parse().unwrap(),
                "--batch" => args.batch = value.parse().unwrap(),
                "--cache" => {
                    assert!(value == "cold" || value == "warm");
                    args.cold = value == "cold";
                }
                "--inspect-dir" => args.inspect_dir = Some(value),
                _ => panic!("unknown option {flag}"),
            }
        }
        assert!(args.n > 0 && args.tile > 0 && args.n % args.tile == 0);
        assert!(args.tile.is_power_of_two());
        assert!(
            args.n <= i32::MAX as usize
                && args.tile <= i32::MAX as usize
                && args.coefficients <= i32::MAX as usize
                && args.iterations <= i32::MAX as usize
        );
        assert!(args.coefficients > 0 && args.iterations > 0 && args.reps > 0 && args.batch > 0);
        // A cold sample must flush before every individual kernel.
        if args.cold || args.profile {
            args.batch = 1;
        }
        args
    }

    /// Build device compiler options: optimization level 3, debug disabled, and line info enabled.
    /// Apply the worker-warp hint when supplied, so both compared kernels use the same settings.
    pub fn options(&self) -> CompileOptions {
        let options = CompileOptions::new()
            .device_debug(false)
            .opt_level(3)
            .lineinfo(true);
        match self.warps {
            Some(warps) => options.num_worker_warps_per_cta(warps),
            None => options,
        }
    }
}

/// Timing events are graph nodes, so Rust submission time is outside their interval.
pub struct TimedGraph {
    graph: CudaGraph<()>,
    start: TimingEvent,
    end: TimingEvent,
    batch: usize,
}

impl TimedGraph {
    /// Capture `batch` calls to `record` between two actual CUDA event nodes on `stream`.
    /// The closure records one kernel launch per call; compilation and capture precede timing.
    /// Keep both event handles alive with the graph so replay can record valid timestamps.
    pub fn capture<F>(stream: &Arc<Stream>, batch: usize, mut record: F) -> Result<Self, Error>
    where
        F: FnMut(&Scope) -> Result<(), DeviceError>,
    {
        let start = TimingEvent::new(stream.device())?;
        let end = TimingEvent::new(stream.device())?;
        let graph = CudaGraph::scope(stream, |scope| {
            start.record_external(stream)?;
            for _ in 0..batch {
                record(scope)?;
            }
            end.record_external(stream)?;
            Ok(())
        })?;
        Ok(Self {
            graph,
            start,
            end,
            batch,
        })
    }

    /// Replay the graph, wait for completion, and return its event interval per kernel in us.
    /// The interval includes GPU scheduling between nodes; CPU submission and waiting are outside it.
    pub fn sample_us(&self, stream: &Arc<Stream>) -> Result<f64, Error> {
        self.graph.launch().sync_on(stream)?;
        Ok(self.start.elapsed_time(&self.end)? as f64 * 1e3 / self.batch as f64)
    }
}

/// Normal event record during capture can become only a dependency edge.
/// EXTERNAL forces real event nodes, which are required for elapsed-time queries.
struct TimingEvent {
    handle: cuda_core::sys::CUevent,
    device: Arc<Device>,
}
impl TimingEvent {
    /// Bind the CUDA device and create an owned event with timestamp recording enabled.
    fn new(device: &Arc<Device>) -> Result<Self, DriverError> {
        device.bind_to_thread()?;
        let mut handle = std::ptr::null_mut();
        unsafe {
            cuda_core::sys::cuEventCreate(&mut handle, 0).result()?;
        }
        Ok(Self {
            handle,
            device: device.clone(),
        })
    }
    /// During stream capture, insert an event-record node using `CU_EVENT_RECORD_EXTERNAL`.
    /// Without this flag, capture can turn recording into a dependency edge without a timestamp.
    fn record_external(&self, stream: &Arc<Stream>) -> Result<(), DriverError> {
        // Both handles are owned and valid; this is called only during stream capture.
        unsafe {
            cuda_core::sys::cuEventRecordWithFlags(
                self.handle,
                stream.cu_stream(),
                cuda_core::sys::CUevent_record_flags_enum_CU_EVENT_RECORD_EXTERNAL,
            )
            .result()
        }
    }
    /// Return milliseconds between this start event and `end` on the same device.
    /// Both event records must have completed before this query.
    fn elapsed_time(&self, end: &Self) -> Result<f32, DriverError> {
        let mut ms = 0.0;
        // The graph launch has synchronized, so both event nodes have completed.
        unsafe {
            cuda_core::sys::cuEventElapsedTime_v2(&mut ms, self.handle, end.handle).result()?;
        }
        Ok(ms)
    }
}
impl Drop for TimingEvent {
    /// Attempt to restore the owning device and release the event without panicking during drop.
    fn drop(&mut self) {
        let _ = self.device.bind_to_thread();
        unsafe {
            let _ = cuda_core::sys::cuEventDestroy_v2(self.handle);
        }
    }
}

/// Per-kernel GPU durations in microseconds, with one value for each timed graph replay.
pub struct Samples(pub Vec<f64>);
impl Samples {
    /// Sort a copy and select the nearest sample at quantile `q`, without interpolation.
    /// Requires nonempty samples and `q` in [0, 1]; the original collection order is preserved.
    pub fn quantile(&self, q: f64) -> f64 {
        let mut values = self.0.clone();
        values.sort_by(f64::total_cmp);
        values[((values.len() - 1) as f64 * q).round() as usize]
    }
    /// Return the 0.5 quantile using the same sample-selection rule as the percentile output.
    pub fn median(&self) -> f64 {
        self.quantile(0.5)
    }
}

/// Warm up each graph ten times, then collect `reps` samples per graph in A/B, B/A order.
/// Alternating order reduces bias from changing GPU clocks; returned samples remain grouped by graph.
/// In cold mode, write a flush buffer before every timed replay, outside the timing events.
/// The buffer size is twice L2, clamped to 64-512 MiB. Warm mode performs no cache flush.
pub fn paired_samples(
    stream: &Arc<Stream>,
    a: &TimedGraph,
    b: &TimedGraph,
    reps: usize,
    cold: bool,
) -> Result<(Samples, Samples), Error> {
    let flush_bytes = (stream.device().l2_cache_size_bytes()? * 2).clamp(64 << 20, 512 << 20);
    let flush_buffer: Option<Tensor<u8>> = if cold {
        Some(zeros(&[flush_bytes]).sync_on(stream)?)
    } else {
        None
    };
    let flush = || -> Result<(), Error> {
        if let Some(buffer) = &flush_buffer {
            // The allocation covers flush_bytes and remains alive until the graph synchronizes.
            unsafe {
                cuda_core::memset_d8_async(
                    buffer.device_pointer().cu_deviceptr(),
                    0,
                    flush_bytes,
                    stream,
                )?;
            }
        }
        Ok(())
    };
    for _ in 0..10 {
        a.sample_us(stream)?;
        b.sample_us(stream)?;
    }
    let mut sa = Vec::with_capacity(reps);
    let mut sb = Vec::with_capacity(reps);
    for i in 0..reps {
        // Alternate pair order as well as the methods to reduce clock drift.
        if i % 2 == 0 {
            flush()?;
            sa.push(a.sample_us(stream)?);
            flush()?;
            sb.push(b.sample_us(stream)?);
        } else {
            flush()?;
            sb.push(b.sample_us(stream)?);
            flush()?;
            sa.push(a.sample_us(stream)?);
        }
    }
    Ok((Samples(sa), Samples(sb)))
}

/// Print workload settings, median times in us, p10/p90, and speedup as median(A)/median(B).
/// A is separate multiply/add and B is explicit FMA; `label` selects arithmetic or polynomial fields.
pub fn print_pair(label: &str, args: &Args, a: &Samples, b: &Samples) {
    let workload = if label == "arithmetic" {
        format!("iterations={},chains={}", args.iterations, args.chains)
    } else {
        format!("count={}", args.coefficients)
    };
    println!("RESULT,{label},n={},tile={},{workload},warps={},cache={},batch={},mul_add_us={:.6},fma_us={:.6},speedup={:.4},a_p10={:.6},a_p90={:.6},b_p10={:.6},b_p90={:.6},reps={}",
        args.n, args.tile,
        args.warps.map_or("default".into(), |v| v.to_string()),
        if args.cold { "cold" } else { "warm" }, args.batch,
        a.median(), b.median(), a.median()/b.median(),
        a.quantile(0.1), a.quantile(0.9), b.quantile(0.1), b.quantile(0.9), a.0.len());
}

/// Compile the selected kernel for sm_120 and save IR plus the optimized cubin under `dir`.
/// `module_ast`, `module_name`, and `kernel` identify the entry; `generics`, `strides`, and
/// `options` must match the measured specialization. File names start with `{tag}_{kernel}`.
/// Use `tileiras`, `nvdisasm`, `cuobjdump`, and `readelf` to export SASS, resource usage,
/// and the cubin's embedded debug PTX text; that PTX text is not optimized SASS.
/// Also print static opcode counts for the selected entry to inspect instruction fusion.
pub fn inspect<F>(
    module_ast: F,
    module_name: &str,
    kernel: &str,
    generics: Vec<String>,
    strides: &[(&str, &[i32])],
    options: CompileOptions,
    dir: &str,
    tag: &str,
) -> Result<(), Error>
where
    F: Fn() -> cutile::cutile_compiler::ast::Module,
{
    use cutile::compile_api::KernelCompiler;
    use cutile::cutile_compiler::cuda_tile_runtime_utils::{run_tileiras, TileirasOptions};
    let artifacts = KernelCompiler::new(module_ast, module_name, kernel)
        .target("sm_120")
        .generics(generics)
        .strides(strides)
        .options(options.clone())
        .compile()?;
    let io_error = |e: std::io::Error| tensor_error(&e.to_string());
    std::fs::create_dir_all(dir).map_err(io_error)?;
    let base = Path::new(dir).join(format!("{tag}_{kernel}"));
    std::fs::write(base.with_extension("mlir"), artifacts.ir_text()).map_err(io_error)?;
    let cubin = run_tileiras(
        &artifacts.bytecode()?,
        "sm_120",
        &TileirasOptions::from_compile_options(&options),
    )?;
    std::fs::write(base.with_extension("cubin"), cubin).map_err(io_error)?;
    let output = std::process::Command::new("nvdisasm")
        .arg("-g")
        .arg(base.with_extension("cubin"))
        .output()
        .map_err(io_error)?;
    if !output.status.success() {
        return Err(tensor_error(&String::from_utf8_lossy(&output.stderr)));
    }
    let sass = String::from_utf8_lossy(&output.stdout);
    std::fs::write(base.with_extension("sass"), sass.as_bytes()).map_err(io_error)?;
    let resources = std::process::Command::new("cuobjdump")
        .arg("--dump-resource-usage")
        .arg(base.with_extension("cubin"))
        .output()
        .map_err(io_error)?;
    std::fs::write(base.with_extension("resources.txt"), &resources.stdout).map_err(io_error)?;
    let ptx = std::process::Command::new("readelf")
        .args(["-p", ".nv_debug_ptx_txt"])
        .arg(base.with_extension("cubin"))
        .output()
        .map_err(io_error)?;
    std::fs::write(base.with_extension("ptx.txt"), &ptx.stdout).map_err(io_error)?;
    let mut ops: BTreeMap<String, usize> = BTreeMap::new();
    let mut in_entry = false;
    for line in sass.lines() {
        if line.trim_start().starts_with(".section\t") {
            in_entry = line.contains(&format!(".text.{kernel}_entry,"));
        }
        if !in_entry {
            continue;
        }
        if let Some((_, instruction)) = line.split_once("*/") {
            let mut words = instruction.split_whitespace();
            let Some(mut opcode) = words.next() else {
                continue;
            };
            if opcode.starts_with('@') {
                opcode = words.next().unwrap_or("");
            }
            if opcode
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_uppercase())
            {
                *ops.entry(
                    opcode
                        .trim_end_matches(';')
                        .split('.')
                        .next()
                        .unwrap()
                        .into(),
                )
                .or_default() += 1;
            }
        }
    }
    println!("SASS,{tag},{kernel},{ops:?}");
    Ok(())
}
