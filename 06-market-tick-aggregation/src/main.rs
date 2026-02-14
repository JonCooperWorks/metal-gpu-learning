// =============================================================================
// LESSON 6: Market Tick Aggregation -- Real Multi-Stage GPU Pipeline
// =============================================================================
//
// This lesson solves a real analytics task from market-data systems:
// aggregate huge batches of (price, size) ticks into summary metrics.
//
// Why GPU can help:
// - Per-tick math is regular (SIMD/SIMT friendly)
// - Reductions are associative (parallel tree reduction)
// - For large N, arithmetic can dominate dispatch overhead
//
// Why CPU can still win:
// - Small N
// - Irregular control flow
// - Frequent sync points
// - Low compute per byte moved
//
// We include both GPU and CPU paths so this lesson demonstrates the crossover,
// not just a one-sided claim.
// =============================================================================

use metal::*;
use objc::rc::autoreleasepool;
use std::env;
use std::ffi::CString;
use std::ptr;
use std::time::Instant;

const DEFAULT_TOTAL_TICKS: u64 = 5_000_000_000;
const DEFAULT_PROGRESS_INTERVAL: u64 = 10;
const DEFAULT_MEMORY_FRACTION: f64 = 0.85;
const DEFAULT_FALLBACK_CHUNK_TICKS: u64 = 4_000_000;
const THREADGROUP_SIZE: u64 = 256;

const SHADER_SOURCE: &str = include_str!("lesson6.metal");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    // One slot: easiest mental model, but no overlap.
    Single,
    // Two slots: allows CPU fill and GPU execution to overlap.
    Double,
}

impl Mode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Double => "double",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ValidationMode {
    // Validate every chunk against CPU reference.
    Full,
    // Validate only chunk 0 + periodic/final chunks.
    Spot,
    // Skip validation for maximum throughput measurement.
    Off,
}

impl ValidationMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Spot => "spot",
            Self::Off => "off",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CompareCpu {
    // Run CPU baseline and print speedup ratio.
    On,
    // Skip CPU baseline to keep benchmark focused on GPU only.
    Off,
}

impl CompareCpu {
    fn as_str(self) -> &'static str {
        match self {
            Self::On => "on",
            Self::Off => "off",
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct KernelParams {
    // Active element count for this chunk.
    count: u32,
    // Explicit padding so Rust and MSL layout stay aligned.
    _pad: u32,
    // Global index of chunk start; used for deterministic formulas.
    base_index: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct TickDerived {
    size: f32,
    notional: f32,
    return_proxy: f32,
    return_proxy_sq: f32,
    price: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct PartialAggregate {
    sum_size: f32,
    sum_notional: f32,
    sum_return_proxy: f32,
    sum_return_proxy_sq: f32,
    min_price: f32,
    max_price: f32,
    tick_count: u32,
}

#[derive(Clone, Copy, Debug)]
struct CliConfig {
    // These fields represent user intent before runtime auto-derivations.
    mode: Mode,
    total_ticks: u64,
    chunk_ticks_override: Option<u64>,
    memory_fraction: f64,
    progress_interval: u64,
    validation: ValidationMode,
    compare_cpu: CompareCpu,
}

#[derive(Clone, Copy, Debug)]
struct Config {
    // Fully resolved runtime config (post auto-sizing).
    mode: Mode,
    total_ticks: u64,
    chunk_ticks: u64,
    memory_fraction: f64,
    progress_interval: u64,
    validation: ValidationMode,
    compare_cpu: CompareCpu,
    chunk_auto: bool,
    detected_ram_bytes: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct AggregateTotals {
    // Accumulator precision is f64 on CPU side to minimize drift when folding
    // many partial f32 values produced by GPU.
    sum_size: f64,
    sum_notional: f64,
    sum_return_proxy: f64,
    sum_return_proxy_sq: f64,
    min_price: f64,
    max_price: f64,
    tick_count: u64,
}

#[derive(Clone, Copy, Debug)]
struct FinalMetrics {
    // Final report fields shown to users.
    total_volume: f64,
    total_notional: f64,
    vwap: f64,
    mean_return_proxy: f64,
    volatility_proxy: f64,
    min_price: f64,
    max_price: f64,
    tick_count: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct RunStats {
    // Timing split helps explain where bottlenecks really are.
    wall_s: f64,
    fill_s: f64,
    gpu_s: f64,
    ticks_processed: u64,
}

struct InFlight {
    // Command buffer ownership is kept until slot reuse/drain point.
    command_buffer: CommandBuffer,
    chunk_index: u64,
    base_index: u64,
    len: usize,
    slot: usize,
    partial_groups: usize,
}

struct GpuResources {
    // These are created once and reused for all chunks.
    command_queue: CommandQueue,
    transform_pipeline: ComputePipelineState,
    reduce_pipeline: ComputePipelineState,
}

fn print_usage() {
    println!(
        "Usage: market-tick-aggregation [--mode single|double] [--total-ticks N] [--chunk-ticks N] [--memory-fraction F] [--progress-interval PCT] [--validate full|spot|off] [--compare-cpu on|off]"
    );
    println!();
    println!("Defaults:");
    println!("  --mode double");
    println!("  --total-ticks {}", DEFAULT_TOTAL_TICKS);
    println!("  --chunk-ticks auto (derived from RAM and in-flight slot count)");
    println!("  --memory-fraction {:.2}", DEFAULT_MEMORY_FRACTION);
    println!("  --progress-interval {}", DEFAULT_PROGRESS_INTERVAL);
    println!("  --validate spot");
    println!("  --compare-cpu on");
}

fn parse_u64(name: &str, value: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|_| format!("Invalid value for {name}: '{value}' (expected unsigned integer)"))
}

fn parse_f64(name: &str, value: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .map_err(|_| format!("Invalid value for {name}: '{value}' (expected float)"))
}

fn parse_args() -> Result<CliConfig, String> {
    let mut mode = Mode::Double;
    let mut total_ticks = DEFAULT_TOTAL_TICKS;
    let mut chunk_ticks_override = None;
    let mut memory_fraction = DEFAULT_MEMORY_FRACTION;
    let mut progress_interval = DEFAULT_PROGRESS_INTERVAL;
    let mut validation = ValidationMode::Spot;
    let mut compare_cpu = CompareCpu::On;

    let args: Vec<String> = env::args().collect();
    let mut i = 1usize;
    // Manual scanner keeps tutorial dependency-free and transparent.
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            "--mode" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| "Missing value for --mode".to_string())?;
                mode = match value.as_str() {
                    "single" => Mode::Single,
                    "double" => Mode::Double,
                    _ => return Err(format!("Invalid --mode value '{value}'")),
                };
                i += 2;
            }
            "--total-ticks" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| "Missing value for --total-ticks".to_string())?;
                total_ticks = parse_u64("--total-ticks", value)?;
                i += 2;
            }
            "--chunk-ticks" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| "Missing value for --chunk-ticks".to_string())?;
                chunk_ticks_override = Some(parse_u64("--chunk-ticks", value)?);
                i += 2;
            }
            "--memory-fraction" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| "Missing value for --memory-fraction".to_string())?;
                memory_fraction = parse_f64("--memory-fraction", value)?;
                i += 2;
            }
            "--progress-interval" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| "Missing value for --progress-interval".to_string())?;
                progress_interval = parse_u64("--progress-interval", value)?;
                i += 2;
            }
            "--validate" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| "Missing value for --validate".to_string())?;
                validation = match value.as_str() {
                    "full" => ValidationMode::Full,
                    "spot" => ValidationMode::Spot,
                    "off" => ValidationMode::Off,
                    _ => return Err(format!("Invalid --validate value '{value}'")),
                };
                i += 2;
            }
            "--compare-cpu" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| "Missing value for --compare-cpu".to_string())?;
                compare_cpu = match value.as_str() {
                    "on" => CompareCpu::On,
                    "off" => CompareCpu::Off,
                    _ => return Err(format!("Invalid --compare-cpu value '{value}'")),
                };
                i += 2;
            }
            flag => return Err(format!("Unknown argument: {flag}")),
        }
    }

    if chunk_ticks_override == Some(0) {
        return Err("--chunk-ticks must be greater than 0".to_string());
    }
    if !(memory_fraction > 0.0 && memory_fraction <= 1.0) {
        return Err("--memory-fraction must be in (0.0, 1.0]".to_string());
    }
    if progress_interval == 0 || progress_interval > 100 {
        return Err("--progress-interval must be in 1..=100".to_string());
    }

    Ok(CliConfig {
        mode,
        total_ticks,
        chunk_ticks_override,
        memory_fraction,
        progress_interval,
        validation,
        compare_cpu,
    })
}

fn detect_system_ram_bytes() -> Option<u64> {
    let mut ram_bytes: u64 = 0;
    let mut size = std::mem::size_of::<u64>();
    let name = CString::new("hw.memsize").ok()?;

    // Direct sysctl call keeps this lightweight; fallback exists on failure.
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&mut ram_bytes as *mut u64).cast(),
            &mut size,
            ptr::null_mut(),
            0,
        )
    };

    if rc == 0 && size == std::mem::size_of::<u64>() {
        Some(ram_bytes)
    } else {
        None
    }
}

// Deterministic generators are shared between CPU and GPU paths so validation
// compares equivalent inputs, not different datasets.
fn price_for_index(idx: u64) -> f32 {
    90.0 + ((idx.wrapping_mul(17).wrapping_add(13)) % 5000) as f32 * 0.01
}

fn size_for_index(idx: u64) -> f32 {
    1.0 + ((idx.wrapping_mul(29).wrapping_add(7)) % 200) as f32 * 0.05
}

fn baseline_price_for_index(idx: u64) -> f32 {
    95.0 + ((idx.wrapping_mul(31).wrapping_add(3)) % 4000) as f32 * 0.01
}

fn should_spot_check(chunk_index: u64, total_chunks: u64, progress_step_chunks: u64) -> bool {
    // Always include final chunk so completion state is explicit.
    chunk_index + 1 == total_chunks || (chunk_index + 1) % progress_step_chunks == 0
}

fn print_progress(chunk_index: u64, total_chunks: u64, total_ticks: u64, chunk_ticks: u64, start: Instant) {
    let done_chunks = chunk_index + 1;
    let done_ticks = done_chunks.saturating_mul(chunk_ticks).min(total_ticks);
    let pct = if total_ticks == 0 {
        100.0
    } else {
        done_ticks as f64 / total_ticks as f64 * 100.0
    };

    println!(
        "  Chunk {:>6}/{}: {:>7.2}B ticks ({:>5.1}%) - {:>7.2}s",
        done_chunks,
        total_chunks,
        done_ticks as f64 / 1e9,
        pct,
        start.elapsed().as_secs_f64(),
    );
}

fn resolve_chunk_ticks(cli: CliConfig, device: &Device) -> Config {
    let detected_ram_bytes = detect_system_ram_bytes();
    let max_buffer = device.max_buffer_length() as u128;

    let (chunk_ticks, chunk_auto) = if let Some(manual) = cli.chunk_ticks_override {
        (manual, false)
    } else {
        // Rough per-tick memory budget per slot:
        // prices (4) + sizes (4) + derived struct (~20) + safety margin.
        // We reserve 32 bytes/tick/slot to avoid overcommitting RAM.
        let bytes_per_tick_per_slot = 32u128;
        let slot_count = match cli.mode {
            Mode::Single => 1u128,
            Mode::Double => 2u128,
        };

        let target_total = detected_ram_bytes
            .map(|r| (r as f64 * cli.memory_fraction) as u128)
            .unwrap_or(DEFAULT_FALLBACK_CHUNK_TICKS as u128 * bytes_per_tick_per_slot * slot_count);

        // In double mode we split memory target across two active slots.
        let per_slot_budget = (target_total / slot_count).max(bytes_per_tick_per_slot);
        let max_by_ram = per_slot_budget / bytes_per_tick_per_slot;

        // Derived buffer is the largest linear-per-tick buffer; clamp by device max.
        let max_by_device = max_buffer / std::mem::size_of::<TickDerived>() as u128;

        let ticks = max_by_ram
            .min(max_by_device)
            .min(cli.total_ticks as u128)
            .max(1) as u64;

        (ticks, true)
    };

    Config {
        mode: cli.mode,
        total_ticks: cli.total_ticks,
        chunk_ticks,
        memory_fraction: cli.memory_fraction,
        progress_interval: cli.progress_interval,
        validation: cli.validation,
        compare_cpu: cli.compare_cpu,
        chunk_auto,
        detected_ram_bytes,
    }
}

fn aggregate_range(base_index: u64, len: usize) -> AggregateTotals {
    let mut out = AggregateTotals {
        min_price: f64::INFINITY,
        max_price: f64::NEG_INFINITY,
        ..AggregateTotals::default()
    };

    // CPU reference path for correctness checks and baseline timing.
    for i in 0..len {
        let idx = base_index + i as u64;
        let price = price_for_index(idx) as f64;
        let size = size_for_index(idx) as f64;
        let notional = price * size;
        let ret = (price - baseline_price_for_index(idx) as f64).abs();

        out.sum_size += size;
        out.sum_notional += notional;
        out.sum_return_proxy += ret;
        out.sum_return_proxy_sq += ret * ret;
        out.min_price = out.min_price.min(price);
        out.max_price = out.max_price.max(price);
        out.tick_count += 1;
    }

    out
}

fn totals_to_metrics(t: AggregateTotals) -> FinalMetrics {
    let count = t.tick_count.max(1) as f64;
    let mean_return = t.sum_return_proxy / count;
    // Numerically stable enough for tutorial scale; clamp avoids tiny negatives
    // caused by floating-point roundoff.
    let variance = (t.sum_return_proxy_sq / count) - mean_return * mean_return;
    let variance = variance.max(0.0);

    FinalMetrics {
        total_volume: t.sum_size,
        total_notional: t.sum_notional,
        vwap: if t.sum_size > 0.0 {
            t.sum_notional / t.sum_size
        } else {
            0.0
        },
        mean_return_proxy: mean_return,
        volatility_proxy: variance.sqrt(),
        min_price: if t.min_price.is_finite() { t.min_price } else { 0.0 },
        max_price: if t.max_price.is_finite() { t.max_price } else { 0.0 },
        tick_count: t.tick_count,
    }
}

fn fold_partials(partials: &[PartialAggregate], out: &mut AggregateTotals) {
    // CPU fold is cheap compared to full per-tick work and avoids global GPU
    // contention from reducing all the way to one value on-device.
    for p in partials {
        out.sum_size += p.sum_size as f64;
        out.sum_notional += p.sum_notional as f64;
        out.sum_return_proxy += p.sum_return_proxy as f64;
        out.sum_return_proxy_sq += p.sum_return_proxy_sq as f64;

        if p.tick_count > 0 {
            out.min_price = out.min_price.min(p.min_price as f64);
            out.max_price = out.max_price.max(p.max_price as f64);
        }
        out.tick_count += p.tick_count as u64;
    }
}

fn assert_aggregate_close(expected: AggregateTotals, got: AggregateTotals, context: &str) -> Result<(), String> {
    let tol_rel: f64 = 5e-4;
    let tol_abs: f64 = 2e-2;

    // Hybrid absolute+relative tolerance works across small and large magnitudes.
    let check = |name: &str, e: f64, g: f64| -> Result<(), String> {
        let diff = (e - g).abs();
        let allowed = tol_abs.max(e.abs() * tol_rel);
        if diff > allowed {
            return Err(format!(
                "{} mismatch for {}: expected {:.6}, got {:.6}, diff {:.6} > {:.6}",
                context, name, e, g, diff, allowed
            ));
        }
        Ok(())
    };

    check("sum_size", expected.sum_size, got.sum_size)?;
    check("sum_notional", expected.sum_notional, got.sum_notional)?;
    check("sum_return_proxy", expected.sum_return_proxy, got.sum_return_proxy)?;
    check(
        "sum_return_proxy_sq",
        expected.sum_return_proxy_sq,
        got.sum_return_proxy_sq,
    )?;
    check("min_price", expected.min_price, got.min_price)?;
    check("max_price", expected.max_price, got.max_price)?;

    if expected.tick_count != got.tick_count {
        return Err(format!(
            "{} mismatch for tick_count: expected {}, got {}",
            context, expected.tick_count, got.tick_count
        ));
    }

    Ok(())
}

fn fill_inputs(base_index: u64, len: usize, prices_ptr: *mut f32, sizes_ptr: *mut f32) {
    // StorageModeShared lets CPU write directly into GPU-visible buffers.
    unsafe {
        for i in 0..len {
            let idx = base_index + i as u64;
            prices_ptr.add(i).write(price_for_index(idx));
            sizes_ptr.add(i).write(size_for_index(idx));
        }
    }
}

fn encode_chunk(
    command_queue: &CommandQueue,
    transform_pipeline: &ComputePipelineState,
    reduce_pipeline: &ComputePipelineState,
    prices: &Buffer,
    sizes: &Buffer,
    derived: &Buffer,
    partials: &Buffer,
    params: &Buffer,
    chunk_len: usize,
    partial_groups: usize,
) -> CommandBuffer {
    let command_buffer = command_queue.new_command_buffer().to_owned();

    // Stage A: per-tick transform into derived features.
    let enc_transform = command_buffer.new_compute_command_encoder();
    enc_transform.set_compute_pipeline_state(transform_pipeline);
    enc_transform.set_buffer(0, Some(prices), 0);
    enc_transform.set_buffer(1, Some(sizes), 0);
    enc_transform.set_buffer(2, Some(derived), 0);
    enc_transform.set_buffer(3, Some(params), 0);
    // 1D launch: one thread per tick.
    enc_transform.dispatch_threads(
        MTLSize::new(chunk_len as u64, 1, 1),
        MTLSize::new(THREADGROUP_SIZE, 1, 1),
    );
    enc_transform.end_encoding();

    // Stage B: reduce derived rows into one partial aggregate per threadgroup.
    let enc_reduce = command_buffer.new_compute_command_encoder();
    enc_reduce.set_compute_pipeline_state(reduce_pipeline);
    enc_reduce.set_buffer(0, Some(derived), 0);
    enc_reduce.set_buffer(1, Some(partials), 0);
    enc_reduce.set_buffer(2, Some(params), 0);
    // One threadgroup writes one partial aggregate row.
    enc_reduce.dispatch_thread_groups(
        MTLSize::new(partial_groups as u64, 1, 1),
        MTLSize::new(THREADGROUP_SIZE, 1, 1),
    );
    enc_reduce.end_encoding();

    command_buffer.commit();
    command_buffer
}

struct SlotBuffers {
    prices: Buffer,
    sizes: Buffer,
    derived: Buffer,
    partials: Buffer,
    params: Buffer,
}

fn slot_aggregate(slot: &SlotBuffers, partial_groups: usize) -> AggregateTotals {
    let partial_ptr = slot.partials.contents() as *const PartialAggregate;
    let partials = unsafe { std::slice::from_raw_parts(partial_ptr, partial_groups) };

    // Initialize extrema so first valid partial sets them correctly.
    let mut agg = AggregateTotals {
        min_price: f64::INFINITY,
        max_price: f64::NEG_INFINITY,
        ..AggregateTotals::default()
    };
    fold_partials(partials, &mut agg);
    agg
}

fn finalize_chunk(
    in_flight: InFlight,
    slots: &[SlotBuffers],
    total_chunks: u64,
    progress_step_chunks: u64,
    config: Config,
    start: Instant,
    gpu_s: &mut f64,
    global: &mut AggregateTotals,
) -> Result<(), String> {
    // This wait is one of the core synchronization costs. Too many waits can
    // erase GPU gains on small batches.
    let wait_start = Instant::now();
    in_flight.command_buffer.wait_until_completed();
    *gpu_s += wait_start.elapsed().as_secs_f64();

    // Read back threadgroup partials for this chunk and fold on CPU.
    let chunk_gpu = slot_aggregate(&slots[in_flight.slot], in_flight.partial_groups);

    // Validation policy can be tightened/relaxed from CLI.
    let should_validate = match config.validation {
        ValidationMode::Off => false,
        ValidationMode::Full => true,
        ValidationMode::Spot => {
            in_flight.chunk_index == 0
                || should_spot_check(in_flight.chunk_index, total_chunks, progress_step_chunks)
        }
    };

    if should_validate {
        let expected = aggregate_range(in_flight.base_index, in_flight.len);
        assert_aggregate_close(
            expected,
            chunk_gpu,
            &format!("chunk {}", in_flight.chunk_index + 1),
        )?;
    }

    global.sum_size += chunk_gpu.sum_size;
    global.sum_notional += chunk_gpu.sum_notional;
    global.sum_return_proxy += chunk_gpu.sum_return_proxy;
    global.sum_return_proxy_sq += chunk_gpu.sum_return_proxy_sq;
    global.min_price = global.min_price.min(chunk_gpu.min_price);
    global.max_price = global.max_price.max(chunk_gpu.max_price);
    global.tick_count += chunk_gpu.tick_count;

    // Progress output cadence is independent from validation mode.
    if should_spot_check(in_flight.chunk_index, total_chunks, progress_step_chunks) {
        print_progress(
            in_flight.chunk_index,
            total_chunks,
            config.total_ticks,
            config.chunk_ticks,
            start,
        );
    }

    Ok(())
}

fn run_single(config: Config, gpu: &GpuResources, slots: &[SlotBuffers]) -> Result<(RunStats, AggregateTotals), String> {
    let total_chunks = config.total_ticks.div_ceil(config.chunk_ticks);
    let progress_step_chunks = ((total_chunks * config.progress_interval) / 100).max(1);

    let slot = &slots[0];
    let prices_ptr = slot.prices.contents() as *mut f32;
    let sizes_ptr = slot.sizes.contents() as *mut f32;
    let params_ptr = slot.params.contents() as *mut KernelParams;

    let start = Instant::now();
    let mut fill_s = 0.0;
    let mut gpu_s = 0.0;
    let mut global = AggregateTotals {
        min_price: f64::INFINITY,
        max_price: f64::NEG_INFINITY,
        ..AggregateTotals::default()
    };

    // Baseline execution model: each chunk fully completes before next starts.
    for chunk_index in 0..total_chunks {
        let base_index = chunk_index * config.chunk_ticks;
        let chunk_len = ((config.total_ticks - base_index).min(config.chunk_ticks)) as usize;
        let partial_groups = chunk_len.div_ceil(THREADGROUP_SIZE as usize).max(1);

        let fill_start = Instant::now();
        fill_inputs(base_index, chunk_len, prices_ptr, sizes_ptr);
        unsafe {
            params_ptr.write(KernelParams {
                count: chunk_len as u32,
                _pad: 0,
                base_index,
            });
        }
        fill_s += fill_start.elapsed().as_secs_f64();

        // Single mode is deliberately serialized: fill -> submit -> wait.
        let submit_start = Instant::now();
        let command_buffer = encode_chunk(
            &gpu.command_queue,
            &gpu.transform_pipeline,
            &gpu.reduce_pipeline,
            &slot.prices,
            &slot.sizes,
            &slot.derived,
            &slot.partials,
            &slot.params,
            chunk_len,
            partial_groups,
        );
        gpu_s += submit_start.elapsed().as_secs_f64();

        finalize_chunk(
            InFlight {
                command_buffer,
                chunk_index,
                base_index,
                len: chunk_len,
                slot: 0,
                partial_groups,
            },
            slots,
            total_chunks,
            progress_step_chunks,
            config,
            start,
            &mut gpu_s,
            &mut global,
        )?;
    }

    Ok((
        RunStats {
            wall_s: start.elapsed().as_secs_f64(),
            fill_s,
            gpu_s,
            ticks_processed: config.total_ticks,
        },
        global,
    ))
}

fn run_double(config: Config, gpu: &GpuResources, slots: &[SlotBuffers]) -> Result<(RunStats, AggregateTotals), String> {
    let total_chunks = config.total_ticks.div_ceil(config.chunk_ticks);
    let progress_step_chunks = ((total_chunks * config.progress_interval) / 100).max(1);

    let mut price_ptrs = [std::ptr::null_mut::<f32>(); 2];
    let mut size_ptrs = [std::ptr::null_mut::<f32>(); 2];
    let mut param_ptrs = [std::ptr::null_mut::<KernelParams>(); 2];
    for slot in 0..2 {
        price_ptrs[slot] = slots[slot].prices.contents() as *mut f32;
        size_ptrs[slot] = slots[slot].sizes.contents() as *mut f32;
        param_ptrs[slot] = slots[slot].params.contents() as *mut KernelParams;
    }

    let start = Instant::now();
    let mut fill_s = 0.0;
    let mut gpu_s = 0.0;
    let mut global = AggregateTotals {
        min_price: f64::INFINITY,
        max_price: f64::NEG_INFINITY,
        ..AggregateTotals::default()
    };
    let mut in_flight: [Option<InFlight>; 2] = [None, None];

    // Pipelined execution model with alternating slots.
    for chunk_index in 0..total_chunks {
        let slot_idx = (chunk_index % 2) as usize;

        // Wait-before-reuse is required to avoid clobbering in-flight slot data.
        if let Some(done) = in_flight[slot_idx].take() {
            finalize_chunk(
                done,
                slots,
                total_chunks,
                progress_step_chunks,
                config,
                start,
                &mut gpu_s,
                &mut global,
            )?;
        }

        let base_index = chunk_index * config.chunk_ticks;
        let chunk_len = ((config.total_ticks - base_index).min(config.chunk_ticks)) as usize;
        let partial_groups = chunk_len.div_ceil(THREADGROUP_SIZE as usize).max(1);

        let fill_start = Instant::now();
        fill_inputs(
            base_index,
            chunk_len,
            price_ptrs[slot_idx],
            size_ptrs[slot_idx],
        );
        unsafe {
            param_ptrs[slot_idx].write(KernelParams {
                count: chunk_len as u32,
                _pad: 0,
                base_index,
            });
        }
        fill_s += fill_start.elapsed().as_secs_f64();

        // Overlap point: CPU fills one slot while GPU can execute the other.
        let submit_start = Instant::now();
        let command_buffer = encode_chunk(
            &gpu.command_queue,
            &gpu.transform_pipeline,
            &gpu.reduce_pipeline,
            &slots[slot_idx].prices,
            &slots[slot_idx].sizes,
            &slots[slot_idx].derived,
            &slots[slot_idx].partials,
            &slots[slot_idx].params,
            chunk_len,
            partial_groups,
        );
        gpu_s += submit_start.elapsed().as_secs_f64();

        in_flight[slot_idx] = Some(InFlight {
            command_buffer,
            chunk_index,
            base_index,
            len: chunk_len,
            slot: slot_idx,
            partial_groups,
        });
    }

    // Drain leftovers after submit loop ends.
    while in_flight[0].is_some() || in_flight[1].is_some() {
        let slot_idx = match (&in_flight[0], &in_flight[1]) {
            (Some(a), Some(b)) => {
                if a.chunk_index <= b.chunk_index {
                    0
                } else {
                    1
                }
            }
            (Some(_), None) => 0,
            (None, Some(_)) => 1,
            (None, None) => unreachable!(),
        };

        if let Some(done) = in_flight[slot_idx].take() {
            finalize_chunk(
                done,
                slots,
                total_chunks,
                progress_step_chunks,
                config,
                start,
                &mut gpu_s,
                &mut global,
            )?;
        }
    }

    Ok((
        RunStats {
            wall_s: start.elapsed().as_secs_f64(),
            fill_s,
            gpu_s,
            ticks_processed: config.total_ticks,
        },
        global,
    ))
}

fn run_cpu_baseline(config: Config) -> (f64, AggregateTotals) {
    let start = Instant::now();
    let mut totals = AggregateTotals {
        min_price: f64::INFINITY,
        max_price: f64::NEG_INFINITY,
        ..AggregateTotals::default()
    };

    // Keep chunk loop shape similar to GPU path for a fair throughput compare.
    let mut done = 0u64;
    while done < config.total_ticks {
        let len = (config.total_ticks - done).min(config.chunk_ticks) as usize;
        let chunk = aggregate_range(done, len);

        totals.sum_size += chunk.sum_size;
        totals.sum_notional += chunk.sum_notional;
        totals.sum_return_proxy += chunk.sum_return_proxy;
        totals.sum_return_proxy_sq += chunk.sum_return_proxy_sq;
        totals.min_price = totals.min_price.min(chunk.min_price);
        totals.max_price = totals.max_price.max(chunk.max_price);
        totals.tick_count += chunk.tick_count;

        done += len as u64;
    }

    (start.elapsed().as_secs_f64(), totals)
}

fn print_metrics(label: &str, m: FinalMetrics) {
    println!("{}", label);
    println!("  tick_count:            {}", m.tick_count);
    println!("  total_volume:          {:.6}", m.total_volume);
    println!("  total_notional:        {:.6}", m.total_notional);
    println!("  VWAP:                  {:.6}", m.vwap);
    println!("  mean_return_proxy:     {:.6}", m.mean_return_proxy);
    println!("  volatility_proxy:      {:.6}", m.volatility_proxy);
    println!("  min_price:             {:.6}", m.min_price);
    println!("  max_price:             {:.6}", m.max_price);
}

fn print_summary(config: Config, gpu_stats: RunStats, gpu_totals: AggregateTotals, cpu: Option<(f64, AggregateTotals)>) {
    let gpu_metrics = totals_to_metrics(gpu_totals);

    println!();
    println!("============================================================");
    println!(" Lesson 06 Results");
    println!("============================================================");
    println!("Mode:                {}", config.mode.as_str());
    println!("Validation:          {}", config.validation.as_str());
    println!("CPU baseline:        {}", config.compare_cpu.as_str());
    println!("GPU wall time:       {:>9.3}s", gpu_stats.wall_s);
    println!(
        "GPU submit/wait:     {:>9.3}s ({:>5.1}%)",
        gpu_stats.gpu_s,
        (gpu_stats.gpu_s / gpu_stats.wall_s) * 100.0
    );
    println!(
        "CPU fill time:       {:>9.3}s ({:>5.1}%)",
        gpu_stats.fill_s,
        (gpu_stats.fill_s / gpu_stats.wall_s) * 100.0
    );
    println!(
        "GPU throughput:      {:>9.3} million ticks/sec",
        gpu_stats.ticks_processed as f64 / 1e6 / gpu_stats.wall_s
    );

    print_metrics("GPU aggregate metrics:", gpu_metrics);

    if let Some((cpu_time, cpu_totals)) = cpu {
        let cpu_metrics = totals_to_metrics(cpu_totals);
        println!();
        println!("CPU baseline time:   {:>9.3}s", cpu_time);
        println!(
            "CPU throughput:      {:>9.3} million ticks/sec",
            gpu_stats.ticks_processed as f64 / 1e6 / cpu_time
        );
        println!("Speedup (CPU/GPU):   {:>9.3}x", cpu_time / gpu_stats.wall_s);

        // Sanity check that CPU and GPU aggregate to effectively the same result.
        let _ = assert_aggregate_close(cpu_totals, gpu_totals, "global totals")
            .map_err(|err| eprintln!("Warning: CPU/GPU aggregate delta: {err}"));

        print_metrics("CPU aggregate metrics:", cpu_metrics);
    }

    println!();
    println!("When CPU might be better:");
    println!("  - Small tick batches where dispatch/sync overhead dominates");
    println!("  - Highly branchy logic with divergent control flow");
    println!("  - Pipelines forced to synchronize with the host very frequently");
    println!();
    println!("Why this GPU pattern scales:");
    println!("  - Stage A is embarrassingly parallel across ticks");
    println!("  - Stage B uses threadgroup reduction then CPU fold, avoiding one hot atomic");
    println!("  - Double mode overlaps CPU fill with GPU execution to reduce idle time");
}

fn main() {
    autoreleasepool(|| {
        // 1) Parse CLI and reject invalid args early.
        let cli = match parse_args() {
            Ok(v) => v,
            Err(err) => {
                eprintln!("Argument error: {err}");
                eprintln!();
                print_usage();
                std::process::exit(2);
            }
        };

        if cli.total_ticks == 0 {
            println!("No work requested (--total-ticks=0). Exiting.");
            return;
        }

        let device = Device::system_default().expect("No Metal-capable GPU found!");
        let config = resolve_chunk_ticks(cli, &device);

        // 2) Compute required buffer sizes for one slot and enforce device caps.
        let derived_bytes = config.chunk_ticks as u128 * std::mem::size_of::<TickDerived>() as u128;
        let partial_groups = config.chunk_ticks.div_ceil(THREADGROUP_SIZE);
        let partial_bytes = partial_groups as u128 * std::mem::size_of::<PartialAggregate>() as u128;
        let prices_bytes = config.chunk_ticks as u128 * std::mem::size_of::<f32>() as u128;
        let sizes_bytes = prices_bytes;

        let max_buffer = device.max_buffer_length() as u128;
        for (name, bytes) in [
            ("prices", prices_bytes),
            ("sizes", sizes_bytes),
            ("derived", derived_bytes),
            ("partials", partial_bytes),
            ("params", std::mem::size_of::<KernelParams>() as u128),
        ] {
            if bytes > max_buffer {
                eprintln!(
                    "Buffer '{}' size {:.3} GB exceeds device max {:.3} GB. Reduce --chunk-ticks.",
                    name,
                    bytes as f64 / 1e9,
                    max_buffer as f64 / 1e9
                );
                std::process::exit(1);
            }
        }

        let total_chunks = config.total_ticks.div_ceil(config.chunk_ticks);

        println!("============================================================");
        println!(" Lesson 06: Market Tick Aggregation");
        println!("============================================================");
        println!("GPU:                 {}", device.name());
        println!("Mode:                {}", config.mode.as_str());
        println!("Validation:          {}", config.validation.as_str());
        println!("CPU baseline:        {}", config.compare_cpu.as_str());
        println!(
            "Total ticks:         {} ({:.2} billion)",
            config.total_ticks,
            config.total_ticks as f64 / 1e9
        );

        if config.chunk_auto {
            match config.detected_ram_bytes {
                Some(ram) => println!(
                    "System RAM detected: {:.2} GB (using {:.0}% target)",
                    ram as f64 / 1e9,
                    config.memory_fraction * 100.0
                ),
                None => println!(
                    "System RAM detected: unavailable (fallback auto chunking at {:.0}% target)",
                    config.memory_fraction * 100.0
                ),
            }
            println!(
                "Chunk ticks:         {} ({:.2} million, auto)",
                config.chunk_ticks,
                config.chunk_ticks as f64 / 1e6
            );
        } else {
            println!(
                "Chunk ticks:         {} ({:.2} million, manual)",
                config.chunk_ticks,
                config.chunk_ticks as f64 / 1e6
            );
        }

        println!("Total chunks:        {}", total_chunks);
        println!("Derived buffer:      {:.3} GB", derived_bytes as f64 / 1e9);
        println!();

        // 3) Compile shader source and build pipeline states once per run.
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(SHADER_SOURCE, &options)
            .expect("Failed to compile Metal shader source");

        let transform_func = library
            .get_function("transform_ticks", None)
            .expect("Missing transform_ticks shader function");
        let reduce_func = library
            .get_function("reduce_partials", None)
            .expect("Missing reduce_partials shader function");

        let transform_pipeline = device
            .new_compute_pipeline_state_with_function(&transform_func)
            .expect("Failed creating transform pipeline");
        let reduce_pipeline = device
            .new_compute_pipeline_state_with_function(&reduce_func)
            .expect("Failed creating reduce pipeline");

        // Defensive checks so threadgroup constants stay valid across hardware.
        if transform_pipeline.max_total_threads_per_threadgroup() < THREADGROUP_SIZE {
            eprintln!(
                "GPU supports max {} threads/threadgroup for transform, but lesson uses {}.",
                transform_pipeline.max_total_threads_per_threadgroup(),
                THREADGROUP_SIZE
            );
            std::process::exit(1);
        }
        if reduce_pipeline.max_total_threads_per_threadgroup() < THREADGROUP_SIZE {
            eprintln!(
                "GPU supports max {} threads/threadgroup for reduce, but lesson uses {}.",
                reduce_pipeline.max_total_threads_per_threadgroup(),
                THREADGROUP_SIZE
            );
            std::process::exit(1);
        }

        let command_queue = device.new_command_queue();
        let gpu = GpuResources {
            command_queue,
            transform_pipeline,
            reduce_pipeline,
        };

        // 4) Allocate one slot (`single`) or two slots (`double`) for overlap.
        let slot_count = match config.mode {
            Mode::Single => 1,
            Mode::Double => 2,
        };

        let mut slots = Vec::with_capacity(slot_count);
        for _ in 0..slot_count {
            let prices = device.new_buffer(prices_bytes as u64, MTLResourceOptions::StorageModeShared);
            let sizes = device.new_buffer(sizes_bytes as u64, MTLResourceOptions::StorageModeShared);
            let derived = device.new_buffer(derived_bytes as u64, MTLResourceOptions::StorageModeShared);
            let partials = device.new_buffer(partial_bytes as u64, MTLResourceOptions::StorageModeShared);
            let params = device.new_buffer(
                std::mem::size_of::<KernelParams>() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            slots.push(SlotBuffers {
                prices,
                sizes,
                derived,
                partials,
                params,
            });
        }

        // 5) Execute GPU path.
        let gpu_result = match config.mode {
            Mode::Single => run_single(config, &gpu, &slots),
            Mode::Double => run_double(config, &gpu, &slots),
        };

        let (gpu_stats, gpu_totals) = match gpu_result {
            Ok(v) => v,
            Err(err) => {
                eprintln!("Execution failed: {err}");
                std::process::exit(1);
            }
        };

        // 6) Optional CPU baseline for crossover/speedup interpretation.
        let cpu_result = if config.compare_cpu == CompareCpu::On {
            Some(run_cpu_baseline(config))
        } else {
            None
        };

        // 7) Print final metrics and practical GPU-vs-CPU guidance.
        print_summary(config, gpu_stats, gpu_totals, cpu_result);
    });
}
