// =============================================================================
// LESSON 5: Double Buffering -- Overlapping CPU Fill with GPU Compute
// =============================================================================
//
// Lesson 2 introduced chunking so we can process datasets larger than memory.
// That solved the memory-capacity problem, but it still leaves a pipeline stall:
//
//   single-buffer loop:
//     1) CPU fills input buffer for chunk N
//     2) GPU processes chunk N
//     3) CPU waits and repeats
//
// If CPU fill is non-trivial, the GPU sits idle between chunks.
//
// THE SOLUTION: DOUBLE BUFFERING
//
// Keep two buffer slots and alternate between them:
//   - Slot A can be in-flight on GPU while CPU fills Slot B
//   - Slot B can be in-flight on GPU while CPU fills Slot A
//
// Correctness rule: never reuse a slot until its previous command buffer is
// complete, otherwise CPU could overwrite data the GPU is still reading.
//
// NEW IN THIS LESSON:
//   - Baseline mode (`single`) and overlapped mode (`double`) in one binary
//   - In-flight command tracking per slot
//   - Deterministic per-chunk validation strategy
//   - RAM-aware auto chunk sizing (with explicit override)
//   - Throughput/bandwidth reporting with overlap interpretation
//
// RUN:
//   cargo run --release -p double-buffering
//   cargo run --release -p double-buffering -- --mode single
//   cargo run --release -p double-buffering -- --memory-fraction 0.95
// =============================================================================

use metal::*;
use objc::rc::autoreleasepool;
use std::env;
use std::ffi::CString;
use std::ptr;
use std::time::Instant;

// =============================================================================
// CONFIGURATION DEFAULTS
// =============================================================================

/// Default total element count when not provided on the CLI.
///
/// This keeps the lesson focused on large-scale streaming behavior.
const DEFAULT_TOTAL_ELEMENTS: u64 = 100_000_000_000;

/// Default progress cadence in percent of total chunks.
///
/// Example: 10 means we print progress roughly every 10% and on the final chunk.
const DEFAULT_PROGRESS_INTERVAL: u64 = 10;

/// Fraction of detected system RAM to target for active working buffers when
/// `--chunk-elements` is not explicitly provided.
///
/// We intentionally leave headroom for the OS and other processes.
const DEFAULT_MEMORY_FRACTION: f64 = 0.85;

/// Fallback chunk size (elements) when RAM detection fails.
///
/// This preserves predictable behavior on environments where `hw.memsize`
/// cannot be queried.
const DEFAULT_FALLBACK_CHUNK_ELEMENTS: u64 = 250_000_000;

// =============================================================================
// THE METAL SHADER
// =============================================================================
//
// The kernel is intentionally simple (multiply by 2). The point of Lesson 5 is
// not arithmetic complexity but pipeline orchestration and overlap behavior.
//
// Buffer mapping:
//   buffer(0) -> input
//   buffer(1) -> output
//
// `gid` is the global thread index, so each thread handles one element.
const SHADER_SOURCE: &str = r#"
    #include <metal_stdlib>
    using namespace metal;

    kernel void double_values(
        device const float *input  [[ buffer(0) ]],
        device float       *output [[ buffer(1) ]],
        uint gid                   [[ thread_position_in_grid ]]
    ) {
        output[gid] = input[gid] * 2.0;
    }
"#;

/// Execution mode.
///
/// We keep both in one binary so users can compare baseline and overlapped
/// behavior with identical shader, data generation, and validation rules.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Single,
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

/// Raw CLI intent before runtime-dependent resolution.
///
/// `chunk_elements_override` lets users bypass auto sizing for repeatable tests.
#[derive(Clone, Copy, Debug)]
struct CliConfig {
    mode: Mode,
    total_elements: u64,
    chunk_elements_override: Option<u64>,
    progress_interval: u64,
    memory_fraction: f64,
}

/// Fully resolved runtime config used by execution paths.
///
/// This includes derived fields such as chosen chunk size and whether it was
/// auto-selected from RAM information.
#[derive(Clone, Copy, Debug)]
struct Config {
    mode: Mode,
    total_elements: u64,
    chunk_elements: u64,
    progress_interval: u64,
    memory_fraction: f64,
    detected_ram_bytes: Option<u64>,
    chunk_auto: bool,
}

/// End-of-run performance metrics.
///
/// We separate fill time and GPU submit/wait time so users can see which stage
/// dominates and whether overlap helped wall-clock throughput.
#[derive(Clone, Copy, Debug, Default)]
struct RunStats {
    wall_s: f64,
    fill_s: f64,
    gpu_s: f64,
    elements_processed: u64,
}

/// Metadata for a command currently occupying a slot in double-buffer mode.
///
/// We retain ownership of `command_buffer` so we can wait/validate later when
/// that slot is reused or during final drain.
struct InFlight {
    command_buffer: CommandBuffer,
    chunk_index: u64,
    base_index: u64,
    len: usize,
}

/// Complete one in-flight slot: wait, validate, and optionally print progress.
///
/// Centralizing this logic keeps slot-finalization behavior identical for both
/// "reuse" and "drain" phases in double mode.
fn finalize_in_flight(
    done: InFlight,
    output_ptr: *const f32,
    total_chunks: u64,
    progress_step_chunks: u64,
    config: Config,
    overall_start: Instant,
    gpu_s: &mut f64,
) -> Result<(), String> {
    // Waiting here is mandatory for correctness before we touch output data.
    let wait_start = Instant::now();
    done.command_buffer.wait_until_completed();
    *gpu_s += wait_start.elapsed().as_secs_f64();

    let output = unsafe { std::slice::from_raw_parts(output_ptr, done.len) };
    let full_check = done.chunk_index == 0;
    let spot_check = should_validate_spot(done.chunk_index, total_chunks, progress_step_chunks);
    if full_check || spot_check {
        validate_chunk(done.base_index, done.len, output, full_check)?;
    }

    if spot_check {
        print_progress(
            done.chunk_index,
            total_chunks,
            config.total_elements,
            config.chunk_elements,
            overall_start,
        );
    }

    Ok(())
}

/// Print CLI usage and defaults.
///
/// This is intentionally explicit so users can quickly discover auto-vs-manual
/// chunking behavior without reading source.
fn print_usage() {
    println!("Usage: double-buffering [--mode single|double] [--total-elements N] [--chunk-elements N] [--memory-fraction F] [--progress-interval PCT]");
    println!();
    println!("Defaults:");
    println!("  --mode double");
    println!("  --total-elements {}", DEFAULT_TOTAL_ELEMENTS);
    println!("  --chunk-elements auto (derived from RAM and mode)");
    println!("  --memory-fraction {:.2}", DEFAULT_MEMORY_FRACTION);
    println!("  --progress-interval {}", DEFAULT_PROGRESS_INTERVAL);
}

/// Parse unsigned integer flags with consistent, contextual error messages.
fn parse_u64(name: &str, value: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|_| format!("Invalid value for {name}: '{value}' (expected unsigned integer)"))
}

/// Parse floating-point flags with consistent, contextual error messages.
fn parse_f64(name: &str, value: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .map_err(|_| format!("Invalid value for {name}: '{value}' (expected float)"))
}

/// Parse CLI flags into raw user intent.
///
/// We use a small explicit scanner (instead of extra dependencies) to keep the
/// lesson self-contained. Validation guardrails are applied here so execution
/// paths can assume sane inputs.
fn parse_args() -> Result<CliConfig, String> {
    let mut mode = Mode::Double;
    let mut total_elements = DEFAULT_TOTAL_ELEMENTS;
    let mut chunk_elements_override: Option<u64> = None;
    let mut progress_interval = DEFAULT_PROGRESS_INTERVAL;
    let mut memory_fraction = DEFAULT_MEMORY_FRACTION;

    let args: Vec<String> = env::args().collect();
    let mut i = 1usize;

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
                    _ => {
                        return Err(format!(
                            "Invalid --mode value '{value}'. Expected 'single' or 'double'"
                        ));
                    }
                };
                i += 2;
            }
            "--total-elements" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| "Missing value for --total-elements".to_string())?;
                total_elements = parse_u64("--total-elements", value)?;
                i += 2;
            }
            "--chunk-elements" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| "Missing value for --chunk-elements".to_string())?;
                chunk_elements_override = Some(parse_u64("--chunk-elements", value)?);
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
            flag => {
                return Err(format!("Unknown argument: {flag}"));
            }
        }
    }

    // Guardrails: keep invalid inputs from leaking into runtime logic.
    if chunk_elements_override == Some(0) {
        return Err("--chunk-elements must be greater than 0 when provided".to_string());
    }
    // Memory fraction is bounded so "auto" remains meaningful and safe.
    if !(memory_fraction > 0.0 && memory_fraction <= 1.0) {
        return Err("--memory-fraction must be in the range (0.0, 1.0]".to_string());
    }
    if progress_interval == 0 || progress_interval > 100 {
        return Err("--progress-interval must be in the range 1..=100".to_string());
    }

    Ok(CliConfig {
        mode,
        total_elements,
        chunk_elements_override,
        progress_interval,
        memory_fraction,
    })
}

/// Detect physical system RAM using macOS `sysctlbyname("hw.memsize")`.
///
/// Returns `None` on failure instead of hard-failing the program so we can fall
/// back to a conservative default chunk size.
fn detect_system_ram_bytes() -> Option<u64> {
    let mut ram_bytes: u64 = 0;
    let mut size = std::mem::size_of::<u64>();
    let name = CString::new("hw.memsize").ok()?;

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

/// Resolve final chunk size from CLI intent plus hardware/runtime constraints.
///
/// Resolution order for auto mode:
///   1) RAM budget target (`memory_fraction` of detected RAM)
///   2) divide by in-flight buffer count (single=2, double=4)
///   3) clamp to Metal max buffer length
///   4) clamp to workload size (no chunk bigger than total data)
///   5) enforce minimum 1 element
///
/// Explicit `--chunk-elements` always wins over auto sizing.
fn resolve_chunk_elements(cli: CliConfig, device: &Device) -> Config {
    let detected_ram_bytes = detect_system_ram_bytes();
    let max_buffer_bytes = device.max_buffer_length() as u128;

    let (chunk_elements, chunk_auto) = if let Some(manual) = cli.chunk_elements_override {
        (manual, false)
    } else {
        let in_flight_buffers = match cli.mode {
            Mode::Single => 2u128,
            Mode::Double => 4u128,
        };

        let target_total_bytes = detected_ram_bytes
            .map(|bytes| (bytes as f64 * cli.memory_fraction) as u128)
            .unwrap_or(DEFAULT_FALLBACK_CHUNK_ELEMENTS as u128 * 4u128 * in_flight_buffers);

        let mut per_buffer_bytes = (target_total_bytes / in_flight_buffers).max(4);
        per_buffer_bytes = per_buffer_bytes.min(max_buffer_bytes);
        per_buffer_bytes = per_buffer_bytes.min(cli.total_elements as u128 * 4);

        let elems = (per_buffer_bytes / 4).max(1) as u64;
        (elems, true)
    };

    Config {
        mode: cli.mode,
        total_elements: cli.total_elements,
        chunk_elements,
        progress_interval: cli.progress_interval,
        memory_fraction: cli.memory_fraction,
        detected_ram_bytes,
        chunk_auto,
    }
}

/// Deterministic value generator for input data.
///
/// Modular mapping keeps values in a precise `f32`-friendly range, even when
/// global indices become very large.
fn value_for_index(global_idx: u64) -> f32 {
    (global_idx % 1_000_000) as f32 + 1.0
}

/// Validate output for one chunk.
///
/// - Full mode: check every element (used for first chunk)
/// - Spot mode: check start/middle/end indices (used for periodic checks)
fn validate_chunk(base_index: u64, len: usize, output: &[f32], full: bool) -> Result<(), String> {
    if len == 0 {
        return Ok(());
    }

    if full {
        for i in 0..len {
            let expected = value_for_index(base_index + i as u64) * 2.0;
            let got = output[i];
            if (got - expected).abs() > 0.01 {
                return Err(format!(
                    "Validation failed at global index {}: expected {}, got {}",
                    base_index + i as u64,
                    expected,
                    got
                ));
            }
        }
        return Ok(());
    }

    let indices = [0usize, len / 2, len - 1];
    for idx in indices {
        let expected = value_for_index(base_index + idx as u64) * 2.0;
        let got = output[idx];
        if (got - expected).abs() > 0.01 {
            return Err(format!(
                "Spot-check failed at global index {}: expected {}, got {}",
                base_index + idx as u64,
                expected,
                got
            ));
        }
    }

    Ok(())
}

/// Print human-readable progress for long runs.
fn print_progress(
    chunk_index: u64,
    total_chunks: u64,
    total_elements: u64,
    chunk_elements: u64,
    start: Instant,
) {
    let done_chunks = chunk_index + 1;
    let done_elements = done_chunks.saturating_mul(chunk_elements).min(total_elements);
    let pct = if total_elements == 0 {
        100.0
    } else {
        done_elements as f64 / total_elements as f64 * 100.0
    };

    println!(
        "  Chunk {:>6}/{}: {:>7.2}B elements ({:>5.1}%) - {:>7.2}s",
        done_chunks,
        total_chunks,
        done_elements as f64 / 1e9,
        pct,
        start.elapsed().as_secs_f64(),
    );
}

/// Decide whether to run spot validation/progress print for a chunk.
///
/// We always include the final chunk so completion is explicit.
fn should_validate_spot(chunk_index: u64, total_chunks: u64, progress_step_chunks: u64) -> bool {
    chunk_index == total_chunks - 1 || (chunk_index + 1) % progress_step_chunks == 0
}

/// Baseline execution path: one input/output buffer set, wait each chunk.
///
/// This is intentionally simple and serves as the "no overlap" reference.
fn run_single(
    config: Config,
    command_queue: &CommandQueue,
    pipeline_state: &ComputePipelineState,
    input_buffer: &Buffer,
    output_buffer: &Buffer,
    progress_step_chunks: u64,
) -> Result<RunStats, String> {
    let total_chunks = config.total_elements.div_ceil(config.chunk_elements);

    let input_ptr = input_buffer.contents() as *mut f32;
    let output_ptr = output_buffer.contents() as *const f32;

    let overall_start = Instant::now();
    let mut fill_s = 0.0;
    let mut gpu_s = 0.0;

    for chunk_index in 0..total_chunks {
        // ---------------------------------------------------------------------
        // Phase 1: choose active range for this chunk (handles last partial)
        // ---------------------------------------------------------------------
        let base_index = chunk_index * config.chunk_elements;
        let active_u64 = (config.total_elements - base_index).min(config.chunk_elements);
        let active = active_u64 as usize;

        // ---------------------------------------------------------------------
        // Phase 2: fill shared input buffer from CPU
        // ---------------------------------------------------------------------
        let fill_start = Instant::now();
        unsafe {
            for i in 0..active {
                input_ptr.add(i).write(value_for_index(base_index + i as u64));
            }
        }
        fill_s += fill_start.elapsed().as_secs_f64();

        // ---------------------------------------------------------------------
        // Phase 3: dispatch and wait (no overlap in single mode)
        // ---------------------------------------------------------------------
        let gpu_start = Instant::now();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline_state);
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(output_buffer), 0);
        encoder.dispatch_threads(
            MTLSize::new(active as u64, 1, 1),
            MTLSize::new(pipeline_state.thread_execution_width(), 1, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        gpu_s += gpu_start.elapsed().as_secs_f64();

        // ---------------------------------------------------------------------
        // Phase 4: validate and optionally print progress
        // ---------------------------------------------------------------------
        let output = unsafe { std::slice::from_raw_parts(output_ptr, active) };
        let full_check = chunk_index == 0;
        let spot_check = should_validate_spot(chunk_index, total_chunks, progress_step_chunks);
        if full_check || spot_check {
            validate_chunk(base_index, active, output, full_check)?;
        }

        if spot_check {
            print_progress(
                chunk_index,
                total_chunks,
                config.total_elements,
                config.chunk_elements,
                overall_start,
            );
        }
    }

    let wall_s = overall_start.elapsed().as_secs_f64();
    Ok(RunStats {
        wall_s,
        fill_s,
        gpu_s,
        elements_processed: config.total_elements,
    })
}

/// Overlapped execution path: two slots with wait-before-reuse synchronization.
///
/// Slot policy:
///   - slot = chunk_index % 2
///   - if slot already in use, finalize that in-flight work first
///   - fill + submit new chunk into freed slot
///
/// This pattern lets CPU fill next slot while GPU works on previously submitted
/// slot, reducing idle time versus single mode.
fn run_double(
    config: Config,
    command_queue: &CommandQueue,
    pipeline_state: &ComputePipelineState,
    input_buffers: &[Buffer; 2],
    output_buffers: &[Buffer; 2],
    progress_step_chunks: u64,
) -> Result<RunStats, String> {
    let total_chunks = config.total_elements.div_ceil(config.chunk_elements);

    let input_ptrs = [
        input_buffers[0].contents() as *mut f32,
        input_buffers[1].contents() as *mut f32,
    ];
    let output_ptrs = [
        output_buffers[0].contents() as *const f32,
        output_buffers[1].contents() as *const f32,
    ];

    let overall_start = Instant::now();
    let mut fill_s = 0.0;
    let mut gpu_s = 0.0;

    // One in-flight record per slot, if occupied.
    let mut in_flight: [Option<InFlight>; 2] = [None, None];

    for chunk_index in 0..total_chunks {
        let slot = (chunk_index % 2) as usize;

        // ---------------------------------------------------------------------
        // Phase 1: wait-before-reuse for this slot
        // ---------------------------------------------------------------------
        // This is the core correctness barrier in double buffering.
        if let Some(done) = in_flight[slot].take() {
            finalize_in_flight(
                done,
                output_ptrs[slot],
                total_chunks,
                progress_step_chunks,
                config,
                overall_start,
                &mut gpu_s,
            )?;
        }

        // ---------------------------------------------------------------------
        // Phase 2: select active range for this chunk
        // ---------------------------------------------------------------------
        let base_index = chunk_index * config.chunk_elements;
        let active_u64 = (config.total_elements - base_index).min(config.chunk_elements);
        let active = active_u64 as usize;

        // ---------------------------------------------------------------------
        // Phase 3: fill selected slot from CPU
        // ---------------------------------------------------------------------
        let fill_start = Instant::now();
        unsafe {
            for i in 0..active {
                input_ptrs[slot].add(i).write(value_for_index(base_index + i as u64));
            }
        }
        fill_s += fill_start.elapsed().as_secs_f64();

        // ---------------------------------------------------------------------
        // Phase 4: submit selected slot to GPU and mark it in-flight
        // ---------------------------------------------------------------------
        let submit_start = Instant::now();
        let command_buffer = command_queue.new_command_buffer().to_owned();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline_state);
        encoder.set_buffer(0, Some(&input_buffers[slot]), 0);
        encoder.set_buffer(1, Some(&output_buffers[slot]), 0);
        encoder.dispatch_threads(
            MTLSize::new(active as u64, 1, 1),
            MTLSize::new(pipeline_state.thread_execution_width(), 1, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        gpu_s += submit_start.elapsed().as_secs_f64();

        in_flight[slot] = Some(InFlight {
            command_buffer,
            chunk_index,
            base_index,
            len: active,
        });
    }

    // -------------------------------------------------------------------------
    // Phase 5: drain remaining in-flight slots after submission loop
    // -------------------------------------------------------------------------
    // We finalize in chunk-index order so progress output stays monotonic.
    while in_flight[0].is_some() || in_flight[1].is_some() {
        let slot = match (&in_flight[0], &in_flight[1]) {
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

        let done = in_flight[slot]
            .take()
            .expect("slot should have in-flight command");
        finalize_in_flight(
            done,
            output_ptrs[slot],
            total_chunks,
            progress_step_chunks,
            config,
            overall_start,
            &mut gpu_s,
        )?;
    }

    let wall_s = overall_start.elapsed().as_secs_f64();
    Ok(RunStats {
        wall_s,
        fill_s,
        gpu_s,
        elements_processed: config.total_elements,
    })
}

/// Print end-of-run metrics and interpretation hints.
///
/// We report both rates and timing composition so users can reason about where
/// optimization headroom exists.
fn print_summary(config: Config, stats: RunStats) {
    let data_gb = stats.elements_processed as f64 * 4.0 / 1e9;
    let read_write_gb = data_gb * 2.0;

    println!();
    println!("============================================================");
    println!(" Mode: {}", config.mode.as_str());
    println!("============================================================");
    println!("Total wall time:      {:>8.3}s", stats.wall_s);
    println!(
        "CPU fill time:        {:>8.3}s ({:>5.1}%)",
        stats.fill_s,
        (stats.fill_s / stats.wall_s) * 100.0
    );
    println!(
        "GPU submit/wait time: {:>8.3}s ({:>5.1}%)",
        stats.gpu_s,
        (stats.gpu_s / stats.wall_s) * 100.0
    );
    println!();
    println!("Throughput:");
    println!(
        "  Effective: {:>9.3} billion elements/sec",
        stats.elements_processed as f64 / 1e9 / stats.wall_s
    );
    println!(
        "  Bandwidth: {:>9.3} GB/s (read + write)",
        read_write_gb / stats.wall_s
    );

    if config.mode == Mode::Double {
        println!();
        println!("Double-buffering overlap note:");
        println!("  Wall time can approach max(CPU fill, GPU compute) instead of their sum.");
        println!("  For small chunks or debug builds, overhead can dominate; use --release and larger chunks.");
    }
}

fn main() {
    autoreleasepool(|| {
        // =====================================================================
        // STEP 1: Parse and validate CLI arguments
        // =====================================================================
        let cli = match parse_args() {
            Ok(cfg) => cfg,
            Err(err) => {
                eprintln!("Argument error: {err}");
                eprintln!();
                print_usage();
                std::process::exit(2);
            }
        };

        // =====================================================================
        // STEP 2: Handle trivial no-op workload early
        // =====================================================================
        if cli.total_elements == 0 {
            println!("No work requested (--total-elements=0). Exiting.");
            return;
        }

        // =====================================================================
        // STEP 3: Discover device and resolve runtime config
        // =====================================================================
        let device = Device::system_default().expect("No Metal-capable GPU found!");
        let config = resolve_chunk_elements(cli, &device);

        // =====================================================================
        // STEP 4: Enforce buffer-size safety constraints
        // =====================================================================
        let chunk_bytes = config.chunk_elements as u128 * std::mem::size_of::<f32>() as u128;
        let max_buffer_len = device.max_buffer_length() as u128;
        if chunk_bytes > max_buffer_len {
            eprintln!(
                "Requested chunk buffer size ({:.3} GB) exceeds device max buffer length ({:.3} GB).",
                chunk_bytes as f64 / 1e9,
                max_buffer_len as f64 / 1e9
            );
            eprintln!("Reduce --chunk-elements and try again.");
            std::process::exit(1);
        }

        let total_chunks = config.total_elements.div_ceil(config.chunk_elements);
        let progress_step_chunks = ((total_chunks * config.progress_interval) / 100).max(1);

        // =====================================================================
        // STEP 5: Print resolved run configuration
        // =====================================================================
        println!("============================================================");
        println!(" Lesson 05: Double Buffering");
        println!("============================================================");
        println!("GPU:                 {}", device.name());
        println!("Mode:                {}", config.mode.as_str());
        println!(
            "Total elements:      {} ({:.2} billion)",
            config.total_elements,
            config.total_elements as f64 / 1e9
        );

        // Auto-vs-manual printing is explicit so users can confirm exactly how
        // chunk size was chosen for this run.
        if config.chunk_auto {
            match config.detected_ram_bytes {
                Some(ram) => println!(
                    "System RAM detected: {:.2} GB (using {:.0}% target)",
                    ram as f64 / 1e9,
                    config.memory_fraction * 100.0
                ),
                None => println!(
                    "System RAM detected: unavailable (using fallback auto chunking; memory fraction {:.0}%)",
                    config.memory_fraction * 100.0
                ),
            }
            println!(
                "Chunk elements:      {} ({:.2} million, auto)",
                config.chunk_elements,
                config.chunk_elements as f64 / 1e6
            );
        } else {
            println!(
                "Chunk elements:      {} ({:.2} million, manual)",
                config.chunk_elements,
                config.chunk_elements as f64 / 1e6
            );
        }
        println!("Total chunks:        {}", total_chunks);
        println!("Chunk buffer size:   {:.3} GB per buffer", chunk_bytes as f64 / 1e9);
        println!();

        // =====================================================================
        // STEP 6: Build compute pipeline once
        // =====================================================================
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(SHADER_SOURCE, &options)
            .expect("Failed to compile shader source for kernel 'double_values'");
        let kernel = library
            .get_function("double_values", None)
            .expect("Failed to load shader function 'double_values'");
        let pipeline_state = device
            .new_compute_pipeline_state_with_function(&kernel)
            .expect("Failed to create compute pipeline state");

        let command_queue = device.new_command_queue();
        let buffer_bytes = chunk_bytes as u64;

        // =====================================================================
        // STEP 7: Execute selected mode (single or double)
        // =====================================================================
        let stats = match config.mode {
            Mode::Single => {
                let input_buffer =
                    device.new_buffer(buffer_bytes, MTLResourceOptions::StorageModeShared);
                let output_buffer =
                    device.new_buffer(buffer_bytes, MTLResourceOptions::StorageModeShared);

                run_single(
                    config,
                    &command_queue,
                    &pipeline_state,
                    &input_buffer,
                    &output_buffer,
                    progress_step_chunks,
                )
            }
            Mode::Double => {
                let input_buffers = [
                    device.new_buffer(buffer_bytes, MTLResourceOptions::StorageModeShared),
                    device.new_buffer(buffer_bytes, MTLResourceOptions::StorageModeShared),
                ];
                let output_buffers = [
                    device.new_buffer(buffer_bytes, MTLResourceOptions::StorageModeShared),
                    device.new_buffer(buffer_bytes, MTLResourceOptions::StorageModeShared),
                ];

                run_double(
                    config,
                    &command_queue,
                    &pipeline_state,
                    &input_buffers,
                    &output_buffers,
                    progress_step_chunks,
                )
            }
        };

        // =====================================================================
        // STEP 8: Report results
        // =====================================================================
        match stats {
            Ok(ok) => print_summary(config, ok),
            Err(err) => {
                eprintln!("Execution failed: {err}");
                std::process::exit(1);
            }
        }
    });
}
