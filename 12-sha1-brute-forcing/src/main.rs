use anyhow::{Context, Result, bail};
use metal::*;
use objc::rc::autoreleasepool;
use serde::Serialize;
use sha1::{Digest, Sha1};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

// Embed the Metal source directly into the binary so the lesson can compile and run
// without needing to read shader files at runtime.
const SHADER_SOURCE: &str = include_str!("sha1_brute_force.metal");

// Reasonable CLI defaults for a demo/lesson run. These are intentionally conservative
// so the program starts quickly on most machines.
const DEFAULT_CHARSET: &str = "lowernum";
const DEFAULT_MIN_LEN: u32 = 1;
const DEFAULT_MAX_LEN: u32 = 6;
const DEFAULT_MODE: &str = "first";
const DEFAULT_VALIDATION: &str = "spot";
const DEFAULT_THREADS_PER_GROUP: u32 = 256;
const DEFAULT_CANDIDATES_PER_THREAD: u32 = 8;
const DEFAULT_PROGRESS_MS: u64 = 500;
const DEFAULT_MAX_MATCHES: u32 = 1024;

// SHA-1 processes data in 64-byte blocks. In this lesson we only support a single block,
// which means: message bytes + 0x80 + 8-byte length field must fit in 64 bytes.
// Therefore the maximum message length is 55 bytes.
const MAX_ONE_BLOCK_LEN: u32 = 55;

// Which alphabet to brute-force over. The GPU kernel gets a compact integer ID,
// but the CLI and reports use these readable names.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Charset {
    Lower,
    LowerNum,
    Printable,
}

impl Charset {
    // Parse the CLI string into the internal enum.
    fn parse(v: &str) -> Result<Self> {
        Ok(match v {
            "lower" => Self::Lower,
            "lowernum" => Self::LowerNum,
            "printable" => Self::Printable,
            _ => bail!("invalid --charset '{v}' (expected lower|lowernum|printable)"),
        })
    }

    // Human-readable value for logs / JSON output.
    fn as_str(self) -> &'static str {
        match self {
            Self::Lower => "lower",
            Self::LowerNum => "lowernum",
            Self::Printable => "printable",
        }
    }

    // Number of symbols in the alphabet. This is the radix/base used to map a numeric
    // candidate index to a concrete string.
    fn radix(self) -> u32 {
        match self {
            Self::Lower => 26,
            Self::LowerNum => 36,
            Self::Printable => 95,
        }
    }

    // Convert one base-radix digit into the actual byte that should appear in the
    // candidate string.
    fn digit_to_byte(self, d: u32) -> u8 {
        match self {
            Self::Lower => b'a' + (d as u8),
            Self::LowerNum => {
                if d < 26 {
                    b'a' + (d as u8)
                } else {
                    b'0' + ((d - 26) as u8)
                }
            }
            Self::Printable => (0x20u8).wrapping_add(d as u8),
        }
    }

    // Compact ID sent to the Metal kernel. The kernel branches on this once per tested
    // candidate to choose the mapping routine.
    fn alphabet_id(self) -> u32 {
        match self {
            Self::Lower => 0,
            Self::LowerNum => 1,
            Self::Printable => 2,
        }
    }
}

// Search strategy: stop on the first match or collect all matches (up to a buffer cap).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MatchMode {
    First,
    All,
}

impl MatchMode {
    fn parse(v: &str) -> Result<Self> {
        Ok(match v {
            "first" => Self::First,
            "all" => Self::All,
            _ => bail!("invalid --mode '{v}' (expected first|all)"),
        })
    }

    // Integer encoding consumed by the kernel.
    fn as_u32(self) -> u32 {
        match self {
            Self::First => 0,
            Self::All => 1,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::First => "first",
            Self::All => "all",
        }
    }
}

// How aggressively to verify GPU results on the CPU.
// - GpuOnly: trust the kernel entirely
// - Spot: validate reported hits and sample a subset of the search space
// - Full: brute-force the same space on CPU for correctness checking (very expensive)
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ValidationMode {
    GpuOnly,
    Spot,
    Full,
}

impl ValidationMode {
    fn parse(v: &str) -> Result<Self> {
        Ok(match v {
            "gpu-only" => Self::GpuOnly,
            "spot" => Self::Spot,
            "full" => Self::Full,
            _ => bail!("invalid --validation '{v}' (expected gpu-only|spot|full)"),
        })
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::GpuOnly => "gpu-only",
            Self::Spot => "spot",
            Self::Full => "full",
        }
    }
}

// Parsed CLI configuration used by the rest of the program.
// Keeping this in one struct makes it easier to pass options around and serialize later.
#[derive(Clone, Debug)]
struct CliConfig {
    hash_hex: String,
    charset: Charset,
    min_len: u32,
    max_len: u32,
    mode: MatchMode,
    validation: ValidationMode,
    threads_per_group: u32,
    candidates_per_thread: u32,
    progress_ms: u64,
    max_matches: u32,
    json: Option<PathBuf>,
    verbose: bool,
}

// Host <-> GPU parameter block. `#[repr(C)]` ensures the field layout matches what the
// Metal shader expects in its `KernelParams` struct.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct KernelParams {
    // Candidate string length for the current dispatch.
    len: u32,
    // Alphabet size (26/36/95). Mostly informational for this kernel version.
    radix: u32,
    // Total number of candidates for this length.
    search_space: u64,
    // How many candidate indices each thread tests inside its inner loop.
    candidates_per_thread: u32,
    // Match mode encoded as 0/1 to avoid sending enums across the FFI boundary.
    mode: u32,
    // Capacity of the `match_indices` output buffer in `all` mode.
    max_matches: u32,
    // Which alphabet mapping the kernel should use.
    alphabet_id: u32,
    // Target SHA-1 digest split into five big-endian 32-bit words.
    target_a: u32,
    target_b: u32,
    target_c: u32,
    target_d: u32,
    target_e: u32,
}

// Per-length metrics emitted to terminal and optionally written to JSON.
#[derive(Clone, Debug, Serialize)]
struct LengthReport {
    len: u32,
    radix: u32,
    candidates: u64,
    gpu_ms: f64,
    mh_s: f64,
    found: bool,
    found_count: u32,
}

// Whole-run report used for JSON output and easy post-analysis.
#[derive(Clone, Debug, Serialize)]
struct RunReport {
    hash: String,
    charset: String,
    mode: String,
    validation: String,
    min_len: u32,
    max_len: u32,
    threads_per_group: u32,
    candidates_per_thread: u32,
    total_candidates_tested: u64,
    wall_ms: f64,
    overall_mh_s: f64,
    matches: Vec<String>,
    lengths: Vec<LengthReport>,
    validation_checked: u64,
    validation_mismatches: u64,
}

// Static usage text shown by `--help` and reused in some parse errors.
fn usage() -> &'static str {
    r#"Lesson 13: GPU SHA1 brute forcing (Metal)

USAGE:
  cargo run --release -p sha1-brute-forcing -- --hash <40hex> [options]

REQUIRED:
  --hash <40hex>                 Target SHA1 in 40 hex characters

OPTIONS:
  --charset lower|lowernum|printable     (default: lowernum)
  --min-len <u32>                        (default: 1)
  --max-len <u32>                        (default: 6)
  --mode first|all                       (default: first)
  --validation gpu-only|spot|full        (default: spot)
  --threads-per-group <u32>              (default: 256)
  --candidates-per-thread <u32>          (default: 8)
  --progress-ms <u64>                    (default: 500)
  --max-matches <u32>                    (default: 1024, only for mode=all)
  --json <path>                          Write JSON report
  --verbose                              Per-length breakdown output

NOTES:
  - This lesson supports candidate lengths up to 55 bytes (SHA1 single-block).
  - Runtime requires valid Metal kernels in src/sha1_brute_force.metal.
"#
}

// Minimal hand-rolled CLI parser. This keeps dependencies light and makes the lesson's
// argument handling explicit, but still validates every input before execution.
fn parse_cli() -> Result<CliConfig> {
    // Start from defaults, then override as flags are encountered.
    let mut hash_hex: Option<String> = None;
    let mut charset = Charset::parse(DEFAULT_CHARSET)?;
    let mut min_len = DEFAULT_MIN_LEN;
    let mut max_len = DEFAULT_MAX_LEN;
    let mut mode = MatchMode::parse(DEFAULT_MODE)?;
    let mut validation = ValidationMode::parse(DEFAULT_VALIDATION)?;
    let mut threads_per_group = DEFAULT_THREADS_PER_GROUP;
    let mut candidates_per_thread = DEFAULT_CANDIDATES_PER_THREAD;
    let mut progress_ms = DEFAULT_PROGRESS_MS;
    let mut max_matches = DEFAULT_MAX_MATCHES;
    let mut json: Option<PathBuf> = None;
    let mut verbose = false;

    // Materialize argv so we can index into it manually (`--flag value` style parsing).
    // This is intentionally explicit to show how CLI parsing works under the hood.
    let args: Vec<String> = env::args().collect();
    // Skip argv[0] (binary path).
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                print!("{}", usage());
                std::process::exit(0);
            }
            "--hash" => {
                i += 1;
                // `args.get(i)` keeps this safe even if the user forgets the value:
                // we convert the `None` case into a readable error via `Context`.
                hash_hex = Some(args.get(i).context("--hash requires a value")?.to_string());
            }
            "--charset" => {
                i += 1;
                charset = Charset::parse(args.get(i).context("--charset requires a value")?)?;
            }
            "--min-len" => {
                i += 1;
                // `.parse()?` infers the destination type (`u32`) from the variable.
                min_len = args.get(i).context("--min-len requires a value")?.parse()?;
            }
            "--max-len" => {
                i += 1;
                max_len = args.get(i).context("--max-len requires a value")?.parse()?;
            }
            "--mode" => {
                i += 1;
                mode = MatchMode::parse(args.get(i).context("--mode requires a value")?)?;
            }
            "--validation" => {
                i += 1;
                validation =
                    ValidationMode::parse(args.get(i).context("--validation requires a value")?)?;
            }
            "--threads-per-group" => {
                i += 1;
                threads_per_group = args
                    .get(i)
                    .context("--threads-per-group requires a value")?
                    .parse()?;
            }
            "--candidates-per-thread" => {
                i += 1;
                candidates_per_thread = args
                    .get(i)
                    .context("--candidates-per-thread requires a value")?
                    .parse()?;
            }
            "--progress-ms" => {
                i += 1;
                progress_ms = args
                    .get(i)
                    .context("--progress-ms requires a value")?
                    .parse()?;
            }
            "--max-matches" => {
                i += 1;
                max_matches = args
                    .get(i)
                    .context("--max-matches requires a value")?
                    .parse()?;
            }
            "--json" => {
                i += 1;
                // Store the path now; writing happens at the end once the run report exists.
                json = Some(PathBuf::from(
                    args.get(i).context("--json requires a path")?,
                ));
            }
            "--verbose" => {
                verbose = true;
            }
            other => bail!("unknown arg: {other}\n\n{}", usage()),
        }
        // Advance to the next token after processing this flag (and its value if any).
        i += 1;
    }

    // Require and validate the target hash before touching the GPU.
    let hash_hex = hash_hex.context("--hash is required")?;
    // Parse once here to fail fast on malformed input; we parse again in `run()` to get the
    // actual bytes. (Could be returned from `parse_cli`, but keeping CLI config printable is nice.)
    parse_sha1_hex(&hash_hex)?;

    // Semantic validation for numeric options.
    if min_len == 0 {
        bail!("--min-len must be >= 1");
    }
    if min_len > max_len {
        bail!("--min-len must be <= --max-len");
    }
    if max_len > MAX_ONE_BLOCK_LEN {
        bail!(
            "--max-len {max_len} exceeds {MAX_ONE_BLOCK_LEN} (SHA1 single-block limit for this lesson)"
        );
    }
    if threads_per_group == 0 {
        bail!("--threads-per-group must be > 0");
    }
    if candidates_per_thread == 0 {
        bail!("--candidates-per-thread must be > 0");
    }
    if progress_ms == 0 {
        bail!("--progress-ms must be > 0");
    }

    Ok(CliConfig {
        hash_hex,
        charset,
        min_len,
        max_len,
        mode,
        validation,
        threads_per_group,
        candidates_per_thread,
        progress_ms,
        max_matches,
        json,
        verbose,
    })
}

// Parse a 40-character SHA-1 hex digest into raw bytes.
fn parse_sha1_hex(hex: &str) -> Result<[u8; 20]> {
    // Accept accidental whitespace around the digest (copy/paste friendly).
    let h = hex.trim();
    if h.len() != 40 {
        bail!("--hash must be 40 hex chars (got {})", h.len());
    }

    let mut out = [0u8; 20];
    for i in 0..20 {
        // Each SHA-1 byte is represented by exactly two hex characters.
        // Example: "0f" -> 15.
        out[i] = u8::from_str_radix(&h[i * 2..i * 2 + 2], 16)
            .with_context(|| format!("invalid hex at byte {i}"))?;
    }
    Ok(out)
}

// The GPU kernel compares SHA-1 state words (`a..e`) as 32-bit values, so we convert the
// byte digest into the same big-endian word layout used inside SHA-1.
fn digest_bytes_to_words_be(d: [u8; 20]) -> (u32, u32, u32, u32, u32) {
    // SHA-1 digest display strings are byte-oriented, but the kernel compares the final
    // state variables A..E as 32-bit words. The SHA-1 spec defines those words in big-endian.
    let a = u32::from_be_bytes([d[0], d[1], d[2], d[3]]);
    let b = u32::from_be_bytes([d[4], d[5], d[6], d[7]]);
    let c = u32::from_be_bytes([d[8], d[9], d[10], d[11]]);
    let d2 = u32::from_be_bytes([d[12], d[13], d[14], d[15]]);
    let e = u32::from_be_bytes([d[16], d[17], d[18], d[19]]);
    (a, b, c, d2, e)
}

// Checked integer exponentiation to compute search-space size (`radix^len`) safely.
// We fail fast if the value would overflow `u64`.
fn pow_u64(base: u64, exp: u32) -> Result<u64> {
    let mut acc = 1u64;
    for _ in 0..exp {
        // Checked multiplication avoids silent wraparound, which would be catastrophic here:
        // it would make us dispatch the wrong amount of work and misreport throughput.
        acc = acc.checked_mul(base).context("candidate space overflow")?;
    }
    Ok(acc)
}

// Convert a numeric candidate index to its corresponding string in the selected alphabet.
// This mirrors the GPU's base-radix mapping and is used for validation/reporting.
fn index_to_candidate(idx: u64, len: u32, charset: Charset) -> String {
    let radix = charset.radix() as u64;
    // We destructively divide `x` by the radix to peel off digits one at a time.
    // This is exactly how you convert an integer to base-N representation.
    let mut x = idx;
    let mut bytes = vec![0u8; len as usize];

    // Least-significant radix digit becomes the first byte, matching the GPU mapping code.
    for i in 0..len {
        // `d` is the current base-radix digit in [0, radix).
        let d = (x % radix) as u32;
        // Integer division shifts us to the next digit for the next iteration.
        x /= radix;
        bytes[i as usize] = charset.digit_to_byte(d);
    }

    String::from_utf8_lossy(&bytes).to_string()
}

// CPU SHA-1 helper used only for validation paths.
fn sha1_bytes(input: &[u8]) -> [u8; 20] {
    let mut hasher = Sha1::new();
    // `sha1` crate supports streaming updates; we only need a single update call here.
    hasher.update(input);
    let digest = hasher.finalize();
    let mut out = [0u8; 20];
    // Convert the crate's generic output buffer into a fixed-size array for ergonomic comparisons.
    out.copy_from_slice(&digest);
    out
}

// Minimal container for the two long-lived Metal objects we need for dispatching compute:
// a command queue and the compiled compute pipeline state.
struct GpuPipelines {
    queue: CommandQueue,
    pipeline: ComputePipelineState,
}

// Compile the embedded Metal source, fetch the named kernel function, and build a compute
// pipeline object. Doing this once amortizes setup cost across all lengths.
fn build_pipelines(device: &Device, kernel_name: &str) -> Result<GpuPipelines> {
    // Metal compile options (language version, fast-math, preprocessor macros, etc.) can be
    // configured here; defaults are fine for this lesson.
    let opts = CompileOptions::new();

    // Compile the in-memory shader source into a Metal library object.
    let library = device
        .new_library_with_source(SHADER_SOURCE, &opts)
        .map_err(|e| anyhow::anyhow!("failed to compile Metal source: {e}"))?;

    // Fetch the named `kernel void ...` entry point from the compiled library.
    let func = library
        .get_function(kernel_name, None)
        .map_err(|e| anyhow::anyhow!("failed to get function {kernel_name}: {e}"))?;

    // Bake the function into a compute pipeline state. Metal uses this object to know
    // resource bindings, machine code, and scheduling properties (including threadgroup limits).
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| anyhow::anyhow!("failed to create compute pipeline: {e}"))?;

    // Command queue is the submission channel from CPU -> GPU.
    let queue = device.new_command_queue();
    Ok(GpuPipelines { queue, pipeline })
}

// Main application logic (kept separate from `main()` so errors can be propagated via
// `Result` and formatted in one place).
fn run() -> Result<()> {
    // 1) Parse and validate host-side configuration.
    let cli = parse_cli()?;

    // Parse once and keep both forms:
    // - raw bytes for CPU validation
    // - packed u32 words for GPU parameter upload
    let digest_bytes = parse_sha1_hex(&cli.hash_hex)?;
    let (ta, tb, tc, td, te) = digest_bytes_to_words_be(digest_bytes);

    // Metal / Objective-C APIs rely heavily on autoreleased objects. Wrapping the run in an
    // autorelease pool prevents temporary ObjC allocations from accumulating too long.
    autoreleasepool(|| -> Result<()> {
        // Pick the default Metal device (typically the integrated Apple GPU on laptops).
        let device = Device::system_default().context("no Metal device available")?;
        println!("GPU: {}", device.name());

        // Keep the kernel name in one place so it is easy to swap variants later.
        let kernel_name = "sha1_brute_force";
        let pipelines = build_pipelines(&device, kernel_name)?;
        println!("Kernel: {}", kernel_name);
        println!(
            "Pipeline limits: thread_execution_width={} max_threads_per_threadgroup={}",
            pipelines.pipeline.thread_execution_width(),
            pipelines.pipeline.max_total_threads_per_threadgroup()
        );

        // Output / control buffers (shared memory so CPU and GPU can both read/write them).
        // This program reuses the same buffers across all candidate lengths.
        // `found_flag`: 0 until any thread wins in `first` mode, then atomically set to 1.
        let found_flag = device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        // `found_index`: winning candidate index in `first` mode.
        let found_index = device.new_buffer(
            std::mem::size_of::<u64>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        // `match_count`: atomic counter incremented by the kernel in `all` mode.
        let match_count = device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        // `match_indices`: output ring/array (not a true ring here) of discovered candidate indices.
        let match_indices = device.new_buffer(
            (std::mem::size_of::<u64>() as u64) * (cli.max_matches as u64),
            MTLResourceOptions::StorageModeShared,
        );
        // `params_buf`: per-dispatch constants (uploaded anew for each length).
        let params_buf = device.new_buffer(
            std::mem::size_of::<KernelParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Helper closures to reset/read shared control state.
        // These use `unsafe` because `metal-rs` exposes raw buffer pointers.
        let reset_controls = || unsafe {
            // These writes happen before command buffer submission, so no explicit synchronization
            // is needed beyond the natural CPU->GPU ordering provided by command submission.
            *(found_flag.contents() as *mut u32) = 0;
            *(found_index.contents() as *mut u64) = 0;
            *(match_count.contents() as *mut u32) = 0;
        };

        let read_found_flag = || unsafe { *(found_flag.contents() as *const u32) };
        let read_found_index = || unsafe { *(found_index.contents() as *const u64) };
        let read_match_count = || unsafe { *(match_count.contents() as *const u32) };

        let read_match_indices = |count: u32| -> Vec<u64> {
            let mut out = Vec::with_capacity(count as usize);
            unsafe {
                let base = match_indices.contents() as *const u64;
                for i in 0..count {
                    // Pointer arithmetic walks the shared buffer as a `u64[]`.
                    out.push(*base.add(i as usize));
                }
            }
            out
        };

        // Run-wide accounting values used for final summary/JSON output.
        let radix = cli.charset.radix() as u64;
        let wall_start = Instant::now();
        let mut total_candidates_tested: u64 = 0;
        let mut all_matches: Vec<String> = Vec::new();
        let mut length_reports: Vec<LengthReport> = Vec::new();
        let mut validation_checked: u64 = 0;
        let mut validation_mismatches: u64 = 0;

        // Progress logging can be triggered either by `--verbose` (every length) or by time.
        let progress_interval = Duration::from_millis(cli.progress_ms);
        let mut next_progress = Instant::now() + progress_interval;

        // Outer loop: dispatch one kernel run per candidate length.
        // The label allows early exit in `first` mode once a match is found.
        'lengths: for len in cli.min_len..=cli.max_len {
            // Search space for fixed length `len` is radix^len.
            let candidates = pow_u64(radix, len)?;
            total_candidates_tested = total_candidates_tested
                .checked_add(candidates)
                .context("total candidate count overflow")?;

            // Clear result buffers before each dispatch.
            reset_controls();

            // Fill the kernel parameter block for this length and target hash.
            let params = KernelParams {
                len,
                radix: cli.charset.radix(),
                search_space: candidates,
                candidates_per_thread: cli.candidates_per_thread,
                mode: cli.mode.as_u32(),
                max_matches: cli.max_matches,
                alphabet_id: cli.charset.alphabet_id(),
                target_a: ta,
                target_b: tb,
                target_c: tc,
                target_d: td,
                target_e: te,
            };
            unsafe {
                // Copy the POD struct directly into shared memory. `KernelParams` is `repr(C)` and
                // contains only integer fields, so bytewise copy is appropriate.
                *(params_buf.contents() as *mut KernelParams) = params;
            }

            // Threadgroup size (aka threads per group) chosen by the CLI/user.
            let threads_per_tg = MTLSize {
                width: cli.threads_per_group as u64,
                height: 1,
                depth: 1,
            };

            // Hardware/pipeline hard limit check. This is kernel- and device-dependent.
            let max_tg = pipelines.pipeline.max_total_threads_per_threadgroup() as u64;
            if threads_per_tg.width > max_tg {
                bail!(
                    "--threads-per-group {} exceeds pipeline max {}",
                    threads_per_tg.width,
                    max_tg
                );
            }

            // Estimate how many logical threads we need to cover the search space given that
            // each thread checks `candidates_per_thread` candidates per inner-loop slice.
            let work_per_thread = cli.candidates_per_thread as u64;
            // Ceiling division: how many logical threads are required if each thread is
            // responsible for `work_per_thread` candidate indices per pass.
            let needed_threads = (candidates + work_per_thread - 1) / work_per_thread;

            // Clamp total grid size to a practical range:
            // - At least 4 threadgroups so small lengths still have some parallelism.
            // - At most 65535 threadgroups along X to stay within the lesson's 1D dispatch plan.
            let min_threads = threads_per_tg.width * 4;
            let max_threads = threads_per_tg.width * 65535;
            let threads = needed_threads.clamp(min_threads, max_threads);

            let threads_per_grid = MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            };

            // Encode and execute the compute dispatch.
            let gpu_start = Instant::now();
            // One command buffer + one compute encoder per length keeps the demo simple.
            let cmd_buf = pipelines.queue.new_command_buffer();
            let enc = cmd_buf.new_compute_command_encoder();

            enc.set_compute_pipeline_state(&pipelines.pipeline);
            // Bindings must match the `[[ buffer(n) ]]` indices in the Metal kernel exactly.
            enc.set_buffer(0, Some(&params_buf), 0);
            enc.set_buffer(1, Some(&found_flag), 0);
            enc.set_buffer(2, Some(&found_index), 0);
            enc.set_buffer(3, Some(&match_count), 0);
            enc.set_buffer(4, Some(&match_indices), 0);

            // `dispatch_threads` lets us specify an exact total thread count while Metal rounds
            // appropriately using the provided threadgroup shape.
            enc.dispatch_threads(threads_per_grid, threads_per_tg);
            enc.end_encoding();

            cmd_buf.commit();
            // This lesson waits synchronously for each length to complete, which keeps control
            // flow simple and makes timing per length straightforward.
            cmd_buf.wait_until_completed();
            let gpu_ms = gpu_start.elapsed().as_secs_f64() * 1000.0;

            // Pull results back from shared buffers.
            let mut found = false;
            let mut found_count = 0u32;
            let mut matches_for_len: Vec<u64> = Vec::new();

            match cli.mode {
                MatchMode::First => {
                    // Kernel writes a single winner (if any) and exits threads early.
                    if read_found_flag() != 0 {
                        found = true;
                        found_count = 1;
                        matches_for_len.push(read_found_index());
                    }
                }
                MatchMode::All => {
                    // `match_count` may exceed `max_matches`; the kernel still counts all hits
                    // even if it stops storing indices once the output buffer is full.
                    let c = read_match_count();
                    found = c > 0;
                    found_count = c;
                    // Cap reads to the output buffer capacity; the kernel may have observed more
                    // matches than we chose to store.
                    matches_for_len = read_match_indices(c.min(cli.max_matches));
                }
            }

            // Optional CPU-side correctness checking.
            if cli.validation != ValidationMode::GpuOnly {
                // Always verify any matches reported by the GPU.
                for &idx in &matches_for_len {
                    // Convert index -> candidate using the same mapping convention as GPU.
                    let cand = index_to_candidate(idx, len, cli.charset);
                    let d = sha1_bytes(cand.as_bytes());
                    validation_checked += 1;
                    if d != digest_bytes {
                        validation_mismatches += 1;
                    }
                }

                match cli.validation {
                    ValidationMode::GpuOnly => {}
                    ValidationMode::Spot => {
                        // Sample up to 256 positions spread across the length's search space.
                        // This catches obvious mapping/hash bugs without full CPU brute force.
                        let sample = 256u64.min(candidates);
                        if sample > 0 {
                            for s in 0..sample {
                                // Even-ish spread across the search space (not perfectly uniform
                                // for all sizes, but good enough to catch systemic bugs).
                                let idx = (s * (candidates / sample.max(1))) % candidates;
                                let cand = index_to_candidate(idx, len, cli.charset);
                                let d = sha1_bytes(cand.as_bytes());
                                validation_checked += 1;
                                if d == digest_bytes && !matches_for_len.contains(&idx) {
                                    validation_mismatches += 1;
                                }
                            }
                        }
                    }
                    ValidationMode::Full => {
                        // Exhaustive CPU verification of the entire search space for this length.
                        // Very slow, but useful when debugging correctness issues.
                        for idx in 0..candidates {
                            let cand = index_to_candidate(idx, len, cli.charset);
                            let d = sha1_bytes(cand.as_bytes());
                            validation_checked += 1;
                            if d == digest_bytes {
                                let gpu_reported = matches_for_len.contains(&idx)
                                    || (cli.mode == MatchMode::First
                                        && found
                                        // In first mode the kernel stores only one index, so we
                                        // also compare against `found_index`.
                                        && read_found_index() == idx);
                                if !gpu_reported {
                                    validation_mismatches += 1;
                                }
                            }
                        }
                    }
                }
            }

            // Throughput metric for this dispatch only.
            let mh_s = (candidates as f64) / (gpu_ms / 1000.0) / 1_000_000.0;
            length_reports.push(LengthReport {
                len,
                radix: cli.charset.radix(),
                candidates,
                gpu_ms,
                mh_s,
                found,
                found_count,
            });

            // Materialize candidate strings for any found indices so final output is readable.
            for idx in matches_for_len {
                // The kernel reports numeric indices to keep GPU output compact; turn them back
                // into strings only for human-facing output and JSON.
                all_matches.push(index_to_candidate(idx, len, cli.charset));
            }

            // Progress output is either per length (`--verbose`) or time-based throttled.
            if cli.verbose || Instant::now() >= next_progress {
                println!(
                    "len={len} radix={} candidates={} gpu_ms={:.2} mh/s={:.2} found={} hits={}",
                    cli.charset.radix(),
                    candidates,
                    gpu_ms,
                    mh_s,
                    found,
                    found_count
                );
                next_progress = Instant::now() + progress_interval;
            }

            // In first-match mode, stop searching longer lengths once we found something.
            if cli.mode == MatchMode::First && found {
                break 'lengths;
            }
        }

        // Final wall-clock summary includes host overhead, validation, and reporting work.
        let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
        let overall_mh_s = (total_candidates_tested as f64) / (wall_ms / 1000.0) / 1_000_000.0;

        println!("\nTarget SHA1: {}", cli.hash_hex.to_lowercase());
        println!("Mode: {}", cli.mode.as_str());
        println!("Charset: {}", cli.charset.as_str());
        println!("Lengths: {}..={}", cli.min_len, cli.max_len);
        println!("Total candidates tested: {}", total_candidates_tested);
        println!("Wall time (ms): {:.2}", wall_ms);
        println!("Overall MH/s: {:.2}", overall_mh_s);
        println!("Matches found: {}", all_matches.len());

        if !all_matches.is_empty() {
            println!("Matches:");
            for m in &all_matches {
                println!("  {m}");
            }
        }

        if cli.validation != ValidationMode::GpuOnly {
            println!(
                "Validation: checked={} mismatches={}",
                validation_checked, validation_mismatches
            );
        }

        // Optional machine-readable report for plotting / regression tracking.
        if let Some(path) = &cli.json {
            // Snapshot all run state into a serializable struct and write pretty JSON so it's easy
            // to diff benchmark runs by hand.
            let report = RunReport {
                hash: cli.hash_hex.to_lowercase(),
                charset: cli.charset.as_str().to_string(),
                mode: cli.mode.as_str().to_string(),
                validation: cli.validation.as_str().to_string(),
                min_len: cli.min_len,
                max_len: cli.max_len,
                threads_per_group: cli.threads_per_group,
                candidates_per_thread: cli.candidates_per_thread,
                total_candidates_tested,
                wall_ms,
                overall_mh_s,
                matches: all_matches,
                lengths: length_reports,
                validation_checked,
                validation_mismatches,
            };
            fs::write(path, serde_json::to_string_pretty(&report)?)?;
            println!("Wrote JSON report: {}", path.display());
        }

        Ok(())
    })
}

fn main() {
    // Keep `main` tiny: run the real program and convert any error into a clean
    // non-zero exit plus a readable message.
    if let Err(e) = run() {
        // `anyhow` already carries context chains, so a single print is usually enough.
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
