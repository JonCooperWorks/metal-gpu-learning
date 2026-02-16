// =============================================================================
// LESSON 11: GPU MD5 BRUTE FORCING (MAX THROUGHPUT, TEACHING-FIRST)
// =============================================================================
//
// IMPORTANT ETHICS / LEGAL NOTE
// -----------------------------
// This lesson is for *authorized* security testing, password recovery on your
// own data, and educational benchmarking.
//
// If you do not have permission to crack a hash, do not run this on it.
//
//
// What this lesson teaches
// ------------------------
// 1) How to structure a throughput-first GPU brute-force search:
//      - map an integer search space onto candidate strings
//      - run an in-kernel hash
//      - compare digest and record matches
// 2) How to tune throughput knobs:
//      - threads per threadgroup
//      - candidates per thread
//      - length ramp strategy 1..N
// 3) How to validate correctness without destroying performance:
//      - gpu-only
//      - spot-check CPU recomputation
//      - full CPU mirror (slow but confidence)
//
// This project follows the tutorial style of earlier lessons in this repo:
// lots of comments, explicit buffer contracts, and no hidden magic.
// =============================================================================

use anyhow::{bail, Context, Result};
use metal::*;
use objc::rc::autoreleasepool;
use serde::Serialize;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const SHADER_SOURCE: &str = include_str!("lesson11.metal");

// Default knobs: chosen to be fast on Apple GPUs without being too extreme.
const DEFAULT_CHARSET: &str = "lowernum";
const DEFAULT_MIN_LEN: u32 = 1;
const DEFAULT_MAX_LEN: u32 = 6;
const DEFAULT_MODE: &str = "first";
const DEFAULT_VALIDATION: &str = "spot";
const DEFAULT_THREADS_PER_GROUP: u32 = 256;
const DEFAULT_CANDIDATES_PER_THREAD: u32 = 8;
const DEFAULT_PROGRESS_MS: u64 = 500;
const DEFAULT_MAX_MATCHES: u32 = 1024;

// MD5 one-block limit for our GPU kernel implementation.
// Messages longer than 55 bytes require multiple blocks (not covered here).
const MAX_ONE_BLOCK_LEN: u32 = 55;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Charset {
    Lower,
    LowerNum,
    Printable,
}

impl Charset {
    fn parse(v: &str) -> Result<Self> {
        Ok(match v {
            "lower" => Self::Lower,
            "lowernum" => Self::LowerNum,
            "printable" => Self::Printable,
            _ => bail!("invalid --charset '{v}' (expected lower|lowernum|printable)"),
        })
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Lower => "lower",
            Self::LowerNum => "lowernum",
            Self::Printable => "printable",
        }
    }

    fn radix(self) -> u32 {
        match self {
            Self::Lower => 26,
            Self::LowerNum => 36,
            Self::Printable => 95,
        }
    }

    // Convert a base-radix digit into an output byte.
    // This must exactly mirror the mapping inside the Metal kernel.
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

    fn kernel_name(self) -> &'static str {
        match self {
            Self::Lower => "md5_bruteforce_lower",
            Self::LowerNum => "md5_bruteforce_lowernum",
            Self::Printable => "md5_bruteforce_printable",
        }
    }
}

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

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct KernelParams {
    len: u32,
    radix: u32,
    total: u64,
    candidates_per_thread: u32,
    mode: u32,
    max_matches: u32,
    target_a: u32,
    target_b: u32,
    target_c: u32,
    target_d: u32,
}

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

// -----------------------------------------------------------------------------
// CLI parsing (minimal and explicit)
// -----------------------------------------------------------------------------

fn usage() -> &'static str {
    r#"Lesson 11: GPU MD5 brute forcing (Metal)

USAGE:
  cargo run --release -p md5-brute-forcing -- --hash <32hex> [options]

REQUIRED:
  --hash <32hex>                 Target MD5 in 32 hex characters

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
  - This lesson supports candidate lengths up to 55 bytes (MD5 single-block).
  - Use only on hashes you are authorized to test.
"#
}

fn parse_cli() -> Result<CliConfig> {
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

    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "--help" | "-h" => {
                print!("{}", usage());
                std::process::exit(0);
            }
            "--hash" => {
                i += 1;
                hash_hex = Some(
                    args.get(i)
                        .context("--hash requires a value")?
                        .to_string(),
                );
            }
            "--charset" => {
                i += 1;
                charset = Charset::parse(args.get(i).context("--charset requires a value")?)?;
            }
            "--min-len" => {
                i += 1;
                min_len = args
                    .get(i)
                    .context("--min-len requires a value")?
                    .parse()?;
            }
            "--max-len" => {
                i += 1;
                max_len = args
                    .get(i)
                    .context("--max-len requires a value")?
                    .parse()?;
            }
            "--mode" => {
                i += 1;
                mode = MatchMode::parse(args.get(i).context("--mode requires a value")?)?;
            }
            "--validation" => {
                i += 1;
                validation = ValidationMode::parse(
                    args.get(i)
                        .context("--validation requires a value")?
                        .as_str(),
                )?;
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
                json = Some(PathBuf::from(
                    args.get(i).context("--json requires a path")?,
                ));
            }
            "--verbose" => verbose = true,
            other => bail!("unknown arg: {other}\n\n{}", usage()),
        }
        i += 1;
    }

    let hash_hex = hash_hex.context("--hash is required")?;

    // Basic validation of ranges.
    if min_len == 0 {
        bail!("--min-len must be >= 1");
    }
    if min_len > max_len {
        bail!("--min-len must be <= --max-len");
    }
    if max_len > MAX_ONE_BLOCK_LEN {
        bail!(
            "--max-len {max_len} exceeds {MAX_ONE_BLOCK_LEN} (MD5 single-block limit for this lesson)"
        );
    }
    if threads_per_group == 0 {
        bail!("--threads-per-group must be > 0");
    }
    if candidates_per_thread == 0 {
        bail!("--candidates-per-thread must be > 0");
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

// -----------------------------------------------------------------------------
// Hash parsing and endian correctness
// -----------------------------------------------------------------------------

// Convert 32 hex chars into 16 bytes.
fn parse_md5_hex(hex: &str) -> Result<[u8; 16]> {
    let h = hex.trim();
    if h.len() != 32 {
        bail!("--hash must be 32 hex chars (got {})", h.len());
    }

    let mut out = [0u8; 16];
    for i in 0..16 {
        let b = u8::from_str_radix(&h[i * 2..i * 2 + 2], 16)
            .with_context(|| format!("invalid hex at byte {i}"))?;
        out[i] = b;
    }
    Ok(out)
}

// MD5 digest is typically printed as 16 bytes (little-endian by word in RFC terms).
// Our Metal kernel computes MD5 words A,B,C,D as 32-bit little-endian words.
// So we parse the 16 digest bytes and interpret each 4-byte chunk as LE u32.
fn digest_bytes_to_words_le(d: [u8; 16]) -> (u32, u32, u32, u32) {
    let a = u32::from_le_bytes([d[0], d[1], d[2], d[3]]);
    let b = u32::from_le_bytes([d[4], d[5], d[6], d[7]]);
    let c = u32::from_le_bytes([d[8], d[9], d[10], d[11]]);
    let d = u32::from_le_bytes([d[12], d[13], d[14], d[15]]);
    (a, b, c, d)
}

// -----------------------------------------------------------------------------
// Candidate enumeration helpers (CPU side)
// -----------------------------------------------------------------------------

fn pow_u64(base: u64, exp: u32) -> Result<u64> {
    // Checked exponentiation: we want explicit error rather than silent overflow.
    let mut acc: u64 = 1;
    for _ in 0..exp {
        acc = acc.checked_mul(base).context("candidate space overflow")?;
    }
    Ok(acc)
}

fn index_to_candidate(idx: u64, len: u32, charset: Charset) -> String {
    let radix = charset.radix() as u64;
    let mut x = idx;
    let mut bytes = vec![0u8; len as usize];

    for i in 0..len {
        let d = (x % radix) as u32;
        x /= radix;
        bytes[i as usize] = charset.digit_to_byte(d);
    }

    // Candidates are bytes; for our charsets they are valid UTF-8.
    String::from_utf8_lossy(&bytes).to_string()
}

fn md5_bytes(input: &[u8]) -> [u8; 16] {
    // `md5` crate returns a 16-byte digest via `compute`.
    // `md5::Digest` is a small wrapper type around `[u8; 16]`.
    let digest = md5::compute(input);
    let mut out = [0u8; 16];
    out.copy_from_slice(&digest.0);
    out
}

// -----------------------------------------------------------------------------
// Metal compilation boilerplate
// -----------------------------------------------------------------------------

struct GpuPipelines {
    queue: CommandQueue,
    pipeline: ComputePipelineState,
}

fn build_pipelines(device: &Device, kernel_name: &str) -> Result<GpuPipelines> {
    let opts = CompileOptions::new();

    // Compile Metal source at runtime.
    // This is slower than precompiled metallib, but it keeps the lesson self-contained.
    let library = device
        .new_library_with_source(SHADER_SOURCE, &opts)
        .map_err(|e| anyhow::anyhow!("failed to compile Metal: {e}"))?;

    let func = library
        .get_function(kernel_name, None)
        .map_err(|e| anyhow::anyhow!("failed to get function {kernel_name}: {e}"))?;

    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| anyhow::anyhow!("failed to create compute pipeline: {e}"))?;

    let queue = device.new_command_queue();

    Ok(GpuPipelines { queue, pipeline })
}

// -----------------------------------------------------------------------------
// The actual brute force search loop
// -----------------------------------------------------------------------------

fn run() -> Result<()> {
    let cli = parse_cli()?;

    // Parse target hash.
    let digest_bytes = parse_md5_hex(&cli.hash_hex)?;
    let (ta, tb, tc, td) = digest_bytes_to_words_le(digest_bytes);

    // A tiny sanity check: verify our CPU md5 matches the provided hash
    // for a known string when user passes it (optional).
    // We keep this lesson self-contained and deterministic; no network, no files.

    // Choose GPU.
    autoreleasepool(|| -> Result<()> {
        let device = Device::system_default().context("no Metal device available")?;
        println!("GPU: {}", device.name());

        let pipelines = build_pipelines(&device, cli.charset.kernel_name())?;
        println!("Kernel: {}", cli.charset.kernel_name());

        // Allocate small control buffers once and reuse them per-length.
        // These are tiny: a few atomics and counters.
        let found_flag = device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let found_index = device.new_buffer(
            std::mem::size_of::<u64>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let match_count = device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let match_indices = device.new_buffer(
            (std::mem::size_of::<u64>() as u64) * (cli.max_matches as u64),
            MTLResourceOptions::StorageModeShared,
        );
        let params_buf = device.new_buffer(
            std::mem::size_of::<KernelParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Helper closures to read/write shared buffers.
        let reset_controls = || {
            unsafe {
                // Found flag + index.
                let pf = found_flag.contents() as *mut u32;
                *pf = 0;
                let pi = found_index.contents() as *mut u64;
                *pi = 0;

                // Match list.
                let pc = match_count.contents() as *mut u32;
                *pc = 0;

                // We do not zero match_indices; match_count tells us what is valid.
            }
        };

        let read_found_flag = || unsafe { *(found_flag.contents() as *const u32) };
        let read_found_index = || unsafe { *(found_index.contents() as *const u64) };
        let read_match_count = || unsafe { *(match_count.contents() as *const u32) };

        let read_match_indices = |count: u32| -> Vec<u64> {
            let mut out = Vec::with_capacity(count as usize);
            unsafe {
                let base = match_indices.contents() as *const u64;
                for i in 0..count {
                    out.push(*base.add(i as usize));
                }
            }
            out
        };

        let radix = cli.charset.radix() as u64;

        // We'll accumulate full run stats and optionally dump JSON.
        let wall_start = Instant::now();
        let mut total_candidates_tested: u64 = 0;
        let mut all_matches: Vec<String> = Vec::new();
        let mut length_reports: Vec<LengthReport> = Vec::new();

        let mut validation_checked: u64 = 0;
        let mut validation_mismatches: u64 = 0;

        // Progress printing uses a time interval rather than per-length, so
        // long lengths still provide feedback.
        let progress_interval = Duration::from_millis(cli.progress_ms);
        let mut next_progress = Instant::now() + progress_interval;

        'lengths: for len in cli.min_len..=cli.max_len {
            let candidates = pow_u64(radix, len)?;
            total_candidates_tested = total_candidates_tested
                .checked_add(candidates)
                .context("total candidate count overflow")?;

            reset_controls();

            // Fill KernelParams in shared buffer.
            let params = KernelParams {
                len,
                radix: cli.charset.radix(),
                total: candidates,
                candidates_per_thread: cli.candidates_per_thread,
                mode: cli.mode.as_u32(),
                max_matches: cli.max_matches,
                target_a: ta,
                target_b: tb,
                target_c: tc,
                target_d: td,
            };
            unsafe {
                *(params_buf.contents() as *mut KernelParams) = params;
            }

            // Choose dispatch sizes.
            //
            // Metal execution model:
            //   - threads_per_threadgroup controls how many threads cooperate in a group
            //   - grid size controls total threads launched
            //
            // For brute force, threads do not need to cooperate, so threadgroup size
            // is mainly a scheduling/granularity choice.
            let threads_per_tg = MTLSize {
                width: cli.threads_per_group as u64,
                height: 1,
                depth: 1,
            };

            // Heuristic for grid size:
            // Launch enough threadgroups to cover the GPU, without going extreme.
            // We use pipeline properties as a hint.
            let max_tg = pipelines.pipeline.max_total_threads_per_threadgroup() as u64;
            if threads_per_tg.width > max_tg {
                bail!(
                    "--threads-per-group {} exceeds pipeline max {}",
                    threads_per_tg.width,
                    max_tg
                );
            }

            // Threadgroup count heuristic:
            // - if candidate space is huge, more groups help saturate the GPU
            // - if candidate space is small, too many groups just adds overhead
            //
            // We'll aim for ~ (candidates / work_per_thread) threads, clamped.
            let work_per_thread = cli.candidates_per_thread as u64;
            let needed_threads = (candidates + work_per_thread - 1) / work_per_thread;

            // clamp threads to a reasonable range
            let min_threads = threads_per_tg.width * 4; // at least 4 threadgroups
            let max_threads = threads_per_tg.width * 65535; // arbitrary cap
            let threads = needed_threads.clamp(min_threads, max_threads);

            let threads_per_grid = MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            };

            // Encode and submit.
            let gpu_start = Instant::now();
            let cmd_buf = pipelines.queue.new_command_buffer();
            let enc = cmd_buf.new_compute_command_encoder();

            enc.set_compute_pipeline_state(&pipelines.pipeline);
            enc.set_buffer(0, Some(&params_buf), 0);
            enc.set_buffer(1, Some(&found_flag), 0);
            enc.set_buffer(2, Some(&found_index), 0);
            enc.set_buffer(3, Some(&match_count), 0);
            enc.set_buffer(4, Some(&match_indices), 0);

            enc.dispatch_threads(threads_per_grid, threads_per_tg);
            enc.end_encoding();

            cmd_buf.commit();

            // Wait for completion.
            // Teaching note: A production cracker would pipeline work across lengths
            // and avoid CPU waits. Here we want clear measurements and simplicity.
            cmd_buf.wait_until_completed();
            let gpu_ms = gpu_start.elapsed().as_secs_f64() * 1000.0;

            // Read match outputs.
            let mut found = false;
            let mut found_count = 0u32;
            let mut matches_for_len: Vec<u64> = Vec::new();

            match cli.mode {
                MatchMode::First => {
                    if read_found_flag() != 0 {
                        found = true;
                        found_count = 1;
                        matches_for_len.push(read_found_index());
                    }
                }
                MatchMode::All => {
                    let c = read_match_count();
                    found = c > 0;
                    found_count = c;
                    // Only the first max_matches indices are recorded.
                    let take = c.min(cli.max_matches);
                    matches_for_len = read_match_indices(take);
                }
            }

            // Validation.
            //
            // We never recompute *every* candidate unless validation=full.
            // The entire point is: GPU should be the fast path.
            if cli.validation != ValidationMode::GpuOnly {
                // Always validate any reported matches.
                for &idx in &matches_for_len {
                    let cand = index_to_candidate(idx, len, cli.charset);
                    let d = md5_bytes(cand.as_bytes());
                    validation_checked += 1;
                    if d != digest_bytes {
                        validation_mismatches += 1;
                    }
                }

                match cli.validation {
                    ValidationMode::Spot => {
                        // Spot-check a small, deterministic sample of indices.
                        // Deterministic matters for teaching: runs are reproducible.
                        let sample = 256u64.min(candidates);
                        if sample > 0 {
                            // Sample indices spread across the space.
                            for s in 0..sample {
                                let idx = (s * (candidates / sample.max(1))) % candidates;
                                let cand = index_to_candidate(idx, len, cli.charset);
                                let d = md5_bytes(cand.as_bytes());
                                validation_checked += 1;

                                // If it equals target but GPU didn't report it,
                                // that's a correctness failure.
                                if d == digest_bytes {
                                    let gpu_reported = matches_for_len.contains(&idx);
                                    if !gpu_reported {
                                        validation_mismatches += 1;
                                    }
                                }
                            }
                        }
                    }
                    ValidationMode::Full => {
                        // Full CPU mirror (very slow): recompute all candidates.
                        // This is explicitly for teaching and debugging.
                        for idx in 0..candidates {
                            let cand = index_to_candidate(idx, len, cli.charset);
                            let d = md5_bytes(cand.as_bytes());
                            validation_checked += 1;

                            if d == digest_bytes {
                                // Ensure GPU reported it.
                                let gpu_reported = matches_for_len.contains(&idx)
                                    || (cli.mode == MatchMode::First && found && read_found_index() == idx);
                                if !gpu_reported {
                                    validation_mismatches += 1;
                                }
                            }
                        }
                    }
                    ValidationMode::GpuOnly => {}
                }
            }

            // Convert match indices to plaintext strings and store.
            let mut match_strings: Vec<String> = Vec::new();
            for idx in matches_for_len {
                match_strings.push(index_to_candidate(idx, len, cli.charset));
            }
            all_matches.extend(match_strings.iter().cloned());

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

            if cli.verbose {
                println!(
                    "len={len} radix={} candidates={} gpu_ms={:.2} mh/s={:.2} found={} hits={}",
                    cli.charset.radix(),
                    candidates,
                    gpu_ms,
                    mh_s,
                    found,
                    found_count
                );
            }

            // If we're in first-match mode and we found something, stop ramping.
            if cli.mode == MatchMode::First && found {
                break 'lengths;
            }

            // Progress ticker for long runs.
            if Instant::now() >= next_progress {
                println!(
                    "progress: completed lengths up to {len} (total tested so far: {})",
                    length_reports
                        .iter()
                        .map(|r| r.candidates)
                        .sum::<u64>()
                );
                next_progress = Instant::now() + progress_interval;
            }
        }

        let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
        let overall_mh_s = (total_candidates_tested as f64) / (wall_ms / 1000.0) / 1_000_000.0;

        // Final result printing.
        println!("\nTarget MD5: {}", cli.hash_hex.to_lowercase());
        println!("Charset: {} (radix={})", cli.charset.as_str(), cli.charset.radix());
        println!("Mode: {}", cli.mode.as_str());
        println!("Validation: {}", cli.validation.as_str());
        println!("Lengths: {}..{}", cli.min_len, cli.max_len);

        if all_matches.is_empty() {
            println!("Result: NOT FOUND in search bounds");
        } else {
            println!("Result: FOUND {} match(es)", all_matches.len());
            // Print a few matches; if mode=all and many hits, keep it readable.
            let show = all_matches.len().min(16);
            for m in all_matches.iter().take(show) {
                println!("  match: {m}");
            }
            if all_matches.len() > show {
                println!("  ... ({} more)", all_matches.len() - show);
            }
        }

        println!("Total candidates tested: {total_candidates_tested}");
        println!("Wall time: {:.2} ms", wall_ms);
        println!("Overall throughput: {:.2} MH/s", overall_mh_s);

        if cli.validation != ValidationMode::GpuOnly {
            println!(
                "Validation checked: {} (mismatches: {})",
                validation_checked, validation_mismatches
            );
        }

        // Optional JSON report.
        if let Some(path) = &cli.json {
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
                matches: all_matches.clone(),
                lengths: length_reports.clone(),
                validation_checked,
                validation_mismatches,
            };

            let json = serde_json::to_string_pretty(&report)?;
            fs::write(path, json).with_context(|| format!("write json report: {}", path.display()))?;
            println!("Wrote JSON report: {}", path.display());
        }

        // If validation mismatched, fail the process.
        if validation_mismatches > 0 {
            bail!("validation mismatches detected: {validation_mismatches}");
        }

        Ok(())
    })
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
