use anyhow::{Context, Result, bail};
use bytemuck::{Pod, Zeroable};
use serde::Serialize;
use sha1::{Digest, Sha1};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

const SHADER_SOURCE: &str = include_str!("sha1_brute_force.wgsl");

const DEFAULT_CHARSET: &str = "lowernum";
const DEFAULT_MIN_LEN: u32 = 1;
const DEFAULT_MAX_LEN: u32 = 6;
const DEFAULT_MODE: &str = "first";
const DEFAULT_VALIDATION: &str = "spot";
const DEFAULT_THREADS_PER_GROUP: u32 = 256;
const DEFAULT_CANDIDATES_PER_THREAD: u32 = 8;
const DEFAULT_PROGRESS_MS: u64 = 500;
const DEFAULT_MAX_MATCHES: u32 = 1024;

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

    fn alphabet_id(self) -> u32 {
        match self {
            Self::Lower => 0,
            Self::LowerNum => 1,
            Self::Printable => 2,
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
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct KernelParams {
    len: u32,
    radix: u32,
    search_space: u64,
    candidates_per_thread: u32,
    mode: u32,
    max_matches: u32,
    alphabet_id: u32,
    target_a: u32,
    target_b: u32,
    target_c: u32,
    target_d: u32,
    target_e: u32,
    _pad0: u32,
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

fn usage() -> &'static str {
    r#"Lesson 13: GPU SHA1 brute forcing (Rust + wgpu)

USAGE:
  cargo run --release -p sha1-rust-kernel -- --hash <40hex> [options]

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
  - Runtime requires adapter support for wgpu feature SHADER_INT64.
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
        match args[i].as_str() {
            "--help" | "-h" => {
                print!("{}", usage());
                std::process::exit(0);
            }
            "--hash" => {
                i += 1;
                hash_hex = Some(args.get(i).context("--hash requires a value")?.to_string());
            }
            "--charset" => {
                i += 1;
                charset = Charset::parse(args.get(i).context("--charset requires a value")?)?;
            }
            "--min-len" => {
                i += 1;
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
                json = Some(PathBuf::from(args.get(i).context("--json requires a path")?));
            }
            "--verbose" => {
                verbose = true;
            }
            other => bail!("unknown arg: {other}\n\n{}", usage()),
        }
        i += 1;
    }

    let hash_hex = hash_hex.context("--hash is required")?;
    parse_sha1_hex(&hash_hex)?;

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

fn parse_sha1_hex(hex: &str) -> Result<[u8; 20]> {
    let h = hex.trim();
    if h.len() != 40 {
        bail!("--hash must be 40 hex chars (got {})", h.len());
    }

    let mut out = [0u8; 20];
    for i in 0..20 {
        out[i] = u8::from_str_radix(&h[i * 2..i * 2 + 2], 16)
            .with_context(|| format!("invalid hex at byte {i}"))?;
    }
    Ok(out)
}

fn digest_bytes_to_words_be(d: [u8; 20]) -> (u32, u32, u32, u32, u32) {
    let a = u32::from_be_bytes([d[0], d[1], d[2], d[3]]);
    let b = u32::from_be_bytes([d[4], d[5], d[6], d[7]]);
    let c = u32::from_be_bytes([d[8], d[9], d[10], d[11]]);
    let d2 = u32::from_be_bytes([d[12], d[13], d[14], d[15]]);
    let e = u32::from_be_bytes([d[16], d[17], d[18], d[19]]);
    (a, b, c, d2, e)
}

fn pow_u64(base: u64, exp: u32) -> Result<u64> {
    let mut acc = 1u64;
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

    String::from_utf8_lossy(&bytes).to_string()
}

fn sha1_bytes(input: &[u8]) -> [u8; 20] {
    let mut hasher = Sha1::new();
    hasher.update(input);
    let digest = hasher.finalize();
    let mut out = [0u8; 20];
    out.copy_from_slice(&digest);
    out
}

fn readback_bytes(device: &wgpu::Device, buffer: &wgpu::Buffer, size: u64) -> Result<Vec<u8>> {
    if size == 0 {
        return Ok(Vec::new());
    }

    let slice = buffer.slice(0..size);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });

    device
        .poll(wgpu::PollType::wait_indefinitely())
        .context("device.poll failed while waiting for map_async")?;

    rx.recv()
        .context("failed receiving map_async callback")?
        .context("map_async failed")?;

    let view = slice.get_mapped_range();
    let data = view.to_vec();
    drop(view);
    buffer.unmap();
    Ok(data)
}

fn readback_u32(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<u32> {
    let data = readback_bytes(device, buffer, 4)?;
    let bytes: [u8; 4] = data[0..4].try_into().context("u32 readback size mismatch")?;
    Ok(u32::from_ne_bytes(bytes))
}

fn readback_u64(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<u64> {
    let data = readback_bytes(device, buffer, 8)?;
    let bytes: [u8; 8] = data[0..8].try_into().context("u64 readback size mismatch")?;
    Ok(u64::from_ne_bytes(bytes))
}

fn readback_u64_vec(device: &wgpu::Device, buffer: &wgpu::Buffer, count: u32) -> Result<Vec<u64>> {
    if count == 0 {
        return Ok(Vec::new());
    }

    let size = (count as u64) * 8;
    let data = readback_bytes(device, buffer, size)?;
    let mut out = Vec::with_capacity(count as usize);
    for chunk in data.chunks_exact(8) {
        let bytes: [u8; 8] = chunk.try_into().context("u64 slice size mismatch")?;
        out.push(u64::from_ne_bytes(bytes));
    }
    Ok(out)
}

fn run() -> Result<()> {
    let cli = parse_cli()?;

    let digest_bytes = parse_sha1_hex(&cli.hash_hex)?;
    let (ta, tb, tc, td, te) = digest_bytes_to_words_be(digest_bytes);

    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .context("failed to request wgpu adapter")?;

    let info = adapter.get_info();
    println!("GPU: {}", info.name);

    if !adapter.features().contains(wgpu::Features::SHADER_INT64) {
        bail!(
            "adapter '{}' does not support SHADER_INT64; lesson 13 requires it for 64-bit search indices",
            info.name
        );
    }

    let limits = adapter.limits();
    if cli.threads_per_group > limits.max_compute_invocations_per_workgroup {
        bail!(
            "--threads-per-group {} exceeds adapter max_compute_invocations_per_workgroup {}",
            cli.threads_per_group,
            limits.max_compute_invocations_per_workgroup
        );
    }
    if cli.threads_per_group > limits.max_compute_workgroup_size_x {
        bail!(
            "--threads-per-group {} exceeds adapter max_compute_workgroup_size_x {}",
            cli.threads_per_group,
            limits.max_compute_workgroup_size_x
        );
    }

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("sha1-rust-kernel-device"),
        required_features: wgpu::Features::SHADER_INT64,
        required_limits: wgpu::Limits::default(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .context("failed to create wgpu device")?;

    let shader = unsafe {
        device.create_shader_module_trusted(
            wgpu::ShaderModuleDescriptor {
                label: Some("sha1-wgsl-kernel"),
                source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
            },
            wgpu::ShaderRuntimeChecks::unchecked(),
        )
    };

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sha1-bind-group-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sha1-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        immediate_size: 0,
    });

    let workgroup_constants = [("WORKGROUP_SIZE", cli.threads_per_group as f64)];
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sha1-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("sha1_brute_force"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &workgroup_constants,
            ..Default::default()
        },
        cache: None,
    });

    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: std::mem::size_of::<KernelParams>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let found_flag_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("found_flag"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let found_index_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("found_index"),
        size: 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let match_count_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("match_count"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let match_slots = cli.max_matches.max(1);
    let match_indices_size = (match_slots as u64) * 8;
    let match_indices_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("match_indices"),
        size: match_indices_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let found_flag_read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("found_flag_read"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let found_index_read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("found_index_read"),
        size: 8,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let match_count_read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("match_count_read"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let match_indices_read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("match_indices_read"),
        size: match_indices_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sha1-bind-group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: found_flag_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: found_index_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: match_count_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: match_indices_buf.as_entire_binding(),
            },
        ],
    });

    println!("Kernel: sha1_brute_force");

    let radix = cli.charset.radix() as u64;
    let wall_start = Instant::now();
    let mut total_candidates_tested: u64 = 0;
    let mut all_matches: Vec<String> = Vec::new();
    let mut length_reports: Vec<LengthReport> = Vec::new();
    let mut validation_checked: u64 = 0;
    let mut validation_mismatches: u64 = 0;

    let progress_interval = Duration::from_millis(cli.progress_ms);
    let mut next_progress = Instant::now() + progress_interval;

    'lengths: for len in cli.min_len..=cli.max_len {
        let candidates = pow_u64(radix, len)?;
        total_candidates_tested = total_candidates_tested
            .checked_add(candidates)
            .context("total candidate count overflow")?;

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
            _pad0: 0,
        };

        queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));
        queue.write_buffer(&found_flag_buf, 0, &0u32.to_ne_bytes());
        queue.write_buffer(&found_index_buf, 0, &0u64.to_ne_bytes());
        queue.write_buffer(&match_count_buf, 0, &0u32.to_ne_bytes());

        let work_per_thread = cli.candidates_per_thread as u64;
        let needed_threads = candidates.div_ceil(work_per_thread);
        let min_threads = (cli.threads_per_group as u64) * 4;
        let max_threads = (cli.threads_per_group as u64) * 65535;
        let threads = needed_threads.clamp(min_threads, max_threads);

        let workgroups = threads.div_ceil(cli.threads_per_group as u64);
        if workgroups > limits.max_compute_workgroups_per_dimension as u64 {
            bail!(
                "required workgroups {} exceed adapter max_compute_workgroups_per_dimension {}",
                workgroups,
                limits.max_compute_workgroups_per_dimension
            );
        }

        let gpu_start = Instant::now();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sha1-encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sha1-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&found_flag_buf, 0, &found_flag_read, 0, 4);
        encoder.copy_buffer_to_buffer(&found_index_buf, 0, &found_index_read, 0, 8);
        encoder.copy_buffer_to_buffer(&match_count_buf, 0, &match_count_read, 0, 4);
        encoder.copy_buffer_to_buffer(
            &match_indices_buf,
            0,
            &match_indices_read,
            0,
            match_indices_size,
        );

        queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("device.poll failed after queue submit")?;

        let gpu_ms = gpu_start.elapsed().as_secs_f64() * 1000.0;

        let mut found = false;
        let mut found_count = 0u32;
        let mut matches_for_len: Vec<u64> = Vec::new();

        match cli.mode {
            MatchMode::First => {
                let found_flag = readback_u32(&device, &found_flag_read)?;
                if found_flag != 0 {
                    found = true;
                    found_count = 1;
                    matches_for_len.push(readback_u64(&device, &found_index_read)?);
                }
            }
            MatchMode::All => {
                let c = readback_u32(&device, &match_count_read)?;
                found = c > 0;
                found_count = c;
                let capped = c.min(cli.max_matches);
                matches_for_len = readback_u64_vec(&device, &match_indices_read, capped)?;
            }
        }

        if cli.validation != ValidationMode::GpuOnly {
            for &idx in &matches_for_len {
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
                    let sample = 256u64.min(candidates);
                    if sample > 0 {
                        for s in 0..sample {
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
                    for idx in 0..candidates {
                        let cand = index_to_candidate(idx, len, cli.charset);
                        let d = sha1_bytes(cand.as_bytes());
                        validation_checked += 1;
                        if d == digest_bytes {
                            let gpu_reported = matches_for_len.contains(&idx);
                            if !gpu_reported {
                                validation_mismatches += 1;
                            }
                        }
                    }
                }
            }
        }

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

        for idx in matches_for_len {
            all_matches.push(index_to_candidate(idx, len, cli.charset));
        }

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

        if cli.mode == MatchMode::First && found {
            break 'lengths;
        }
    }

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
            matches: all_matches,
            lengths: length_reports,
            validation_checked,
            validation_mismatches,
        };
        fs::write(path, serde_json::to_string_pretty(&report)?)?;
        println!("Wrote JSON report: {}", path.display());
    }

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
