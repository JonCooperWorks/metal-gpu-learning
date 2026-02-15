// =============================================================================
// LESSON 10: Medicaid Provider Spending -- Multi-Kernel GPU Analytics
// =============================================================================
//
// This lesson is intentionally heavily annotated. The goal is to show exactly
// how to move from a real public parquet dataset to a selectable 1..N GPU
// analysis pipeline in Rust + Metal.
//
// High-level flow:
//   1) Parse CLI options and validate requested kernel IDs.
//   2) Read parquet rows and project only required columns.
//   3) Encode group keys to compact integer IDs (GPU-friendly).
//   4) Build normalized feature vectors used by all kernels.
//   5) Run selected kernels independently on GPU (separate dispatches).
//   6) Optionally compare GPU outputs with CPU reference scores.
//   7) Assemble top-K results and optional JSON report.
// =============================================================================

use anyhow::{bail, Context, Result};
use arrow_array::{
    Array, Date32Array, Float32Array, Float64Array, Int32Array, Int64Array, LargeStringArray,
    StringArray, UInt32Array, UInt64Array,
};
use metal::*;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Serialize;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const SHADER_SOURCE: &str = include_str!("lesson10.metal");
const THREADGROUP_SIZE: u64 = 256;
const DEFAULT_TOP_K: usize = 50;
const DEFAULT_MIN_CLAIMS: u32 = 12;
const DEFAULT_MEMORY_FRACTION: f64 = 0.80;
const DEFAULT_CHUNK_ROWS: usize = 1_000_000;

// Threshold defaults from the agreed plan.
const DEFAULT_K2_THRESHOLD: f32 = 3.0;
const DEFAULT_K3_THRESHOLD: f32 = 3.5;
const DEFAULT_K4_THRESHOLD: f32 = 3.5;
const DEFAULT_K5_RATIO_THRESHOLD: f32 = 2.0;
const DEFAULT_K5_ABS_FLOOR: f32 = 5_000.0;
const DEFAULT_K6_THRESHOLD: f32 = 2.5;
const DEFAULT_K7_PERCENTILE: f32 = 99.0;
const DEFAULT_K8_TOP_PERCENT: f32 = 1.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Single,
    Double,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ValidationMode {
    Full,
    Spot,
    Off,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CompareCpu {
    On,
    Off,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GroupBy {
    BillingNpi,
    ServicingNpi,
    Hcpcs,
    Month,
    BillingNpiHcpcs,
}

impl GroupBy {
    fn parse(v: &str) -> Result<Self> {
        Ok(match v {
            "billing_npi" => Self::BillingNpi,
            "servicing_npi" => Self::ServicingNpi,
            "hcpcs" => Self::Hcpcs,
            "month" => Self::Month,
            "billing_npi_hcpcs" => Self::BillingNpiHcpcs,
            _ => bail!("invalid --group-by '{v}'"),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize)]
#[serde(transparent)]
struct KernelId(u8);

impl TryFrom<u8> for KernelId {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> Result<Self> {
        if (1..=8).contains(&value) {
            Ok(Self(value))
        } else {
            bail!("kernel ID {value} is out of range; valid IDs are 1..8")
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct KernelSpec {
    id: KernelId,
    name: &'static str,
    description: &'static str,
}

const KERNEL_SPECS: [KernelSpec; 8] = [
    KernelSpec {
        id: KernelId(1),
        name: "Top spenders",
        description: "Aggregates spending by selected group key and ranks by total paid.",
    },
    KernelSpec {
        id: KernelId(2),
        name: "Z-score anomaly",
        description: "Absolute z-score over TOTAL_PAID.",
    },
    KernelSpec {
        id: KernelId(3),
        name: "MAD anomaly",
        description: "Modified z-score using median absolute deviation over TOTAL_PAID.",
    },
    KernelSpec {
        id: KernelId(4),
        name: "Paid-per-claim anomaly",
        description: "Modified z-score over TOTAL_PAID/TOTAL_CLAIMS.",
    },
    KernelSpec {
        id: KernelId(5),
        name: "Month-over-month spike",
        description: "Relative paid spike score using prior month baseline within group.",
    },
    KernelSpec {
        id: KernelId(6),
        name: "Provider drift",
        description: "Rolling baseline drift in sigma units.",
    },
    KernelSpec {
        id: KernelId(7),
        name: "HCPCS rarity-weighted",
        description: "Spending weighted by inverse HCPCS frequency.",
    },
    KernelSpec {
        id: KernelId(8),
        name: "Distance outlier",
        description: "Feature-space distance from global centroid.",
    },
];

#[derive(Clone, Debug)]
struct CliConfig {
    input: PathBuf,
    kernels: Vec<KernelId>,
    list_kernels: bool,
    group_by: GroupBy,
    top_k: usize,
    min_claims: u32,
    chunk_rows: usize,
    memory_fraction: f64,
    mode: Mode,
    compare_cpu: CompareCpu,
    validate: ValidationMode,
    output_json: Option<PathBuf>,
    k2_threshold: f32,
    k3_threshold: f32,
    k4_threshold: f32,
    k5_ratio_threshold: f32,
    k5_abs_floor: f32,
    k6_threshold: f32,
    k7_percentile: f32,
    k8_top_percent: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct KernelParams {
    // Number of rows to process in this dispatch.
    count: u32,
    // Global row offset for this chunk.
    offset: u32,
    // Distribution stats for z-score and MAD style kernels.
    mean: f32,
    std_dev: f32,
    median: f32,
    mad: f32,
    // Generic auxiliary values used by specific kernels.
    aux_a: f32,
    aux_b: f32,
}

#[derive(Clone, Debug)]
struct InputRow {
    hcpcs: String,
    month_label: String,
    month_day: i32,
    beneficiaries: u32,
    claims: u32,
    paid: f32,
    group_id: u32,
}

#[derive(Clone, Debug)]
struct FeatureMatrix {
    paid: Vec<f32>,
    claims: Vec<f32>,
    beneficiaries: Vec<f32>,
    paid_per_claim: Vec<f32>,
    mom_ratio: Vec<f32>,
    drift_sigma: Vec<f32>,
    rarity_weight: Vec<f32>,
    distance_score: Vec<f32>,
    mom_abs_delta: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Default)]
struct DistributionStats {
    mean: f32,
    std_dev: f32,
    median: f32,
    mad: f32,
}

#[derive(Debug, Serialize)]
struct OutputEntry {
    group_key: String,
    month: String,
    paid: f64,
    claims: u64,
    beneficiaries: u64,
    score: f64,
    reason: String,
}

#[derive(Debug, Serialize)]
struct KernelReport {
    kernel_id: u8,
    kernel_name: String,
    gpu_ms: f64,
    cpu_ms: f64,
    validation_ms: f64,
    kernel_total_ms: f64,
    threshold: f64,
    rows_flagged: usize,
    top: Vec<OutputEntry>,
    validation_mismatches: usize,
}

#[derive(Debug, Serialize)]
struct RunReport {
    input_path: String,
    input_size_bytes: u64,
    selected_kernels: Vec<u8>,
    row_count: usize,
    group_count: usize,
    total_duration_ms: f64,
    reports: Vec<KernelReport>,
}

#[derive(Clone, Copy, Debug, Default)]
struct KernelTimings {
    gpu_ms: f64,
    cpu_ms: f64,
    validation_ms: f64,
}

struct GpuContext {
    device: Device,
    command_queue: CommandQueue,
    pipelines: BTreeMap<u8, ComputePipelineState>,
}

fn print_usage() {
    println!(
        "Usage: medicaid-provider-spending --input <path.parquet> --kernels <id,id,...> [options]"
    );
    println!();
    println!("Core flags:");
    println!("  --input <path>                  Local parquet path (required unless --list-kernels)");
    println!("  --kernels <1,3,7>              Comma-separated kernel IDs (required unless --list-kernels)");
    println!("  --list-kernels                 Print kernel catalog and exit");
    println!("  --group-by billing_npi|servicing_npi|hcpcs|month|billing_npi_hcpcs (default: billing_npi)");
    println!("  --top-k <u32>                  Number of rows/groups to display (default: {DEFAULT_TOP_K})");
    println!("  --min-claims <u32>             Drop rows with fewer claims (default: {DEFAULT_MIN_CLAIMS})");
    println!("  --chunk-rows <u32>             Dispatch chunk size (default: {DEFAULT_CHUNK_ROWS})");
    println!("  --memory-fraction <f64>        Memory budget hint for future chunk auto-sizing (default: {DEFAULT_MEMORY_FRACTION:.2})");
    println!("  --mode single|double           Scheduling mode label (default: double)");
    println!("  --compare-cpu on|off           Compute CPU reference scores (default: on)");
    println!("  --validate full|spot|off       GPU vs CPU score checks (default: spot)");
    println!("  --output-json <path>           Write report JSON");
    println!();
    println!("Threshold flags:");
    println!("  --k2-threshold <f32>           Z-score threshold (default: {DEFAULT_K2_THRESHOLD})");
    println!("  --k3-threshold <f32>           MAD threshold on paid (default: {DEFAULT_K3_THRESHOLD})");
    println!("  --k4-threshold <f32>           MAD threshold on paid/claim (default: {DEFAULT_K4_THRESHOLD})");
    println!("  --k5-ratio-threshold <f32>     MoM ratio threshold (default: {DEFAULT_K5_RATIO_THRESHOLD})");
    println!("  --k5-abs-floor <f32>           MoM absolute delta floor (default: {DEFAULT_K5_ABS_FLOOR})");
    println!("  --k6-threshold <f32>           Drift sigma threshold (default: {DEFAULT_K6_THRESHOLD})");
    println!("  --k7-percentile <f32>          Rarity anomaly percentile (default: {DEFAULT_K7_PERCENTILE})");
    println!("  --k8-top-percent <f32>         Distance top-percent cutoff (default: {DEFAULT_K8_TOP_PERCENT})");
}

// -----------------------------------------------------------------------------
// parse_args
// -----------------------------------------------------------------------------
// This is a manual CLI parser (no clap dependency) so readers can see all
// defaults and validations in one place.
//
// Practical notes for beginners:
// - `--kernels` accepts comma-separated numeric IDs (for example `1,2,8`).
// - `--list-kernels` short-circuits normal required arguments.
// - We keep threshold flags explicit so each kernel can be reasoned about in
//   isolation during experiments.
fn parse_args() -> Result<CliConfig> {
    let mut input = None;
    let mut kernels = None;
    let mut list_kernels = false;
    let mut group_by = GroupBy::BillingNpi;
    let mut top_k = DEFAULT_TOP_K;
    let mut min_claims = DEFAULT_MIN_CLAIMS;
    let mut chunk_rows = DEFAULT_CHUNK_ROWS;
    let mut memory_fraction = DEFAULT_MEMORY_FRACTION;
    let mut mode = Mode::Double;
    let mut compare_cpu = CompareCpu::On;
    let mut validate = ValidationMode::Spot;
    let mut output_json = None;

    let mut k2_threshold = DEFAULT_K2_THRESHOLD;
    let mut k3_threshold = DEFAULT_K3_THRESHOLD;
    let mut k4_threshold = DEFAULT_K4_THRESHOLD;
    let mut k5_ratio_threshold = DEFAULT_K5_RATIO_THRESHOLD;
    let mut k5_abs_floor = DEFAULT_K5_ABS_FLOOR;
    let mut k6_threshold = DEFAULT_K6_THRESHOLD;
    let mut k7_percentile = DEFAULT_K7_PERCENTILE;
    let mut k8_top_percent = DEFAULT_K8_TOP_PERCENT;

    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            "--input" => {
                input = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--kernels" => {
                kernels = args.get(i + 1).cloned();
                i += 2;
            }
            "--list-kernels" => {
                list_kernels = true;
                i += 1;
            }
            "--group-by" => {
                let value = args.get(i + 1).context("missing value for --group-by")?;
                group_by = GroupBy::parse(value)?;
                i += 2;
            }
            "--top-k" => {
                top_k = args
                    .get(i + 1)
                    .context("missing value for --top-k")?
                    .parse::<usize>()
                    .context("--top-k expects unsigned integer")?;
                i += 2;
            }
            "--min-claims" => {
                min_claims = args
                    .get(i + 1)
                    .context("missing value for --min-claims")?
                    .parse::<u32>()
                    .context("--min-claims expects unsigned integer")?;
                i += 2;
            }
            "--chunk-rows" => {
                chunk_rows = args
                    .get(i + 1)
                    .context("missing value for --chunk-rows")?
                    .parse::<usize>()
                    .context("--chunk-rows expects unsigned integer")?;
                i += 2;
            }
            "--memory-fraction" => {
                memory_fraction = args
                    .get(i + 1)
                    .context("missing value for --memory-fraction")?
                    .parse::<f64>()
                    .context("--memory-fraction expects float")?;
                i += 2;
            }
            "--mode" => {
                mode = match args.get(i + 1).context("missing value for --mode")?.as_str() {
                    "single" => Mode::Single,
                    "double" => Mode::Double,
                    v => bail!("invalid --mode '{v}'"),
                };
                i += 2;
            }
            "--compare-cpu" => {
                compare_cpu = match args
                    .get(i + 1)
                    .context("missing value for --compare-cpu")?
                    .as_str()
                {
                    "on" => CompareCpu::On,
                    "off" => CompareCpu::Off,
                    v => bail!("invalid --compare-cpu '{v}'"),
                };
                i += 2;
            }
            "--validate" => {
                validate = match args.get(i + 1).context("missing value for --validate")?.as_str() {
                    "full" => ValidationMode::Full,
                    "spot" => ValidationMode::Spot,
                    "off" => ValidationMode::Off,
                    v => bail!("invalid --validate '{v}'"),
                };
                i += 2;
            }
            "--output-json" => {
                output_json = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--k2-threshold" => {
                k2_threshold = args
                    .get(i + 1)
                    .context("missing value for --k2-threshold")?
                    .parse::<f32>()
                    .context("--k2-threshold expects float")?;
                i += 2;
            }
            "--k3-threshold" => {
                k3_threshold = args
                    .get(i + 1)
                    .context("missing value for --k3-threshold")?
                    .parse::<f32>()
                    .context("--k3-threshold expects float")?;
                i += 2;
            }
            "--k4-threshold" => {
                k4_threshold = args
                    .get(i + 1)
                    .context("missing value for --k4-threshold")?
                    .parse::<f32>()
                    .context("--k4-threshold expects float")?;
                i += 2;
            }
            "--k5-ratio-threshold" => {
                k5_ratio_threshold = args
                    .get(i + 1)
                    .context("missing value for --k5-ratio-threshold")?
                    .parse::<f32>()
                    .context("--k5-ratio-threshold expects float")?;
                i += 2;
            }
            "--k5-abs-floor" => {
                k5_abs_floor = args
                    .get(i + 1)
                    .context("missing value for --k5-abs-floor")?
                    .parse::<f32>()
                    .context("--k5-abs-floor expects float")?;
                i += 2;
            }
            "--k6-threshold" => {
                k6_threshold = args
                    .get(i + 1)
                    .context("missing value for --k6-threshold")?
                    .parse::<f32>()
                    .context("--k6-threshold expects float")?;
                i += 2;
            }
            "--k7-percentile" => {
                k7_percentile = args
                    .get(i + 1)
                    .context("missing value for --k7-percentile")?
                    .parse::<f32>()
                    .context("--k7-percentile expects float")?;
                i += 2;
            }
            "--k8-top-percent" => {
                k8_top_percent = args
                    .get(i + 1)
                    .context("missing value for --k8-top-percent")?
                    .parse::<f32>()
                    .context("--k8-top-percent expects float")?;
                i += 2;
            }
            unknown => {
                bail!("unknown argument '{unknown}'")
            }
        }
    }

    if !(0.0..=1.0).contains(&memory_fraction) {
        bail!("--memory-fraction must be between 0 and 1")
    }
    if !(0.0..=100.0).contains(&k7_percentile) || k7_percentile == 0.0 {
        bail!("--k7-percentile must be in (0, 100]")
    }
    if !(0.0..100.0).contains(&k8_top_percent) {
        bail!("--k8-top-percent must be in (0, 100)")
    }

    if list_kernels {
        return Ok(CliConfig {
            input: input.unwrap_or_default(),
            kernels: Vec::new(),
            list_kernels,
            group_by,
            top_k,
            min_claims,
            chunk_rows,
            memory_fraction,
            mode,
            compare_cpu,
            validate,
            output_json,
            k2_threshold,
            k3_threshold,
            k4_threshold,
            k5_ratio_threshold,
            k5_abs_floor,
            k6_threshold,
            k7_percentile,
            k8_top_percent,
        });
    }

    let input = input.context("--input is required unless --list-kernels is used")?;
    let kernels_raw = kernels.context("--kernels is required unless --list-kernels is used")?;

    let parsed = parse_kernel_ids(&kernels_raw)?;

    Ok(CliConfig {
        input,
        kernels: parsed,
        list_kernels,
        group_by,
        top_k,
        min_claims,
        chunk_rows,
        memory_fraction,
        mode,
        compare_cpu,
        validate,
        output_json,
        k2_threshold,
        k3_threshold,
        k4_threshold,
        k5_ratio_threshold,
        k5_abs_floor,
        k6_threshold,
        k7_percentile,
        k8_top_percent,
    })
}

fn parse_kernel_ids(value: &str) -> Result<Vec<KernelId>> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();

    for piece in value.split(',') {
        let trimmed = piece.trim();
        if trimmed.is_empty() {
            continue;
        }
        let id_num = trimmed
            .parse::<u8>()
            .with_context(|| format!("kernel id '{trimmed}' is not a valid integer"))?;
        let id = KernelId::try_from(id_num)?;
        if !seen.insert(id) {
            bail!("kernel ID {} is duplicated in --kernels", id.0);
        }
        out.push(id);
    }

    if out.is_empty() {
        bail!("--kernels must include at least one ID in 1..8")
    }

    out.sort();
    Ok(out)
}

fn print_kernel_catalog() {
    println!("Kernel Catalog (stable IDs):");
    for spec in KERNEL_SPECS {
        println!("  {}: {}", spec.id.0, spec.name);
        println!("     {}", spec.description);
    }
}

fn kernel_name(id: KernelId) -> &'static str {
    for spec in KERNEL_SPECS {
        if spec.id == id {
            return spec.name;
        }
    }
    "unknown"
}

fn month_label_from_date32(days_since_epoch: i32) -> String {
    // We only need a stable month label for grouping and reporting. Date32 is
    // days since 1970-01-01, so an approximate conversion is enough here.
    let epoch_year = 1970i32;
    let approx_year = epoch_year + days_since_epoch / 365;
    let day_of_year = days_since_epoch.rem_euclid(365);
    let approx_month = 1 + (day_of_year / 30).clamp(0, 11);
    format!("{:04}-{:02}-01", approx_year, approx_month)
}

fn extract_string(array: &dyn Array, row: usize) -> Option<String> {
    if let Some(v) = array.as_any().downcast_ref::<StringArray>() {
        return (!v.is_null(row)).then(|| v.value(row).to_string());
    }
    if let Some(v) = array.as_any().downcast_ref::<LargeStringArray>() {
        return (!v.is_null(row)).then(|| v.value(row).to_string());
    }
    None
}

fn extract_u32(array: &dyn Array, row: usize) -> Option<u32> {
    if let Some(v) = array.as_any().downcast_ref::<UInt32Array>() {
        return (!v.is_null(row)).then(|| v.value(row));
    }
    if let Some(v) = array.as_any().downcast_ref::<UInt64Array>() {
        return (!v.is_null(row)).then(|| v.value(row) as u32);
    }
    if let Some(v) = array.as_any().downcast_ref::<Int32Array>() {
        return (!v.is_null(row)).then(|| v.value(row).max(0) as u32);
    }
    if let Some(v) = array.as_any().downcast_ref::<Int64Array>() {
        return (!v.is_null(row)).then(|| v.value(row).max(0) as u32);
    }
    if let Some(v) = array.as_any().downcast_ref::<Float32Array>() {
        return (!v.is_null(row)).then(|| v.value(row).max(0.0) as u32);
    }
    if let Some(v) = array.as_any().downcast_ref::<Float64Array>() {
        return (!v.is_null(row)).then(|| v.value(row).max(0.0) as u32);
    }
    None
}

fn extract_f32(array: &dyn Array, row: usize) -> Option<f32> {
    if let Some(v) = array.as_any().downcast_ref::<Float32Array>() {
        return (!v.is_null(row)).then(|| v.value(row));
    }
    if let Some(v) = array.as_any().downcast_ref::<Float64Array>() {
        return (!v.is_null(row)).then(|| v.value(row) as f32);
    }
    if let Some(v) = array.as_any().downcast_ref::<Int64Array>() {
        return (!v.is_null(row)).then(|| v.value(row) as f32);
    }
    if let Some(v) = array.as_any().downcast_ref::<Int32Array>() {
        return (!v.is_null(row)).then(|| v.value(row) as f32);
    }
    if let Some(v) = array.as_any().downcast_ref::<UInt64Array>() {
        return (!v.is_null(row)).then(|| v.value(row) as f32);
    }
    if let Some(v) = array.as_any().downcast_ref::<UInt32Array>() {
        return (!v.is_null(row)).then(|| v.value(row) as f32);
    }
    None
}

fn extract_month(array: &dyn Array, row: usize) -> Option<(i32, String)> {
    if let Some(v) = array.as_any().downcast_ref::<Date32Array>() {
        if v.is_null(row) {
            return None;
        }
        let day = v.value(row);
        return Some((day, month_label_from_date32(day)));
    }

    // Some parquet files may expose date columns as strings. If that happens,
    // we still keep the pipeline running by deriving a sortable day surrogate.
    let s = extract_string(array, row)?;
    let mut parts = s.split('-');
    let year = parts.next()?.parse::<i32>().ok()?;
    let month = parts.next()?.parse::<i32>().ok()?;
    let day = parts.next().unwrap_or("01").parse::<i32>().ok()?;
    let approx_days = (year - 1970) * 365 + (month - 1) * 30 + (day - 1);
    Some((approx_days, format!("{:04}-{:02}-01", year, month)))
}

fn build_group_key(group_by: GroupBy, billing: &str, servicing: &str, hcpcs: &str, month: &str) -> String {
    match group_by {
        GroupBy::BillingNpi => billing.to_string(),
        GroupBy::ServicingNpi => servicing.to_string(),
        GroupBy::Hcpcs => hcpcs.to_string(),
        GroupBy::Month => month.to_string(),
        GroupBy::BillingNpiHcpcs => format!("{billing}|{hcpcs}"),
    }
}

// -----------------------------------------------------------------------------
// load_rows
// -----------------------------------------------------------------------------
// Reads parquet in batches, projects only needed columns, filters rows, and
// builds compact integer group IDs.
//
// Why `group_id` exists:
// - strings are expensive to carry through large numeric loops
// - integer IDs are GPU/CPU cache-friendly and faster to hash/sort
//
// Return value:
// - `rows`: dense row structs used by report assembly
// - `group_lookup`: reverse map (group_id -> human-readable group key)
fn load_rows(path: &Path, min_claims: u32, group_by: GroupBy) -> Result<(Vec<InputRow>, Vec<String>)> {
    let file = fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;

    let mut builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("failed to build parquet reader for {}", path.display()))?;
    builder = builder.with_batch_size(65_536);

    let schema = builder.schema().clone();

    let idx_billing = schema
        .index_of("BILLING_PROVIDER_NPI_NUM")
        .context("missing BILLING_PROVIDER_NPI_NUM")?;
    let idx_servicing = schema
        .index_of("SERVICING_PROVIDER_NPI_NUM")
        .context("missing SERVICING_PROVIDER_NPI_NUM")?;
    let idx_hcpcs = schema.index_of("HCPCS_CODE").context("missing HCPCS_CODE")?;
    let idx_month = schema
        .index_of("CLAIM_FROM_MONTH")
        .context("missing CLAIM_FROM_MONTH")?;
    let idx_benef = schema
        .index_of("TOTAL_UNIQUE_BENEFICIARIES")
        .context("missing TOTAL_UNIQUE_BENEFICIARIES")?;
    let idx_claims = schema.index_of("TOTAL_CLAIMS").context("missing TOTAL_CLAIMS")?;
    let idx_paid = schema.index_of("TOTAL_PAID").context("missing TOTAL_PAID")?;

    let mut reader = builder.build().context("failed to create parquet batch reader")?;

    let mut rows = Vec::new();
    let mut group_map = HashMap::<String, u32>::new();
    let mut group_lookup = Vec::<String>::new();

    for batch_result in &mut reader {
        let batch = batch_result.context("failed to read parquet batch")?;
        let billing_col = batch.column(idx_billing);
        let servicing_col = batch.column(idx_servicing);
        let hcpcs_col = batch.column(idx_hcpcs);
        let month_col = batch.column(idx_month);
        let benef_col = batch.column(idx_benef);
        let claims_col = batch.column(idx_claims);
        let paid_col = batch.column(idx_paid);

        for i in 0..batch.num_rows() {
            let billing = match extract_string(billing_col.as_ref(), i) {
                Some(v) => v,
                None => continue,
            };
            let servicing = match extract_string(servicing_col.as_ref(), i) {
                Some(v) => v,
                None => continue,
            };
            let hcpcs = match extract_string(hcpcs_col.as_ref(), i) {
                Some(v) => v,
                None => continue,
            };
            let (month_day, month_label) = match extract_month(month_col.as_ref(), i) {
                Some(v) => v,
                None => continue,
            };
            let beneficiaries = match extract_u32(benef_col.as_ref(), i) {
                Some(v) => v,
                None => continue,
            };
            let claims = match extract_u32(claims_col.as_ref(), i) {
                Some(v) => v,
                None => continue,
            };
            if claims < min_claims {
                continue;
            }
            let paid = match extract_f32(paid_col.as_ref(), i) {
                Some(v) => v,
                None => continue,
            };

            let group_key = build_group_key(group_by, &billing, &servicing, &hcpcs, &month_label);
            let group_id = if let Some(existing) = group_map.get(&group_key) {
                *existing
            } else {
                let next = group_lookup.len() as u32;
                group_lookup.push(group_key.clone());
                group_map.insert(group_key, next);
                next
            };

            rows.push(InputRow {
                hcpcs,
                month_label,
                month_day,
                beneficiaries,
                claims,
                paid,
                group_id,
            });
        }
    }

    Ok((rows, group_lookup))
}

fn stats(values: &[f32]) -> DistributionStats {
    // Distribution summary used by z-score and MAD-based anomaly kernels.
    // Equations:
    //   mean = (1/N) * sum_i x_i
    //   std  = sqrt((1/N) * sum_i (x_i - mean)^2)
    //   median = middle(sorted(x))
    //   MAD = median(|x_i - median|)
    if values.is_empty() {
        return DistributionStats::default();
    }

    let mean = values.iter().copied().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f32>()
        / values.len() as f32;
    let std_dev = variance.sqrt().max(1e-6);

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let median = sorted[sorted.len() / 2];

    let mut abs_dev: Vec<f32> = sorted.iter().map(|v| (v - median).abs()).collect();
    abs_dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let mad = abs_dev[abs_dev.len() / 2].max(1e-6);

    DistributionStats {
        mean,
        std_dev,
        median,
        mad,
    }
}

fn percentile(values: &[f32], pct: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let rank = ((pct / 100.0) * (sorted.len() as f32 - 1.0)).round() as usize;
    sorted[rank.min(sorted.len().saturating_sub(1))]
}

// -----------------------------------------------------------------------------
// compute_features
// -----------------------------------------------------------------------------
// Converts raw row fields into numeric features consumed by kernels.
//
// Feature families:
// 1) Base level: paid, claims, beneficiaries, paid_per_claim
// 2) Temporal: mom_ratio, mom_abs_delta, drift_sigma
// 3) Rarity/context: rarity_weight, distance_score
//
// Design choice:
// - compute expensive feature engineering once on CPU
// - keep GPU kernels simple, branch-light, and per-row stateless
fn compute_features(rows: &[InputRow]) -> FeatureMatrix {
    // Build shared feature vectors once so kernels can reuse them.
    // This keeps GPU kernels simple and makes formulas explicit.
    let n = rows.len();
    let mut paid = vec![0.0f32; n];
    let mut claims = vec![0.0f32; n];
    let mut beneficiaries = vec![0.0f32; n];
    let mut paid_per_claim = vec![0.0f32; n];
    let mut mom_ratio = vec![0.0f32; n];
    let mut mom_abs_delta = vec![0.0f32; n];
    let mut drift_sigma = vec![0.0f32; n];
    let mut rarity_weight = vec![0.0f32; n];
    let mut distance_score = vec![0.0f32; n];

    // First pass: base columns and paid-per-claim ratio.
    for (i, row) in rows.iter().enumerate() {
        paid[i] = row.paid;
        claims[i] = row.claims as f32;
        beneficiaries[i] = row.beneficiaries as f32;
        paid_per_claim[i] = if row.claims == 0 {
            0.0
        } else {
            row.paid / row.claims as f32
        };
    }

    // Month-over-month and drift are group-dependent, so we sort indices by
    // (group_id, month_day) and then walk each group in temporal order.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|a, b| {
        rows[*a]
            .group_id
            .cmp(&rows[*b].group_id)
            .then(rows[*a].month_day.cmp(&rows[*b].month_day))
    });

    let mut running: HashMap<u32, (f32, f32, u32, f32)> = HashMap::new();
    for &idx in &indices {
        let row = &rows[idx];
        let entry = running.entry(row.group_id).or_insert((0.0, 0.0, 0, 0.0));
        let prev_paid = entry.3;

        if prev_paid > 0.0 {
            // Month-over-month ratio:
            //   ratio = current_paid / previous_paid
            // and absolute delta:
            //   delta = |current_paid - previous_paid|
            mom_ratio[idx] = row.paid / prev_paid;
            mom_abs_delta[idx] = (row.paid - prev_paid).abs();
        }

        // Drift sigma compares current paid against rolling mean/std of the
        // same group (before adding this row into baseline).
        if entry.2 >= 3 {
            let mean = entry.0 / entry.2 as f32;
            let variance = (entry.1 / entry.2 as f32) - mean * mean;
            let std = variance.max(1e-6).sqrt();
            drift_sigma[idx] = (row.paid - mean) / std;
        }

        entry.0 += row.paid;
        entry.1 += row.paid * row.paid;
        entry.2 += 1;
        entry.3 = row.paid;
    }

    // HCPCS rarity: inverse frequency so uncommon codes get larger weight.
    let mut hcpcs_freq = HashMap::<&str, u32>::new();
    for row in rows {
        *hcpcs_freq.entry(row.hcpcs.as_str()).or_insert(0) += 1;
    }
    for (i, row) in rows.iter().enumerate() {
        let freq = *hcpcs_freq.get(row.hcpcs.as_str()).unwrap_or(&1) as f32;
        rarity_weight[i] = 1.0 / freq.max(1.0);
    }

    // Distance outlier feature uses normalized coordinates.
    let paid_norm = normalize(&paid);
    let claims_norm = normalize(&claims);
    let bene_norm = normalize(&beneficiaries);
    let ppc_norm = normalize(&paid_per_claim);

    let mean_paid = paid_norm.iter().sum::<f32>() / n.max(1) as f32;
    let mean_claims = claims_norm.iter().sum::<f32>() / n.max(1) as f32;
    let mean_bene = bene_norm.iter().sum::<f32>() / n.max(1) as f32;
    let mean_ppc = ppc_norm.iter().sum::<f32>() / n.max(1) as f32;

    for i in 0..n {
        let d0 = paid_norm[i] - mean_paid;
        let d1 = claims_norm[i] - mean_claims;
        let d2 = bene_norm[i] - mean_bene;
        let d3 = ppc_norm[i] - mean_ppc;
        distance_score[i] = (d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3).sqrt();
    }

    FeatureMatrix {
        paid,
        claims,
        beneficiaries,
        paid_per_claim,
        mom_ratio,
        drift_sigma,
        rarity_weight,
        distance_score,
        mom_abs_delta,
    }
}

fn normalize(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let min = values
        .iter()
        .fold(f32::INFINITY, |acc, v| if *v < acc { *v } else { acc });
    let max = values
        .iter()
        .fold(f32::NEG_INFINITY, |acc, v| if *v > acc { *v } else { acc });
    let span = (max - min).max(1e-6);
    values.iter().map(|v| (*v - min) / span).collect()
}

impl GpuContext {
    fn new() -> Result<Self> {
        let device = Device::system_default().context("no Metal device found")?;
        let command_queue = device.new_command_queue();

        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(SHADER_SOURCE, &options)
            .map_err(|e| anyhow::anyhow!("failed to compile Metal source: {e}"))?;

        let mut pipelines = BTreeMap::new();
        for id in 1u8..=8u8 {
            let name = format!("kernel_{id}");
            let func = library
                .get_function(&name, None)
                .map_err(|e| anyhow::anyhow!("failed to find function {name}: {e}"))?;
            let state = device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| anyhow::anyhow!("failed to create pipeline for {name}: {e}"))?;
            pipelines.insert(id, state);
        }

        Ok(Self {
            device,
            command_queue,
            pipelines,
        })
    }
}

fn gpu_scores(
    gpu: &GpuContext,
    kernel: KernelId,
    features: &FeatureMatrix,
    params_template: KernelParams,
    chunk_rows: usize,
) -> Result<Vec<f32>> {
    // This function dispatches exactly one selected kernel over all rows.
    // If row count N exceeds chunk_rows, we issue multiple dispatches.
    //
    // Dispatch model:
    // - each logical thread handles exactly one row in current chunk
    // - kernel uses `offset` to read global row index
    // - output buffer stores contiguous scores for current chunk only
    let n = features.paid.len();
    let mut output = vec![0.0f32; n];

    let byte_len = (n * std::mem::size_of::<f32>()) as u64;

    // Allocate shared buffers once per kernel run. Offsets are handled through
    // KernelParams.offset so each dispatch can process a contiguous chunk.
    let paid_buf = gpu.device.new_buffer_with_data(
        features.paid.as_ptr() as *const _,
        byte_len,
        MTLResourceOptions::StorageModeShared,
    );
    let claims_buf = gpu.device.new_buffer_with_data(
        features.claims.as_ptr() as *const _,
        byte_len,
        MTLResourceOptions::StorageModeShared,
    );
    let bene_buf = gpu.device.new_buffer_with_data(
        features.beneficiaries.as_ptr() as *const _,
        byte_len,
        MTLResourceOptions::StorageModeShared,
    );
    let ppc_buf = gpu.device.new_buffer_with_data(
        features.paid_per_claim.as_ptr() as *const _,
        byte_len,
        MTLResourceOptions::StorageModeShared,
    );
    let mom_buf = gpu.device.new_buffer_with_data(
        features.mom_ratio.as_ptr() as *const _,
        byte_len,
        MTLResourceOptions::StorageModeShared,
    );
    let drift_buf = gpu.device.new_buffer_with_data(
        features.drift_sigma.as_ptr() as *const _,
        byte_len,
        MTLResourceOptions::StorageModeShared,
    );
    let rarity_buf = gpu.device.new_buffer_with_data(
        features.rarity_weight.as_ptr() as *const _,
        byte_len,
        MTLResourceOptions::StorageModeShared,
    );
    let distance_buf = gpu.device.new_buffer_with_data(
        features.distance_score.as_ptr() as *const _,
        byte_len,
        MTLResourceOptions::StorageModeShared,
    );
    let mom_abs_buf = gpu.device.new_buffer_with_data(
        features.mom_abs_delta.as_ptr() as *const _,
        byte_len,
        MTLResourceOptions::StorageModeShared,
    );

    let pipeline = gpu
        .pipelines
        .get(&kernel.0)
        .with_context(|| format!("missing pipeline for kernel {}", kernel.0))?;

    let mut offset = 0usize;
    while offset < n {
        let len = (n - offset).min(chunk_rows.max(1));

        let mut params = params_template;
        params.count = len as u32;
        params.offset = offset as u32;

        let params_buf = gpu.device.new_buffer_with_data(
            (&params as *const KernelParams).cast(),
            std::mem::size_of::<KernelParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = gpu.device.new_buffer(
            (len * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = gpu.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        // Buffer index mapping must match lesson10.metal exactly:
        // 0 paid, 1 claims, 2 beneficiaries, 3 ppc, 4 mom_ratio,
        // 5 drift_sigma, 6 rarity_weight, 7 distance_score,
        // 8 mom_abs_delta, 9 KernelParams, 10 output.
        encoder.set_buffer(0, Some(&paid_buf), 0);
        encoder.set_buffer(1, Some(&claims_buf), 0);
        encoder.set_buffer(2, Some(&bene_buf), 0);
        encoder.set_buffer(3, Some(&ppc_buf), 0);
        encoder.set_buffer(4, Some(&mom_buf), 0);
        encoder.set_buffer(5, Some(&drift_buf), 0);
        encoder.set_buffer(6, Some(&rarity_buf), 0);
        encoder.set_buffer(7, Some(&distance_buf), 0);
        encoder.set_buffer(8, Some(&mom_abs_buf), 0);
        encoder.set_buffer(9, Some(&params_buf), 0);
        encoder.set_buffer(10, Some(&out_buf), 0);

        let tg = MTLSize {
            width: THREADGROUP_SIZE,
            height: 1,
            depth: 1,
        };
        let grid = MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(grid, tg);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let ptr = out_buf.contents() as *const f32;
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        output[offset..offset + len].copy_from_slice(slice);

        offset += len;
    }

    Ok(output)
}

fn cpu_score_for_kernel(id: KernelId, idx: usize, features: &FeatureMatrix, paid_stats: DistributionStats, ppc_stats: DistributionStats) -> f32 {
    match id.0 {
        1 => features.paid[idx],
        // z-score anomaly on paid.
        // z = |(x - mean) / std|
        2 => ((features.paid[idx] - paid_stats.mean) / paid_stats.std_dev.max(1e-6)).abs(),
        // Modified z-score with MAD.
        // mz = 0.6745 * |(x - median) / MAD|
        3 => (0.6745 * (features.paid[idx] - paid_stats.median).abs() / paid_stats.mad.max(1e-6)).abs(),
        4 => {
            // Same modified-z idea on paid-per-claim feature.
            (0.6745 * (features.paid_per_claim[idx] - ppc_stats.median).abs() / ppc_stats.mad.max(1e-6)).abs()
        }
        5 => features.mom_ratio[idx],
        6 => features.drift_sigma[idx].abs(),
        7 => features.paid[idx] * features.rarity_weight[idx],
        8 => features.distance_score[idx],
        _ => 0.0,
    }
}

fn validate_gpu_cpu(
    mode: ValidationMode,
    gpu_scores: &[f32],
    cpu_scores: &[f32],
) -> usize {
    if mode == ValidationMode::Off {
        return 0;
    }

    let indices: Vec<usize> = if mode == ValidationMode::Full {
        (0..gpu_scores.len()).collect()
    } else {
        if gpu_scores.is_empty() {
            return 0;
        }
        vec![0, gpu_scores.len() / 2, gpu_scores.len().saturating_sub(1)]
    };

    let mut mismatches = 0usize;
    for i in indices {
        let diff = (gpu_scores[i] - cpu_scores[i]).abs();
        if diff > 1e-3 {
            mismatches += 1;
        }
    }
    mismatches
}

fn top_spenders_report(
    rows: &[InputRow],
    group_lookup: &[String],
    top_k: usize,
    timings: KernelTimings,
    mismatches: usize,
) -> KernelReport {
    // Important: kernel 1 ranking result is based on grouped aggregation from
    // raw rows (sum paid/claims/beneficiaries by group_id), not per-row score
    // sorting. This matches business question "top spenders by group totals".
    #[derive(Default)]
    struct Agg {
        paid: f64,
        claims: u64,
        beneficiaries: u64,
    }

    let mut agg = HashMap::<u32, Agg>::new();
    for row in rows {
        let e = agg.entry(row.group_id).or_default();
        e.paid += row.paid as f64;
        e.claims += row.claims as u64;
        e.beneficiaries += row.beneficiaries as u64;
    }

    let mut items: Vec<(u32, Agg)> = agg.into_iter().collect();
    items.sort_by(|a, b| {
        b.1.paid
            .partial_cmp(&a.1.paid)
            .unwrap_or(Ordering::Equal)
            .then(group_lookup[a.0 as usize].cmp(&group_lookup[b.0 as usize]))
    });

    let top = items
        .into_iter()
        .take(top_k)
        .map(|(gid, a)| OutputEntry {
            group_key: group_lookup[gid as usize].clone(),
            month: "ALL".to_string(),
            paid: a.paid,
            claims: a.claims,
            beneficiaries: a.beneficiaries,
            score: a.paid,
            reason: "Kernel 1 total paid ranking".to_string(),
        })
        .collect::<Vec<_>>();

    KernelReport {
        kernel_id: 1,
        kernel_name: kernel_name(KernelId(1)).to_string(),
        gpu_ms: timings.gpu_ms,
        cpu_ms: timings.cpu_ms,
        validation_ms: timings.validation_ms,
        kernel_total_ms: timings.gpu_ms + timings.cpu_ms + timings.validation_ms,
        threshold: 0.0,
        rows_flagged: top.len(),
        top,
        validation_mismatches: mismatches,
    }
}

fn anomaly_report(
    id: KernelId,
    rows: &[InputRow],
    group_lookup: &[String],
    features: &FeatureMatrix,
    scores: &[f32],
    cfg: &CliConfig,
    timings: KernelTimings,
    mismatches: usize,
) -> KernelReport {
    // Threshold selection mirrors CLI defaults:
    // - kernels 2..6 use explicit numeric thresholds
    // - kernel 7 uses percentile cutoff
    // - kernel 8 uses top-percent distance cutoff
    let threshold = match id.0 {
        2 => cfg.k2_threshold,
        3 => cfg.k3_threshold,
        4 => cfg.k4_threshold,
        5 => cfg.k5_ratio_threshold,
        6 => cfg.k6_threshold,
        7 => percentile(scores, cfg.k7_percentile),
        8 => percentile(scores, 100.0 - cfg.k8_top_percent),
        _ => 0.0,
    };

    let mut candidates: Vec<OutputEntry> = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        let score = scores[i];
        let pass = match id.0 {
            5 => score >= threshold && score.is_finite() && features.mom_abs_delta[i] >= cfg.k5_abs_floor,
            _ => score >= threshold && score.is_finite(),
        };

        if pass {
            candidates.push(OutputEntry {
                group_key: group_lookup[row.group_id as usize].clone(),
                month: row.month_label.clone(),
                paid: row.paid as f64,
                claims: row.claims as u64,
                beneficiaries: row.beneficiaries as u64,
                score: score as f64,
                reason: format!("Kernel {} exceeded threshold {:.4}", id.0, threshold),
            });
        }
    }

    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then(a.group_key.cmp(&b.group_key))
            .then(a.month.cmp(&b.month))
    });

    let rows_flagged = candidates.len();
    let top = candidates.into_iter().take(cfg.top_k).collect::<Vec<_>>();

    KernelReport {
        kernel_id: id.0,
        kernel_name: kernel_name(id).to_string(),
        gpu_ms: timings.gpu_ms,
        cpu_ms: timings.cpu_ms,
        validation_ms: timings.validation_ms,
        kernel_total_ms: timings.gpu_ms + timings.cpu_ms + timings.validation_ms,
        threshold: threshold as f64,
        rows_flagged,
        top,
        validation_mismatches: mismatches,
    }
}

// -----------------------------------------------------------------------------
// run
// -----------------------------------------------------------------------------
// End-to-end orchestrator:
// 1) load parquet rows
// 2) decide whether features/GPU are needed
// 3) execute selected kernels
// 4) optionally run CPU compare and validation
// 5) build and print report
//
// Performance insight:
// - kernel 1 report itself is CPU group aggregation
// - kernel 1 GPU scoring is useful only for compare/validation
// - therefore we include a fast-path that skips unnecessary scoring work
//   when `--compare-cpu off --validate off`.
fn run(cfg: CliConfig) -> Result<()> {
    if cfg.list_kernels {
        print_kernel_catalog();
        return Ok(());
    }

    let started_total = Instant::now();

    let metadata = fs::metadata(&cfg.input)
        .with_context(|| format!("failed to stat input file {}", cfg.input.display()))?;

    println!("Loading rows from {}", cfg.input.display());
    let load_started = Instant::now();
    let (rows, group_lookup) = load_rows(&cfg.input, cfg.min_claims, cfg.group_by)?;
    println!(
        "Loaded {} rows across {} groups in {:.3}s",
        rows.len(),
        group_lookup.len(),
        load_started.elapsed().as_secs_f64()
    );

    if rows.is_empty() {
        bail!("no rows remain after filtering; adjust --min-claims or input file")
    }

    let kernel1_requested = cfg.kernels.iter().any(|k| *k == KernelId(1));
    let non_kernel1_requested = cfg.kernels.iter().any(|k| *k != KernelId(1));
    let kernel1_needs_scores = kernel1_requested
        && (cfg.compare_cpu == CompareCpu::On || cfg.validate != ValidationMode::Off);

    // Performance fast-path:
    // - Kernel 1 final report is built from grouped CPU rows, not from GPU scores.
    // - If CPU compare and validation are both off, kernel 1 does not need score vectors.
    let need_features = non_kernel1_requested || kernel1_needs_scores;
    let need_gpu = non_kernel1_requested || kernel1_needs_scores;

    let (features_opt, paid_stats, ppc_stats) = if need_features {
        println!("Building shared feature vectors...");
        let features = compute_features(&rows);
        let paid_stats = stats(&features.paid);
        let ppc_stats = stats(&features.paid_per_claim);
        (Some(features), paid_stats, ppc_stats)
    } else {
        (
            None,
            DistributionStats::default(),
            DistributionStats::default(),
        )
    };

    let gpu_opt = if need_gpu {
        println!("Creating Metal pipelines...");
        Some(GpuContext::new()?)
    } else {
        None
    };

    println!(
        "Dispatch mode={} memory_fraction={:.2} chunk_rows={}",
        match cfg.mode {
            Mode::Single => "single",
            Mode::Double => "double",
        },
        cfg.memory_fraction,
        cfg.chunk_rows
    );

    let mut reports = Vec::<KernelReport>::new();

    for kernel in &cfg.kernels {
        println!("Running kernel {} ({})...", kernel.0, kernel_name(*kernel));

        if *kernel == KernelId(1) && !kernel1_needs_scores {
            println!(
                "Kernel 1 fast-path: skipping per-row GPU/CPU scoring; using grouped CPU aggregation."
            );
            reports.push(top_spenders_report(
                &rows,
                &group_lookup,
                cfg.top_k,
                KernelTimings::default(),
                0,
            ));
            continue;
        }

        let features = features_opt
            .as_ref()
            .context("internal error: missing features for requested kernel path")?;
        let gpu = gpu_opt
            .as_ref()
            .context("internal error: missing GPU context for requested kernel path")?;

        let params = KernelParams {
            count: 0,
            offset: 0,
            mean: paid_stats.mean,
            std_dev: paid_stats.std_dev,
            median: paid_stats.median,
            mad: paid_stats.mad,
            aux_a: ppc_stats.median,
            aux_b: ppc_stats.mad,
        };

        let gpu_started = Instant::now();
        let gpu_out = gpu_scores(gpu, *kernel, features, params, cfg.chunk_rows)?;
        let gpu_ms = gpu_started.elapsed().as_secs_f64() * 1000.0;

        // CPU reference scoring is optional and mainly used for verification.
        let cpu_started = Instant::now();
        let cpu_out = if cfg.compare_cpu == CompareCpu::On || cfg.validate != ValidationMode::Off {
            (0..rows.len())
                .map(|i| cpu_score_for_kernel(*kernel, i, features, paid_stats, ppc_stats))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let cpu_ms = if cpu_out.is_empty() {
            0.0
        } else {
            cpu_started.elapsed().as_secs_f64() * 1000.0
        };

        let validation_started = Instant::now();
        let mismatches = if cpu_out.is_empty() {
            0
        } else {
            validate_gpu_cpu(cfg.validate, &gpu_out, &cpu_out)
        };
        let validation_ms = if cfg.validate == ValidationMode::Off || cpu_out.is_empty() {
            0.0
        } else {
            validation_started.elapsed().as_secs_f64() * 1000.0
        };

        let timings = KernelTimings {
            gpu_ms,
            cpu_ms,
            validation_ms,
        };

        if *kernel == KernelId(1) {
            reports.push(top_spenders_report(
                &rows,
                &group_lookup,
                cfg.top_k,
                timings,
                mismatches,
            ));
        } else {
            reports.push(anomaly_report(
                *kernel,
                &rows,
                &group_lookup,
                features,
                &gpu_out,
                &cfg,
                timings,
                mismatches,
            ));
        }
    }

    let total_duration_ms = started_total.elapsed().as_secs_f64() * 1000.0;

    let selected_kernels = cfg.kernels.iter().map(|k| k.0).collect::<Vec<_>>();
    let report = RunReport {
        input_path: cfg.input.display().to_string(),
        input_size_bytes: metadata.len(),
        selected_kernels,
        row_count: rows.len(),
        group_count: group_lookup.len(),
        total_duration_ms,
        reports,
    };

    println!();
    println!("=== Run Summary ===");
    println!("Rows:          {}", report.row_count);
    println!("Groups:        {}", report.group_count);
    println!("Kernels:       {:?}", report.selected_kernels);
    println!("Total runtime: {:.3} ms", report.total_duration_ms);

    for kernel_report in &report.reports {
        println!();
        println!(
            "Kernel {} ({}) | gpu {:.3} ms | cpu {:.3} ms | validate {:.3} ms | total {:.3} ms | threshold {:.4} | flagged {} | mismatches {}",
            kernel_report.kernel_id,
            kernel_report.kernel_name,
            kernel_report.gpu_ms,
            kernel_report.cpu_ms,
            kernel_report.validation_ms,
            kernel_report.kernel_total_ms,
            kernel_report.threshold,
            kernel_report.rows_flagged,
            kernel_report.validation_mismatches,
        );
        for (idx, row) in kernel_report.top.iter().take(cfg.top_k).enumerate() {
            println!(
                "  #{:02} group={} month={} paid={:.2} claims={} bene={} score={:.4}",
                idx + 1,
                row.group_key,
                row.month,
                row.paid,
                row.claims,
                row.beneficiaries,
                row.score,
            );
        }
    }

    if let Some(path) = cfg.output_json {
        let payload = serde_json::to_vec_pretty(&report).context("failed to serialize JSON report")?;
        fs::write(&path, payload)
            .with_context(|| format!("failed to write output JSON {}", path.display()))?;
        println!();
        println!("JSON report written to {}", path.display());
    }

    Ok(())
}

fn main() -> Result<()> {
    let cfg = parse_args()?;
    run(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Date32Array, Float64Array, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::arrow_writer::ArrowWriter;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    fn write_fixture() -> Result<NamedTempFile> {
        let tmp = NamedTempFile::new().context("failed to create temp file")?;

        let schema = Arc::new(Schema::new(vec![
            Field::new("BILLING_PROVIDER_NPI_NUM", DataType::Utf8, false),
            Field::new("SERVICING_PROVIDER_NPI_NUM", DataType::Utf8, false),
            Field::new("HCPCS_CODE", DataType::Utf8, false),
            Field::new("CLAIM_FROM_MONTH", DataType::Date32, false),
            Field::new("TOTAL_UNIQUE_BENEFICIARIES", DataType::Int64, false),
            Field::new("TOTAL_CLAIMS", DataType::Int64, false),
            Field::new("TOTAL_PAID", DataType::Float64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["111", "111", "222", "333", "333", "333"])),
                Arc::new(StringArray::from(vec!["555", "555", "666", "777", "777", "777"])),
                Arc::new(StringArray::from(vec!["A100", "A100", "B200", "C300", "C300", "C300"])),
                Arc::new(Date32Array::from(vec![18_000, 18_030, 18_000, 18_000, 18_030, 18_060])),
                Arc::new(Int64Array::from(vec![10, 12, 15, 20, 25, 30])),
                Arc::new(Int64Array::from(vec![15, 20, 30, 40, 45, 55])),
                Arc::new(Float64Array::from(vec![5000.0, 22000.0, 3500.0, 8000.0, 9000.0, 40000.0])),
            ],
        )
        .context("failed to build fixture record batch")?;

        let file = fs::File::create(tmp.path()).context("failed to open fixture path for write")?;
        let mut writer = ArrowWriter::try_new(file, schema, None).context("failed to create ArrowWriter")?;
        writer.write(&batch).context("failed to write fixture batch")?;
        writer.close().context("failed to close fixture writer")?;

        Ok(tmp)
    }

    #[test]
    fn parse_kernel_ids_happy_path() -> Result<()> {
        let ids = parse_kernel_ids("1,3,8")?;
        assert_eq!(ids, vec![KernelId(1), KernelId(3), KernelId(8)]);
        Ok(())
    }

    #[test]
    fn parse_kernel_ids_reject_duplicates() {
        let err = parse_kernel_ids("1,1").unwrap_err().to_string();
        assert!(err.contains("duplicated"));
    }

    #[test]
    fn parse_kernel_ids_reject_out_of_range() {
        let err = parse_kernel_ids("9").unwrap_err().to_string();
        assert!(err.contains("out of range"));
    }

    #[test]
    fn percentile_works() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&values, 50.0), 3.0);
        assert_eq!(percentile(&values, 100.0), 5.0);
    }

    #[test]
    fn top_spenders_deterministic_tie_break() {
        let rows = vec![
            InputRow {
                hcpcs: "x".into(),
                month_label: "2020-01-01".into(),
                month_day: 1,
                beneficiaries: 1,
                claims: 12,
                paid: 100.0,
                group_id: 0,
            },
            InputRow {
                hcpcs: "x".into(),
                month_label: "2020-01-01".into(),
                month_day: 1,
                beneficiaries: 1,
                claims: 12,
                paid: 100.0,
                group_id: 1,
            },
        ];
        let lookup = vec!["A".to_string(), "B".to_string()];
        let report = top_spenders_report(
            &rows,
            &lookup,
            2,
            KernelTimings {
                gpu_ms: 0.0,
                cpu_ms: 0.0,
                validation_ms: 0.0,
            },
            0,
        );
        assert_eq!(report.top[0].group_key, "A");
        assert_eq!(report.top[1].group_key, "B");
    }

    #[test]
    fn fixture_loads_and_gpu_cpu_parity_for_all_kernels() -> Result<()> {
        let tmp = write_fixture()?;
        let (rows, _) = load_rows(tmp.path(), 12, GroupBy::BillingNpi)?;
        assert!(!rows.is_empty());

        // The test should still pass on machines without Metal by focusing on
        // parquet load correctness. GPU parity executes only when GPU exists.
        let maybe_gpu = GpuContext::new();
        if maybe_gpu.is_err() {
            return Ok(());
        }
        let gpu = maybe_gpu?;

        let features = compute_features(&rows);
        let paid_stats = stats(&features.paid);
        let ppc_stats = stats(&features.paid_per_claim);
        let params = KernelParams {
            count: 0,
            offset: 0,
            mean: paid_stats.mean,
            std_dev: paid_stats.std_dev,
            median: paid_stats.median,
            mad: paid_stats.mad,
            aux_a: ppc_stats.median,
            aux_b: ppc_stats.mad,
        };

        for id in 1u8..=8u8 {
            let gpu_out = gpu_scores(&gpu, KernelId(id), &features, params, 1024)?;
            for i in 0..rows.len() {
                let cpu = cpu_score_for_kernel(KernelId(id), i, &features, paid_stats, ppc_stats);
                let diff = (gpu_out[i] - cpu).abs();
                assert!(diff <= 1e-3, "kernel {} mismatch at {} diff {}", id, i, diff);
            }
        }

        Ok(())
    }
}
