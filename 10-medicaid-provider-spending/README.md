# Lesson 10: Medicaid Provider Spending -- Multi-Kernel GPU Analytics

This lesson shows a beginner-friendly analytics pipeline over a real public
Medicaid dataset using Rust + Metal.

## How to Read This Code

Read in this order:

1. `src/main.rs` constants, CLI parsing, and kernel catalog
2. `src/main.rs` parquet load + feature engineering
3. `src/lesson10.metal` kernel math (`kernel_1` .. `kernel_8`)
4. `src/main.rs` GPU dispatch + CPU validation + reporting

## CPU vs GPU Responsibilities

- CPU:
  - parquet ingestion and row filtering
  - key encoding and feature engineering
  - thresholding, ranking, and report formatting
- GPU:
  - per-row scoring kernels (1..8), one score output per row

## Dataset source

- Dataset page: <https://opendata.hhs.gov/datasets/medicaid-provider-spending/>
- Version used in this lesson: `2026-02-09`
- Parquet URL:
  <https://stopendataprod.blob.core.windows.net/datasets/medicaid-provider-spending/2026-02-09/medicaid-provider-spending.parquet>

## Download

```bash
mkdir -p /Users/jonathan/Development/gpuprogramming/10-medicaid-provider-spending/data
curl -L --fail --progress-bar \
  https://stopendataprod.blob.core.windows.net/datasets/medicaid-provider-spending/2026-02-09/medicaid-provider-spending.parquet \
  -o /Users/jonathan/Development/gpuprogramming/10-medicaid-provider-spending/data/medicaid-provider-spending.parquet
```

Verify:

```bash
shasum -a 256 /Users/jonathan/Development/gpuprogramming/10-medicaid-provider-spending/data/medicaid-provider-spending.parquet
```

Expected SHA-256:

`a998e5ae11a391f1eb0d8464b3866a3ee7fe18aa13e56d411c50e72e3a0e35c7`

## Kernel Catalog

| ID | Kernel | Score formula | Primary use | Fast-path notes |
|---|---|---|---|---|
| 1 | Top spenders | `score = paid` | Baseline spending magnitude | If `--compare-cpu off --validate off`, host skips per-row GPU/CPU scoring and directly runs grouped CPU aggregation for top spenders. |
| 2 | Z-score anomaly | `z = |(paid - mean) / std|` | Detect extreme paid values vs global mean/std | Uses `--k2-threshold`. |
| 3 | MAD anomaly | `mz = 0.6745 * |(paid - median) / MAD|` | Robust outlier detection when mean/std are skewed | Uses `--k3-threshold`. |
| 4 | Paid-per-claim MAD anomaly | `mz_ppc = 0.6745 * |(ppc - median_ppc) / MAD_ppc|` | Detect unusually expensive claims | Uses `--k4-threshold`. |
| 5 | Month-over-month spike | `ratio = paid_current / paid_previous` | Detect sudden temporal jumps | Uses `--k5-ratio-threshold` and `--k5-abs-floor`. |
| 6 | Drift sigma anomaly | `score = |drift_sigma|` | Detect deviation from rolling group baseline | Uses `--k6-threshold`. |
| 7 | HCPCS rarity-weighted anomaly | `score = paid * rarity_weight` | Surface high spend in rare procedure patterns | Cutoff from `--k7-percentile`. |
| 8 | Distance outlier | `score = distance_from_normalized_centroid` | Flag multivariate outliers | Cutoff from `--k8-top-percent`. |

## Math Primer

## 1) Z-score

`z = |(x - mean) / std|`

Large `z` means a row is far from average in standard-deviation units.

## 2) Modified z-score with MAD

`mz = 0.6745 * |(x - median) / MAD|`

This is more robust than z-score when outliers already skew the mean/std.

## 3) Paid-per-claim

`ppc = total_paid / total_claims`

Useful for detecting high cost intensity per claim volume.

## 4) Month-over-month spike

`ratio = paid_current / paid_previous`

Combined with absolute delta floor to avoid tiny-denominator artifacts.

## 5) Drift sigma

Compares current row against rolling baseline of its group.
Interpretation is similar to z-score but over temporal baseline.

## 6) Rarity weighting

`score = paid * (1 / frequency(hcpcs))`

Rare procedure codes get higher weight for equal paid amount.

## 7) Distance outlier

Euclidean distance in normalized feature space:

`distance = sqrt(sum_i (x_i - mean_i)^2)`

Rows far from centroid get larger scores.

## Equation -> Code Map

- Feature construction and statistics:
  - `10-medicaid-provider-spending/src/main.rs`
- CPU reference kernel formulas:
  - `10-medicaid-provider-spending/src/main.rs` (`cpu_score_for_kernel`)
- GPU kernel formulas:
  - `10-medicaid-provider-spending/src/lesson10.metal`
- Thresholding and ranking:
  - `10-medicaid-provider-spending/src/main.rs` (`anomaly_report`)

## Run examples

List kernels:

```bash
cargo run --release -p medicaid-provider-spending -- --list-kernels
```

Run all kernels with CPU compare/validation:

```bash
cargo run --release -p medicaid-provider-spending -- \
  --input /Users/jonathan/Development/gpuprogramming/10-medicaid-provider-spending/data/medicaid-provider-spending.parquet \
  --kernels 1,2,3,4,5,6,7,8 \
  --compare-cpu on \
  --validate spot \
  --output-json /Users/jonathan/Development/gpuprogramming/10-medicaid-provider-spending/data/report.json
```

## Glossary

- `parquet`: columnar on-disk data format
- `feature`: numeric input derived from raw columns
- `kernel`: GPU scoring function
- `dispatch`: executing a kernel on GPU threads
- `percentile`: value below which a percentage of scores fall
- `MAD`: median absolute deviation

## Common Mistakes

- Forgetting `--kernels` (required unless `--list-kernels`).
- Treating IDs as provider names (they are identifiers).
- Running percentile thresholds on too-small samples.
- Ignoring `compare-cpu`/`validate` during kernel edits.
- Interpreting anomalies as compliance decisions without domain review.
