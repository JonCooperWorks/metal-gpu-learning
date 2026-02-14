# Lesson 07: Credit Card Fraud Detector -- GPU Rules Engine

## Real Problem Statement

Payment systems often need to score huge batches of card transactions and emit
operational fraud metrics in near real time:

- per-transaction fraud flags
- confusion matrix (TP/FP/TN/FN)
- precision/recall/FPR/accuracy
- score distribution histograms for threshold tuning

This lesson implements a production-shaped pipeline for that workflow.

## Pipeline Diagram

1. Stage A (`score_transactions`):
   - read transaction feature columns
   - compute weighted risk score
   - classify `predicted_fraud = score >= threshold`
2. Stage B (`reduce_metrics`):
   - reduce per-transaction outcomes into per-threadgroup partials
   - write partial confusion counts + score sums + histogram slices
3. CPU fold:
   - combine partials into final batch metrics

## Why GPU vs CPU

GPU helps when:

- scoring logic is regular and parallel per transaction
- reductions are associative (sum/count/histogram)
- batch sizes are large enough to amortize dispatch/sync overhead
- double buffering can overlap CPU fill and GPU execution

CPU can win when:

- batches are small
- logic is highly branchy/divergent
- host-device synchronization is frequent

The lesson prints CPU baseline throughput so crossover points are visible.

## Synthetic Labels and Leakage Avoidance

- Transaction features are deterministic synthetic data.
- Ground-truth labels come from a **hidden rule** distinct from the scoring
  formula to avoid metric leakage.
- `--fraud-rate-boost` scales label prevalence for experimentation.

## CLI

- `--mode single|double` (default: `double`)
- `--total-transactions <u64>` (default: `2000000000`)
- `--chunk-transactions <u64>` (optional override; auto if omitted)
- `--memory-fraction <f64>` (default: `0.85`)
- `--progress-interval <1..100>` (default: `10`)
- `--validate full|spot|off` (default: `spot`)
- `--compare-cpu on|off` (default: `on`)
- `--threshold <f32>` (default: `0.72`)
- `--score-bins <u32>` (default: `20`, max: `64`)
- `--fraud-rate-boost <f32>` (default: `1.0`)

## Run

```bash
cargo run --release -p credit-card-fraud-detector
cargo run --release -p credit-card-fraud-detector -- --mode single
cargo run --release -p credit-card-fraud-detector -- --total-transactions 1000000 --chunk-transactions 250000 --validate full
cargo run --release -p credit-card-fraud-detector -- --threshold 0.80 --score-bins 32
```

## Interpreting Metrics

- Raising threshold often increases precision and lowers recall.
- Lowering threshold often increases recall and false positives.
- Histogram shape helps identify if score separation is strong or weak.
- If `double` mode does not improve throughput, chunk size may be too small or
  synchronization/launch overhead may dominate.

## Compliance Note

This is educational synthetic data and rule logic. It is not a production fraud
policy, not a calibrated model, and not suitable for compliance decisions.
