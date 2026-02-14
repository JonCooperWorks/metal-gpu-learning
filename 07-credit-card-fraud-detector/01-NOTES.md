# 01 Notes - Credit Card Fraud GPU Lesson

## What We Built
- This project is **not training** a model.
- It is a **GPU-accelerated inference + metrics pipeline** on synthetic transactions.
- Pipeline shape:
  1. Generate synthetic transaction features + synthetic ground-truth labels.
  2. Score each transaction with a fixed rule formula.
  3. Classify via threshold (`score >= threshold`).
  4. Aggregate metrics (TP/FP/TN/FN, precision/recall/FPR/accuracy, histogram).
  5. Compare GPU throughput vs CPU baseline.

## Mental Model (Map/Reduce)
- `score_transactions` kernel is like a **map**:
  - one GPU thread handles one transaction index (`gid`)
  - reads feature columns, computes score, writes one output row
- `reduce_metrics` kernel is like a **reduce**:
  - threads aggregate many output rows into partial metrics per threadgroup
  - CPU folds partials into global totals

## Buffers and Indexing
- Buffers are bound by slot/index (`set_buffer(0..N, ...)`).
- The per-row offset is mostly implicit via thread id (`gid`), e.g. `amounts[gid]`.
- Yes, kernels mutate output buffers (and shared threadgroup memory in reduction).

## Why GPU Can Be Faster
- Work is regular and parallel per transaction.
- Reductions are associative (sum/count/histogram), so easy to parallelize.
- Double buffering overlaps CPU fill with GPU compute.
- GPU wins more as batch size gets large enough to amortize dispatch/sync overhead.

## Why CPU Can Still Win
- Small batches can be overhead-dominated.
- Highly branchy/divergent logic can hurt GPU efficiency.
- Frequent host-device synchronization reduces GPU advantage.

## Single vs Double Mode
- `single`: fill -> submit -> wait (simple, less overlap)
- `double`: ping-pong slots to overlap CPU and GPU work (higher throughput when balanced)

## Real Processor Analogy
- Typical production setup:
  - **Online path (CPU/low-latency):** immediate auth decision in milliseconds.
  - **Batch/async path (often GPU-friendly):** deeper retrospective checks, graph analysis, monitoring, reporting, and policy/model updates.

## Validation Modes
- `--validate off`: fastest, no chunk correctness checks.
- `--validate spot`: first + periodic + final chunk checks.
- `--validate full`: every chunk checked.

### Spot Overhead Intuition
- Overhead scales with number of validated chunks, not total rows directly.
- For your run with **932 chunks** and `progress-interval=10`, spot validates about **12 chunks**.
- That is around **~1.5-2%** runtime overhead in your measured regime.

## Cost Intuition
- If GPU spend is roughly linear with runtime, then disabling spot validation can save about that same runtime percentage.
- Example: at $1,000,000/year, **~1.5-2%** is roughly **$15k-$20k/year**.
- Production caution: removing validation entirely increases silent-regression risk.

## Key Intuition You Built
- Think in terms of:
  - **Per-row parallel scoring (map)**
  - **Parallel metric aggregation (reduce)**
  - **Chunking + overlap + validation tradeoffs**
  - **Latency path vs throughput path**
