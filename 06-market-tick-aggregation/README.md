# Lesson 06: Market Tick Aggregation -- Real GPU Data Pipeline

## Real Problem Statement

A common market-data backend job is to ingest a huge stream of `(price, size)`
ticks and emit fast summary metrics for risk dashboards, monitoring, and
pre-trade analytics:

- total traded volume
- total notional
- VWAP (volume-weighted average price)
- return volatility proxy
- min/max observed price

This lesson implements that pipeline with two GPU stages and a CPU baseline.

## Why Not Just CPU?

CPU is often simpler and can be faster for small batches. GPU helps when:

- work per tick is regular (same math for each element)
- batches are large enough to amortize dispatch/sync overhead
- reduction is associative and parallelizable
- you can overlap CPU fill and GPU compute

GPU may lose when:

- batch size is small
- logic is branch-heavy/irregular
- frequent host-device synchronization is required
- memory transfer/dispatch overhead dominates useful compute

This lesson prints both GPU and CPU throughput so you can see the crossover.

## Pipeline Shape

Stage A (`transform_ticks`):
1. Read `price` and `size`
2. Compute `notional = price * size`
3. Compute deterministic `return_proxy = abs(price - baseline_price(index))`
4. Write derived struct per tick

Stage B (`reduce_partials`):
1. Reduce derived structs into threadgroup-level partial aggregates
2. Output one partial aggregate per threadgroup
3. CPU folds partials into final global metrics

This threadgroup-reduction + CPU fold pattern avoids a single global atomic
hotspot and scales much better than every thread atomically touching one value.

## Modes

- `--mode single`: one slot, wait every chunk (baseline)
- `--mode double`: two slots, overlap CPU fill with GPU work (default)

## CLI Flags

- `--mode single|double` (default: `double`)
- `--total-ticks <u64>` (default: `5000000000`)
- `--chunk-ticks <u64>` (optional override; auto-sized if omitted)
- `--memory-fraction <f64>` (default: `0.85`, used for auto chunking)
- `--progress-interval <1..100>` (default: `10`)
- `--validate full|spot|off` (default: `spot`)
- `--compare-cpu on|off` (default: `on`)

## Run

```bash
cargo run --release -p market-tick-aggregation
cargo run --release -p market-tick-aggregation -- --mode single
cargo run --release -p market-tick-aggregation -- --total-ticks 1000000 --chunk-ticks 250000 --validate full
cargo run --release -p market-tick-aggregation -- --compare-cpu off
```

## Interpreting Benchmark Output

You get:

- GPU wall time, CPU fill time, GPU submit/wait time
- GPU effective ticks/sec
- CPU baseline ticks/sec (optional)
- speedup ratio (`cpu_time / gpu_time`)

Interpretation guidance:

- If double mode is faster than single, overlap is effective.
- If CPU wins, your batch is likely too small or too sync-heavy.
- If GPU wins, arithmetic intensity and batch size likely exceeded overhead.

## Precision Note

This lesson uses `f32` for tutorial simplicity and throughput. Production finance
systems often require `f64` or fixed-point math for strict accounting precision.
