# Lesson 05: Double Buffering -- Overlapping CPU and GPU Work

## What You'll Learn

Chunking solved the memory-limit problem from Lesson 02, but chunking with a
single buffer set still leaves performance on the table:

- CPU fills input buffer for chunk N
- GPU processes chunk N
- CPU waits
- Repeat

If CPU fill is slow, the GPU sits idle between chunks.

This lesson uses **double buffering** to overlap both stages:

- Slot A: GPU processes chunk N
- Slot B: CPU fills chunk N+1
- Swap A/B and repeat

## Why This Matters

For large workloads (like 100 billion elements), total time trends toward:

`max(cpu_fill_time, gpu_compute_time)`

instead of:

`cpu_fill_time + gpu_compute_time`

That overlap is the core optimization pattern used in real compute pipelines.

## Modes

- `--mode single`: single input/output buffer set, wait every chunk
- `--mode double`: two buffer slots with overlap (default)

## CLI Flags

- `--mode single|double` (default: `double`)
- `--total-elements <u64>` (default: `100000000000`)
- `--chunk-elements <u64>` (optional override; default is auto-derived from RAM)
- `--memory-fraction <f64>` (default: `0.85`, used only when chunk size is auto)
- `--progress-interval <1..100>` (default: `10`, percentage cadence)

If `--chunk-elements` is not provided, the program:

1. detects system RAM (`hw.memsize`)
2. takes `memory_fraction` of that RAM as a target budget
3. divides by in-flight buffer count (2 for `single`, 4 for `double`)
4. clamps to Metal max buffer size and workload size

## Run

```bash
cargo run --release -p double-buffering
cargo run --release -p double-buffering -- --mode single
cargo run --release -p double-buffering -- --total-elements 10000000000
```

## Interpreting Output

The lesson prints:

- total wall time
- CPU fill time
- GPU submit/wait time
- effective throughput (billion elements/sec)
- estimated memory bandwidth (read + write GB/s)

If `double` mode is not faster, common causes are:

- too-small chunks (pipeline overhead dominates)
- non-optimized build (use `--release`)
- CPU and GPU already balanced for that specific workload

## Validation Strategy

- Full correctness check on chunk 0
- Spot checks (start/middle/end indices) on periodic chunks and final chunk
- Deterministic data generation for reproducible checks

## Next

You now have the core large-data compute pattern:

1. chunking for memory limits
2. pipelining for overlap

Natural extensions are multi-stage pipelines and N-buffer rings.
