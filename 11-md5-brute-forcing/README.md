# Lesson 11: MD5 Brute Forcing on the GPU (Rust + Metal)

This lesson brute-forces a **single** MD5 hash using **Metal compute**.

It is intentionally written like a tutorial: heavy annotation in both the Rust
host code and the Metal kernel code.

## Ethics and Authorization

Only use this on hashes you are authorized to test (CTFs, your own data, or
explicit permission). This is a hash-cracking technique.

## What You Will Learn

- How to map an integer search space `0..radix^len` onto candidate strings.
- How to implement MD5 on the GPU (single 512-bit block path).
- How to tune throughput:
  - `--threads-per-group`
  - `--candidates-per-thread`
  - charset selection (`--charset`)
  - length ramp (`--min-len..--max-len`)
- How to validate correctness without destroying performance:
  - `gpu-only`, `spot`, `full`

## Quick Start

### 1) Build

From repo root:

```bash
cargo build --release -p md5-brute-forcing
```

### 2) Crack a tiny known hash

MD5("a") is `0cc175b9c0f1b6a831c399e269772661`.

```bash
cargo run --release -p md5-brute-forcing -- \
  --hash 0cc175b9c0f1b6a831c399e269772661 \
  --charset lower \
  --min-len 1 --max-len 1 \
  --mode first \
  --validation spot \
  --verbose
```

### 3) A slightly bigger demo

If you want to try to find "abc123" you can compute its MD5 first:

```bash
printf 'abc123' | md5
```

Then run with `--charset lowernum --max-len 6`.

## How It Works

### 1) The brute-force search space

Choose:

- charset with radix `R`:
  - `lower` => 26
  - `lowernum` => 36
  - `printable` => 95
- length `L`

Total candidates at that length:

`N = R^L`

This explodes quickly. For example:

- `lowernum`, L=6 => 36^6 = 2,176,782,336 candidates

### 2) Mapping index -> candidate string

Each candidate string is derived from an integer index `idx` in base `R`:

- digit0 = idx % R
- idx /= R
- digit1 = idx % R
- ...

Then each digit becomes a byte (depending on charset).

This mapping is implemented in both:

- Rust: `index_to_candidate` in `src/main.rs`
- Metal: `map_lower`, `map_lowernum`, `map_printable` in `src/lesson11.metal`

### 3) MD5 on the GPU (single-block)

MD5 processes 512-bit blocks.

If the message length is <= 55 bytes, padding works like:

- append `0x80`
- pad zeros until byte 56
- write message length in bits as little-endian u64 at bytes 56..63

Then one compression function produces a 128-bit digest (A,B,C,D).

This lesson implements only that fast one-block path, so:

- `--max-len` is limited to 55.

### 4) GPU parallelism strategy

We dispatch a grid of threads. Each thread tests **many** candidates:

- grid-stride outer loop spreads work across the whole grid
- inner loop tests `--candidates-per-thread` indices per iteration

That reduces loop overhead and improves ALU utilization.

### 5) Match recording

- `--mode first`:
  - the first thread that finds a hit uses an atomic CAS to claim `found_flag`
  - writes `found_index`
  - other threads early-exit

- `--mode all`:
  - threads atomically append indices to an output array
  - bounded by `--max-matches`

## Validation Modes

- `gpu-only`: fastest, but no correctness checking.
- `spot`: default; recomputes GPU matches on CPU + checks a deterministic sample.
- `full`: recompute every candidate on CPU (very slow, but useful for debugging).

If any validation mismatch is detected, the program exits with an error.

## Performance Tuning Checklist

If you want to push maximum throughput:

1. Use `--validation gpu-only`.
2. Prefer `--mode first` unless you need all matches.
3. Increase `--candidates-per-thread` cautiously (8, 16, 32 are typical).
4. Keep `--threads-per-group` at powers of two (128/256/512) and under the
   device max.
5. Use narrower charsets if your target permits it.

## Files to Read

- `/Users/jonathan/Development/gpuprogramming/11-md5-brute-forcing/src/main.rs`
- `/Users/jonathan/Development/gpuprogramming/11-md5-brute-forcing/src/lesson11.metal`
