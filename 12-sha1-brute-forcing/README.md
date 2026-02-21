# Lesson 12: SHA1 Brute Forcing on the GPU (Rust + Metal)

This lesson brute-forces a **single** SHA-1 hash using **Metal compute**.

It is written in the same teaching-first style as earlier lessons, with explicit
host/kernel contracts and validation modes.

## Ethics and Authorization

Only use this on hashes you are authorized to test (CTFs, your own data, or
explicit permission). This is a hash-cracking technique.

## What You Will Learn

- How to map an integer search space `0..radix^len` onto candidate strings.
- How to implement SHA-1 (single-block path) on the GPU.
- How to use one kernel with runtime charset selection:
  - `lower`
  - `lowernum`
  - `printable`
- How to tune throughput:
  - `--threads-per-group`
  - `--candidates-per-thread`
  - length ramp (`--min-len..--max-len`)
- How to validate correctness:
  - `gpu-only`, `spot`, `full`

## Quick Start

### 1) Build

From repo root:

```bash
cargo build --release -p sha1-brute-forcing
```

### 2) Crack a tiny known hash

SHA1("abc") is `a9993e364706816aba3e25717850c26c9cd0d89d`.

```bash
cargo run --release -p sha1-brute-forcing -- \
  --hash a9993e364706816aba3e25717850c26c9cd0d89d \
  --charset lower \
  --min-len 3 --max-len 3 \
  --mode first \
  --validation spot \
  --verbose
```

## How It Works

### 1) Search space

Choose charset radix `R` and length `L`, then candidates are:

`N = R^L`

Radix values:

- `lower` => 26
- `lowernum` => 36
- `printable` => 95

### 2) Host-side length ramp

The kernel takes a single `len` per dispatch.

`--min-len` and `--max-len` are implemented on the host by dispatching the
same kernel once per length in the range.

### 3) One-kernel charset selection

This lesson uses one kernel entrypoint (`sha1_brute_force`) and passes
`alphabet_id` in `KernelParams`:

- `0` => lower
- `1` => lowernum
- `2` => printable

### 4) SHA-1 details that must match exactly

- Digest compare is 5 words (`H0..H4`), not 4.
- Digest words are interpreted as big-endian words.
- SHA-1 padding writes message bit length as big-endian u64 in the last 8 bytes.

Any mismatch here (especially endianness) causes silent misses.

## Validation Modes

- `gpu-only`: fastest, no CPU checks.
- `spot`: validate GPU hits + deterministic sample.
- `full`: CPU mirror of all candidates (slowest, best for debugging).

## CLI

```text
--hash <40hex>                 Target SHA1 in 40 hex characters
--charset lower|lowernum|printable
--min-len <u32>
--max-len <u32>
--mode first|all
--validation gpu-only|spot|full
--threads-per-group <u32>
--candidates-per-thread <u32>
--progress-ms <u64>
--max-matches <u32>
--json <path>
--verbose
```

## Files to Read

- `/Users/jonathan/Development/gpuprogramming/12-sha1-brute-forcing/src/main.rs`
- `/Users/jonathan/Development/gpuprogramming/12-sha1-brute-forcing/src/sha1_brute_force.metal`
