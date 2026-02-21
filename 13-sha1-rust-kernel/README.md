# Lesson 13: SHA1 Brute Force with Rust + wgpu

This lesson ports lesson 12's Metal SHA1 brute-force kernel to `wgpu`/WGSL while
keeping the same host driver behavior and CLI pattern.

## Run

```bash
cargo run --release -p sha1-rust-kernel -- --hash <40hex>
```

Example (find `"a"` in `lower`, length 1):

```bash
cargo run -p sha1-rust-kernel -- \
  --hash 86f7e437faa5a7fce15d1ddcb9eaeaea377667b8 \
  --charset lower \
  --min-len 1 \
  --max-len 1 \
  --mode first \
  --validation full
```

## Notes

- Candidate length is limited to 55 bytes (single SHA1 block path).
- Runtime requires adapter support for `wgpu::Features::SHADER_INT64`.

## Benchmark: Metal vs WGSL (same algorithm, same knobs)

The numbers below compare:
- Metal lesson: `sha1-brute-forcing` (lesson 12)
- WGSL lesson: `sha1-rust-kernel` (lesson 13)

Both were run with the same workload and launch knobs:

```bash
--hash ffffffffffffffffffffffffffffffffffffffff \
--charset lowernum \
--min-len 5 --max-len 5 \
--mode first \
--validation gpu-only \
--threads-per-group 256 \
--candidates-per-thread 8 \
--verbose
```

Reproduction commands from workspace root:

```bash
./target/release/sha1-brute-forcing \
  --hash ffffffffffffffffffffffffffffffffffffffff \
  --charset lowernum \
  --min-len 5 --max-len 5 \
  --mode first \
  --validation gpu-only \
  --threads-per-group 256 \
  --candidates-per-thread 8 \
  --verbose

./target/release/sha1-rust-kernel \
  --hash ffffffffffffffffffffffffffffffffffffffff \
  --charset lowernum \
  --min-len 5 --max-len 5 \
  --mode first \
  --validation gpu-only \
  --threads-per-group 256 \
  --candidates-per-thread 8 \
  --verbose
```

Result summary (same machine, 5 runs each):

| Metric | Metal (lesson 12) | WGSL/wgpu (lesson 13) |
|---|---:|---:|
| 5-run avg overall MH/s | 1013.28 | 678.64 |
| 5-run avg wall ms | 60.01 | 89.11 |
| Steady-state avg MH/s (runs 2-5) | 1049.04 | 683.28 |

On this machine and workload, the Metal path is faster by about `1.53x` in steady-state throughput.

## Why Metal is faster here

- The Metal host path uses shared buffers and direct completion wait (`wait_until_completed`) with immediate reads from shared memory.
- The wgpu path performs extra readback staging work each dispatch (`copy_buffer_to_buffer`), then waits (`device.poll`), then maps/reads buffers.
- WGSL path can also incur additional backend/runtime overhead versus native MSL in integer-heavy kernels.

This comparison is specific to this hardware, runtime versions, and workload settings; it is not a universal rule that Metal always beats WGSL.

## Metal to WGSL kernel mapping

The kernels are intentionally parallel in structure:

| Concept | Metal lesson 12 | WGSL lesson 13 | Notes |
|---|---|---|---|
| Kernel params contract | `KernelParams` in `src/sha1_brute_force.metal` | `KernelParams` in `src/sha1_brute_force.wgsl` | Same logical fields and meaning. |
| Charset mapping | `map_lower`, `map_lowernum`, `map_printable` | `map_lower`, `map_lowernum`, `map_printable` | Same index-to-candidate idea. |
| SHA1 block path | `sha1_one_block` | `sha1_one_block` | Same single-block SHA1 flow and ring-buffer schedule. |
| Early-exit in first mode | `found_flag` checks | `found_flag` checks | Same short-circuit behavior. |
| First-match write | `record_first` | `record_first` | Atomic winner writes `found_index`. |
| All-match collection | `record_all` | `record_all` | Atomic append with `max_matches` cap. |
| Grid stepping | `base`, `step`, `search_space` | `base`, `step`, `search_space` | Same strided traversal of search space. |

Key semantic differences:
- Metal uses native `ulong` divide/modulo in mapping helpers.
- WGSL uses a custom `divmod_u64_by_u32` helper to keep the same behavior portable in shader codegen paths.
- Metal uses `[[buffer(n)]]` resource bindings; WGSL uses `@group(0) @binding(n)`.

## Reading guide for annotated WGSL

Read `/Users/jonathan/Development/gpuprogramming/13-sha1-rust-kernel/src/sha1_brute_force.wgsl` in this order:

1. Buffer contracts and bindings (`KernelParams`, found/match buffers, `@group/@binding`).
2. Utility functions (`rotl`, `load_be_u32`, `sha1_f`, `sha1_k`, `divmod_u64_by_u32`).
3. Candidate mappers (`map_lower`, `map_lowernum`, `map_printable`).
4. `sha1_one_block` for padding, schedule generation, and 80 rounds.
5. Atomic record helpers (`record_first`, `record_all`).
6. `sha1_brute_force` dispatch loop (`base/step` striding, early exit, candidate loop).
