# Lesson 01: Hello GPU -- Device Discovery

## What You'll Learn

This is the absolute starting point for Metal programming. Before running any
code on the GPU, you need to **find it**. This lesson connects to your Mac's GPU
and queries its properties -- no shaders, no compute, just "hello, GPU."

## Key Concepts

### MTLDevice

The `MTLDevice` is the central object in Metal. It represents a single GPU and
is your entry point for everything:

- Creating buffers (GPU memory)
- Compiling shaders
- Building compute/render pipelines
- Creating command queues

You get one by calling `Device::system_default()`, which returns the system's
preferred GPU. On Apple Silicon (M1/M2/M3/M4), this is the unified GPU built
into the chip.

### autoreleasepool

Metal's Rust bindings (`metal-rs`) talk to Objective-C under the hood. Objective-C
uses reference counting for memory, and many Metal API calls return "autoreleased"
objects. Wrapping your code in `autoreleasepool(|| { ... })` ensures those objects
are properly cleaned up. In a GUI app the run loop handles this, but in a CLI app
you need to do it yourself.

## Properties Queried

| Property | What it tells you |
|----------|-------------------|
| `name()` | Human-readable GPU name (e.g., "Apple M4 Max") |
| `registry_id()` | Unique system I/O registry identifier |
| `max_buffer_length()` | Largest single buffer you can allocate |
| `max_threads_per_threadgroup()` | Max threads in one threadgroup (width x height x depth) |
| `is_low_power()` | True for integrated GPUs (false on Apple Silicon) |
| `has_unified_memory()` | True when CPU and GPU share physical RAM (always true on Apple Silicon) |

## Run

```bash
cargo run -p hello-gpu
```

## Expected Output

```
=== Metal Device Info ===

Device name:            Apple M4 Max
Registry ID:            4294968485
Max buffer length:      39813 MB
Max threads/threadgroup: 1024 x 1024 x 1024 (= 1073741824 total)
Low power:              false
Unified memory:         true

=== GPU is ready! ===
In the next lesson, we'll actually run code on it.
```

## Next

[Lesson 02: Double Values](../02-double-values/) -- your first compute kernel.
