# Lesson 02: Double Values -- Your First Compute Kernel

## What You'll Learn

This is the "Hello World" of GPU compute. You'll take an array of floats, send
it to the GPU, have a Metal kernel double every element in parallel, and read
the results back. This lesson walks through the **entire Metal compute pipeline**
end-to-end.

## The Metal Compute Pipeline

Every Metal compute program follows the same 10-step flow. Once you understand
this, every future lesson is just a variation:

```
1. Device           -- get the GPU
2. Library          -- compile shader source code
3. Function         -- pick a kernel function from the library
4. Pipeline State   -- build the GPU's execution plan
5. Command Queue    -- create a queue for submitting work
6. Command Buffer   -- create a single-use container for commands
7. Compute Encoder  -- record commands (set pipeline, bind buffers, dispatch)
8. Buffers          -- GPU-accessible memory with your data
9. Dispatch         -- tell the GPU how many threads to launch
10. Commit & Wait   -- submit and block until done
```

Think of it like a restaurant:

| Metal concept | Restaurant analogy |
|---------------|-------------------|
| Device | The kitchen (GPU hardware) |
| Library | The cookbook (compiled shaders) |
| Function | A specific recipe |
| Pipeline State | The kitchen's prep for that recipe |
| Command Queue | The order window |
| Command Buffer | A ticket with one or more orders |
| Compute Encoder | The waiter writing down the order |
| Buffers | The ingredients and plates |
| Dispatch | "Make 1024 servings, please" |
| Commit | Hanging the ticket in the window |

## Key Concepts

### Metal Shading Language (MSL)

The GPU code is written in MSL, which is based on C++14. In this lesson the
shader source is embedded as a Rust string constant and compiled at runtime
with `device.new_library_with_source()`.

### The `kernel` keyword

Marks a function as a compute kernel (as opposed to a vertex/fragment shader
used for rendering).

### `[[ buffer(N) ]]`

Attribute that binds a shader parameter to buffer slot N. On the Rust side,
`encoder.set_buffer(N, ...)` plugs a buffer into that slot.

### `[[ thread_position_in_grid ]]`

Each GPU thread gets a unique ID via this attribute. When you dispatch 1024
threads, each gets a `gid` from 0 to 1023 -- this is how each thread knows
which element to work on.

### `StorageModeShared`

Both CPU and GPU can access the buffer. On Apple Silicon with unified memory,
this is zero-copy -- the same physical RAM is visible to both processors.

### `dispatch_threads` vs `dispatch_thread_groups`

- **`dispatch_threads`**: You specify the total number of threads. Metal handles
  the threadgroup math for you. Simpler.
- **`dispatch_thread_groups`**: You specify the number of groups and threads per
  group. You handle edge cases yourself.

### `thread_execution_width()`

Returns the GPU's SIMD width (32 on Apple Silicon). This is how many threads
execute in lockstep. A good default threadgroup size.

## The Shader

```metal
kernel void double_values(
    device float *input  [[ buffer(0) ]],
    device float *output [[ buffer(1) ]],
    uint gid             [[ thread_position_in_grid ]]
) {
    output[gid] = input[gid] * 2.0;
}
```

Each thread doubles exactly one element. Thread 0 handles `input[0]`, thread 1
handles `input[1]`, and so on -- all 1024 threads run simultaneously.

## Run

```bash
cargo run -p double-values
```

## Expected Output

```
Using GPU: Apple M4 Max

First 10 results:
  input[0] = 1.0  -->  output[0] = 2.0
  input[1] = 2.0  -->  output[1] = 4.0
  input[2] = 3.0  -->  output[2] = 6.0
  ...
  input[9] = 10.0  -->  output[9] = 20.0

All 1024 results correct: true

Congratulations! You just ran your first Metal compute kernel!
The GPU doubled all 1024 values in parallel.
```

## Next

[Lesson 03: Vector Add](../03-vector-add/) -- multiple input buffers and bounds checking.
