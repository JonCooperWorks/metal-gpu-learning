# Lesson 03: Vector Add -- Multiple Input Buffers

## What You'll Learn

Building on the full pipeline from Lesson 02, this lesson introduces a common
pattern: passing **multiple input buffers** to a single kernel. We add two
arrays element-wise (`C[i] = A[i] + B[i]`) and also handle a non-power-of-2
array size, which requires bounds checking in the shader.

## New Concepts

### Multiple Buffer Bindings

Buffer indices are numbered slots on the GPU. Your shader declares which slot
each parameter lives in, and the Rust side plugs buffers into matching slots:

```
Rust side:                       GPU shader side:
encoder.set_buffer(0, &a)   --> device float *a [[ buffer(0) ]]
encoder.set_buffer(1, &b)   --> device float *b [[ buffer(1) ]]
encoder.set_buffer(2, &c)   --> device float *c [[ buffer(2) ]]
encoder.set_buffer(3, &n)   --> device uint  &n [[ buffer(3) ]]
```

### Passing Scalar Parameters

To pass a single value (like the array count) to a shader, you can either:

1. **Create a buffer** with that value (what we do here -- explicit and clear)
2. **Use `encoder.set_bytes()`** to pass small data inline (more convenient, but
   limited to 4KB)

### Bounds Checking in the Shader

When dispatching threads, the count is often rounded up to a multiple of the
threadgroup size (e.g., 32). If your array has 1000 elements, Metal may launch
1024 threads. Without a bounds check, threads 1000-1023 would read/write out of
bounds:

```metal
if (gid >= count) {
    return;  // Extra threads do nothing
}
```

### `const` Qualifier in MSL

Adding `device const float *a` tells the compiler (and readers) that this
buffer is read-only. This can enable optimizations and makes intent clear.

## The Shader

```metal
kernel void vector_add(
    device const float *a     [[ buffer(0) ]],
    device const float *b     [[ buffer(1) ]],
    device float       *c     [[ buffer(2) ]],
    device const uint  &count [[ buffer(3) ]],
    uint gid                  [[ thread_position_in_grid ]]
) {
    if (gid >= count) {
        return;
    }
    c[gid] = a[gid] + b[gid];
}
```

## Run

```bash
cargo run -p vector-add
```

## Expected Output

```
Using GPU: Apple M4 Max

Vector Addition: C[i] = A[i] + B[i]

    i      A[i]      B[i]      C[i]
-----------------------------------
    0       0.0    1000.0    1000.0
    1       1.0     999.0    1000.0
    2       2.0     998.0    1000.0
  ...
  999     999.0       1.0    1000.0

All 1000 results equal 1000.0: true

The GPU added two 1000-element vectors in parallel!
Notice how the non-power-of-2 size (1000) worked fine
thanks to the bounds check in the shader.
```

## Things to Try

- Change `num_elements` to 1 or 7 -- does it still work with tiny sizes?
- Remove the `if (gid >= count)` check in the shader and use a non-power-of-2
  size. What happens? (Hint: undefined behavior -- you might get wrong results,
  a crash, or it might seem fine but corrupt memory silently.)
- Try making A and B different lengths -- what would you need to change?

## Next

[Lesson 04: Parallel Sum](../04-parallel-sum/) -- atomic operations and parallel reduction.
