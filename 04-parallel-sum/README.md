# Lesson 04: Parallel Sum -- Atomic Operations

## What You'll Learn

In the previous lessons, every GPU thread worked independently -- no thread needed
another thread's result. This is called "embarrassingly parallel." But what happens
when threads need to **combine their results into a single value**? This is a
**reduction**, and it introduces the problem of data races on the GPU.

This lesson sums an array of integers using **atomic operations** -- the simplest
(though not the fastest) way to safely combine results from thousands of parallel
threads.

## The Problem: Data Races

If 1000 threads all try to do `sum += data[gid]` at the same time:

```
Thread A: reads sum = 5
Thread B: reads sum = 5        (before A writes!)
Thread A: writes sum = 5 + 3 = 8
Thread B: writes sum = 5 + 7 = 12   (overwrites A's result!)
```

Thread A's addition is lost. The final result is wrong.

## The Solution: Atomic Operations

An atomic operation is **indivisible** -- the hardware guarantees that no other
thread can read or modify the memory location while the operation is in progress:

```
Thread A: atomic_add(sum, 3)   --> sum goes from 5 to 8, no interruption
Thread B: atomic_add(sum, 7)   --> sum goes from 8 to 15, no interruption
```

## New Concepts

### `atomic_uint` / `atomic_int`

Special types in MSL that only support atomic operations. You **cannot** use
regular operators (`+`, `=`, `+=`) on them -- you must use `atomic_*` functions.

### `atomic_fetch_add_explicit()`

Atomically adds a value and returns the **old** value (before the addition).
The three arguments are:

1. Pointer to the atomic variable
2. The value to add
3. The memory ordering

### `memory_order_relaxed`

The weakest (and fastest) memory ordering. It guarantees **atomicity** but not
**ordering** -- different threads might see updates in different orders. This is
fine for a sum because addition is commutative (`1+2+3 == 3+1+2`).

Stronger orderings (`acquire`, `release`, `seq_cst`) are needed when atomics are
used for synchronization (locks, flags). Not necessary here.

### `volatile`

Tells the compiler "other threads may change this at any time." Without it, the
compiler might cache the value in a register and miss updates from other threads.

## The Shader

```metal
kernel void parallel_sum(
    device const uint         *data  [[ buffer(0) ]],
    volatile device atomic_uint *sum   [[ buffer(1) ]],
    device const uint         &count [[ buffer(2) ]],
    uint gid                         [[ thread_position_in_grid ]]
) {
    if (gid >= count) {
        return;
    }
    atomic_fetch_add_explicit(sum, data[gid], memory_order_relaxed);
}
```

## Run

```bash
cargo run -p parallel-sum
```

## Expected Output

```
Using GPU: Apple M4 Max

Summing integers 1 through 10000 on the GPU
Expected result: 50005000

GPU result:      50005000
Expected result: 50005000

Result matches! The GPU correctly summed 10000 values.

CPU verification: 50005000
CPU matches GPU:  true
```

## Performance Note

This atomic approach is **correct but slow**. Every thread atomically updates the
same memory location, which means they effectively serialize -- defeating the
purpose of parallelism. For 10,000 elements it's fine, but for millions it would
be a bottleneck.

The fast way to do a parallel reduction is a **tree reduction**:

```
Pass 1:  [1, 2, 3, 4, 5, 6, 7, 8]
              ↓         ↓
Pass 2:  [3,    7,    11,   15   ]
              ↓         ↓
Pass 3:  [10,        26         ]
              ↓
Pass 4:  [36                    ]
```

Each pass halves the data, using **threadgroup shared memory** and **barriers**
for synchronization within a threadgroup. This would be a great topic for a
future lesson.

## Things to Try

- Increase `num_elements` to 100,000 or 1,000,000. Does it still work? Watch
  out for `u32` overflow (~4.29 billion max).
- Change the input data to all 1s -- the expected sum is just N. Easy to verify.
- What happens if you forget to initialize the sum buffer to 0?
- Try removing the `volatile` keyword. Does it still produce correct results?
  (It might -- `volatile` mostly matters for correctness under optimization.)

## What's Next?

You now know the fundamentals of Metal GPU compute:

- Device discovery and properties
- The full compute pipeline (library, function, pipeline state, command queue,
  command buffer, encoder)
- Buffer creation and binding
- Thread dispatch and grid sizing
- Bounds checking for non-power-of-2 data
- Atomic operations for thread communication

Future lessons could cover:

- **Threadgroup shared memory** and barriers (fast reductions)
- **2D dispatch** for image processing or matrix operations
- **Multiple compute passes** (output of one kernel feeds into the next)
- **Performance measurement** with GPU timestamps
- **Precompiled shaders** (.metallib files and build scripts)
