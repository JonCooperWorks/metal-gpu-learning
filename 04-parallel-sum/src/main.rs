// =============================================================================
// LESSON 4: Parallel Sum -- Atomic Operations on the GPU
// =============================================================================
//
// In the previous lessons, each GPU thread worked on its own independent piece
// of data (embarrassingly parallel). But what if threads need to combine their
// results into a single value? This is called a REDUCTION.
//
// The simplest example: summing an array. We want one final number, but we have
// thousands of threads all computing partial results simultaneously.
//
// THE PROBLEM:
//   If 1000 threads all try to do `sum += data[gid]` at the same time, they'll
//   corrupt the result. Thread A reads sum=5, Thread B reads sum=5, both add
//   their values, both write back -- one update is lost! This is a DATA RACE.
//
// THE SOLUTION: ATOMIC OPERATIONS
//   An atomic operation is guaranteed to complete without interruption. When
//   Thread A does an atomic_add, no other thread can read/modify the same
//   memory location until Thread A's operation finishes. The hardware enforces
//   this at the memory controller level.
//
//   Trade-off: Atomics are MUCH slower than regular memory access because
//   threads have to take turns. For a proper high-performance reduction, you'd
//   use a multi-pass approach with threadgroup shared memory. But atomics are
//   the right starting point for understanding the concept.
//
// NEW CONCEPTS:
//   - `atomic_uint` / `atomic_int` -- atomic types in MSL
//   - `atomic_fetch_add_explicit()` -- atomically adds a value
//   - `memory_order_relaxed` -- the simplest memory ordering (sufficient here)
//   - `volatile` -- prevents the compiler from optimizing away our writes
//   - CPU-side verification of GPU results
//
// RUN: cargo run -p parallel-sum
// =============================================================================

use metal::*;
use objc::rc::autoreleasepool;
use std::mem;

// =============================================================================
// THE METAL SHADER
// =============================================================================
const SHADER_SOURCE: &str = r#"
    #include <metal_stdlib>
    using namespace metal;

    // This kernel sums all elements of an array into a single output value.
    //
    // ATOMIC TYPES:
    //   `volatile device atomic_uint *sum`
    //
    //   Let's break this down piece by piece:
    //
    //   - `device`    -- lives in device (GPU) memory
    //   - `volatile`  -- tells the compiler "other threads may modify this at
    //                     any time, so don't cache it in a register or optimize
    //                     away reads/writes"
    //   - `atomic_uint` -- an unsigned int that supports atomic operations.
    //                      You CAN'T use regular operators (+, =) on it.
    //                      You MUST use atomic_* functions.
    //   - `*sum`      -- it's a pointer to a single atomic_uint
    //
    // MEMORY ORDER:
    //   `memory_order_relaxed` means we only guarantee atomicity -- we don't
    //   make any guarantees about the ORDER in which different threads' atomic
    //   operations become visible. This is fine for a simple sum because
    //   addition is commutative (order doesn't matter: 1+2+3 == 3+1+2).
    //
    //   Stronger orderings (acquire, release, seq_cst) are needed when you're
    //   using atomics for synchronization (e.g., implementing a lock). We don't
    //   need those here.
    //
    kernel void parallel_sum(
        device const uint         *data  [[ buffer(0) ]],
        volatile device atomic_uint *sum   [[ buffer(1) ]],
        device const uint         &count [[ buffer(2) ]],
        uint gid                         [[ thread_position_in_grid ]]
    ) {
        // Bounds check (same pattern as Lesson 3)
        if (gid >= count) {
            return;
        }

        // Atomically add data[gid] to sum.
        //
        // What happens under the hood:
        //   1. The hardware locks the memory location holding *sum
        //   2. Reads the current value of *sum
        //   3. Adds data[gid] to it
        //   4. Writes the new value back
        //   5. Unlocks the memory location
        //
        // All of this happens as one indivisible operation. No other thread
        // can sneak in between steps 2 and 4.
        //
        // The return value (which we ignore) is the OLD value of *sum before
        // the addition. This is useful in some algorithms but not here.
        atomic_fetch_add_explicit(sum, data[gid], memory_order_relaxed);
    }
"#;

fn main() {
    autoreleasepool(|| {
        let device = Device::system_default().expect("No Metal-capable GPU found!");
        println!("Using GPU: {}\n", device.name());

        // =====================================================================
        // Compile shader & create pipeline
        // =====================================================================
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(SHADER_SOURCE, &options)
            .expect("Failed to compile shader");
        let kernel = library
            .get_function("parallel_sum", None)
            .expect("Couldn't find 'parallel_sum' function");
        let pipeline_state = device
            .new_compute_pipeline_state_with_function(&kernel)
            .expect("Failed to create pipeline state");

        // =====================================================================
        // Prepare input data
        // =====================================================================
        //
        // Let's sum the numbers 1 through N. The expected result is N*(N+1)/2,
        // which gives us an easy way to verify correctness.
        let num_elements: usize = 10_000;
        let input_data: Vec<u32> = (1..=num_elements as u32).collect();

        // Expected sum using the formula: 1 + 2 + ... + N = N*(N+1)/2
        let expected_sum: u64 = (num_elements as u64) * (num_elements as u64 + 1) / 2;

        println!("Summing integers 1 through {} on the GPU", num_elements);
        println!("Expected result: {}\n", expected_sum);

        // =====================================================================
        // Create Metal buffers
        // =====================================================================

        // Input buffer: our array of integers
        let input_buffer = device.new_buffer_with_data(
            input_data.as_ptr() as *const _,
            (num_elements * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Sum buffer: a single u32, initialized to 0.
        // This is where the atomic accumulation happens.
        //
        // IMPORTANT: We must initialize it to 0! If we just allocate without
        // initializing, it could contain garbage data and our sum would be wrong.
        let sum_value: u32 = 0;
        let sum_buffer = device.new_buffer_with_data(
            &sum_value as *const u32 as *const _,
            mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Count buffer
        let count = num_elements as u32;
        let count_buffer = device.new_buffer_with_data(
            &count as *const u32 as *const _,
            mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // =====================================================================
        // Encode and dispatch
        // =====================================================================
        let command_queue = device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(&input_buffer), 0);   // buffer(0) = data
        encoder.set_buffer(1, Some(&sum_buffer), 0);     // buffer(1) = sum (atomic)
        encoder.set_buffer(2, Some(&count_buffer), 0);   // buffer(2) = count

        let grid_size = MTLSize::new(num_elements as u64, 1, 1);
        let threadgroup_size = MTLSize::new(pipeline_state.thread_execution_width(), 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // =====================================================================
        // Read back the result
        // =====================================================================
        //
        // The sum buffer contains a single u32 with the accumulated result.
        let gpu_sum = unsafe { *(sum_buffer.contents() as *const u32) };

        println!("GPU result:      {}", gpu_sum);
        println!("Expected result: {}", expected_sum);

        // NOTE: We're using u32 for the sum, which can hold values up to
        // ~4.29 billion. For N=10000, the sum is 50,005,000 which fits fine.
        // For larger N, you'd need u64 or a multi-pass approach.
        if gpu_sum as u64 == expected_sum {
            println!("\nResult matches! The GPU correctly summed {} values.", num_elements);
        } else {
            println!("\nMISMATCH! Something went wrong.");
            println!("This could happen if:");
            println!("  - The sum overflowed u32 (try smaller N)");
            println!("  - The sum buffer wasn't initialized to 0");
        }

        // =====================================================================
        // Let's also verify with the CPU for comparison
        // =====================================================================
        let cpu_sum: u64 = input_data.iter().map(|&x| x as u64).sum();
        println!("\nCPU verification: {}", cpu_sum);
        println!(
            "CPU matches GPU:  {}",
            cpu_sum == gpu_sum as u64
        );

        // =====================================================================
        // PERFORMANCE NOTE
        // =====================================================================
        //
        // This atomic-based approach works correctly but is NOT the fastest way
        // to do a parallel sum. Because every thread is atomically updating the
        // same memory location, they effectively serialize -- defeating the
        // purpose of parallelism!
        //
        // A faster approach is a TREE REDUCTION:
        //   Pass 1: Each threadgroup sums its chunk into threadgroup memory
        //   Pass 2: Sum the partial sums
        //
        // This uses `threadgroup` (shared) memory and barrier synchronization,
        // which would be a great topic for a future lesson!
        println!("\n--- NOTE ---");
        println!("Atomic operations serialize access, making this approach");
        println!("simple but slow. A tree reduction with threadgroup shared");
        println!("memory would be much faster for large arrays.");
    });
}
