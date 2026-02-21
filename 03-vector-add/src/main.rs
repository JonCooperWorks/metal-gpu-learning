// =============================================================================
// LESSON 3: Vector Add -- Multiple Input Buffers
// =============================================================================
//
// Now that you know the full Metal compute pipeline (from Lesson 2), this
// lesson introduces a practical pattern: passing MULTIPLE input buffers to a
// single kernel. We'll add two arrays element-wise: C[i] = A[i] + B[i].
//
// NEW CONCEPTS IN THIS LESSON:
//
//   - Multiple buffer bindings: buffer(0), buffer(1), buffer(2)
//   - Bounds checking in the shader (what happens when thread count > data?)
//   - Passing a scalar parameter via a buffer
//   - A cleaner code structure now that you know the basics
//
// MENTAL MODEL -- Buffer Binding:
//
//   Think of buffer indices as numbered slots on the GPU. Your shader declares
//   which slot each parameter lives in with [[ buffer(N) ]]. On the Rust side,
//   you "plug in" a buffer to each slot with encoder.set_buffer(N, ...).
//
//        Rust side:                     GPU shader side:
//        encoder.set_buffer(0, &a)  --> device float *a [[ buffer(0) ]]
//        encoder.set_buffer(1, &b)  --> device float *b [[ buffer(1) ]]
//        encoder.set_buffer(2, &c)  --> device float *c [[ buffer(2) ]]
//        encoder.set_buffer(3, &n)  --> device uint  *n [[ buffer(3) ]]
//
// RUN: cargo run -p vector-add
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

    // This kernel adds two arrays element-wise: c[i] = a[i] + b[i]
    //
    // NEW: We now have THREE buffer bindings (a, b, c) plus a scalar parameter.
    //
    // NEW: `device const uint &count [[ buffer(3) ]]`
    //   - `const` means the shader won't modify this value
    //   - `&count` is a reference to a single uint (not an array)
    //   - We pass the array length so the shader can do bounds checking
    //
    // WHY BOUNDS CHECKING?
    //   When we dispatch threads, the total thread count is often rounded up
    //   to a multiple of the threadgroup size (e.g., 32). If our array has
    //   1000 elements, we might launch 1024 threads. Threads 1000-1023 would
    //   read/write out of bounds without this check!
    //
    kernel void vector_add(
        device const float *a     [[ buffer(0) ]],
        device const float *b     [[ buffer(1) ]],
        device float       *c     [[ buffer(2) ]],
        device const uint  &count [[ buffer(3) ]],
        uint gid                  [[ thread_position_in_grid ]]
    ) {
        // Guard against out-of-bounds access. Threads beyond our data just
        // return immediately and do nothing.
        if (gid >= count) {
            return;
        }

        c[gid] = a[gid] + b[gid];
    }
"#;

fn main() {
    autoreleasepool(|| {
        let device = Device::system_default().expect("No Metal-capable GPU found!");
        println!("Using GPU: {}\n", device.name());

        // =====================================================================
        // Compile shader & create pipeline (same pattern as Lesson 2)
        // =====================================================================
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(SHADER_SOURCE, &options)
            .expect("Failed to compile shader");
        let kernel = library
            .get_function("vector_add", None)
            .expect("Couldn't find 'vector_add' function");
        let pipeline_state = device
            .new_compute_pipeline_state_with_function(&kernel)
            .expect("Failed to create pipeline state");

        // =====================================================================
        // Prepare input data
        // =====================================================================
        //
        // Using 1000 elements (not a power of 2) intentionally -- this forces
        // us to handle the bounds-checking case since threadgroups typically
        // have 32 or 64 threads.
        let num_elements: usize = 1000;

        // A = [0.0, 1.0, 2.0, ..., 999.0]
        let a_data: Vec<f32> = (0..num_elements).map(|x| x as f32).collect();

        // B = [1000.0, 999.0, 998.0, ..., 1.0]
        let b_data: Vec<f32> = (0..num_elements)
            .map(|x| (num_elements - x) as f32)
            .collect();

        // Expected: C[i] = A[i] + B[i] = i + (1000 - i) = 1000.0 for all i

        // =====================================================================
        // Create Metal buffers
        // =====================================================================
        let buffer_size = (num_elements * mem::size_of::<f32>()) as u64;

        // Input buffer A -- created with our data
        let buffer_a = device.new_buffer_with_data(
            a_data.as_ptr() as *const _,
            buffer_size,
            MTLResourceOptions::StorageModeShared,
        );

        // Input buffer B -- created with our data
        let buffer_b = device.new_buffer_with_data(
            b_data.as_ptr() as *const _,
            buffer_size,
            MTLResourceOptions::StorageModeShared,
        );

        // Output buffer C -- empty, GPU will fill it
        let buffer_c = device.new_buffer(buffer_size, MTLResourceOptions::StorageModeShared);

        // Count buffer -- a single u32 telling the shader how many elements
        // we have. This is how you pass scalar parameters to Metal shaders.
        //
        // NOTE: In Metal, you can also use encoder.set_bytes() to pass small
        // amounts of data without creating a buffer. But using a buffer works
        // everywhere and is more explicit for learning.
        let count = num_elements as u32;
        let buffer_count = device.new_buffer_with_data(
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

        // Bind all four buffers to their respective indices.
        // These indices MUST match the [[ buffer(N) ]] attributes in the shader.
        encoder.set_buffer(0, Some(&buffer_a), 0); // buffer(0) = a
        encoder.set_buffer(1, Some(&buffer_b), 0); // buffer(1) = b
        encoder.set_buffer(2, Some(&buffer_c), 0); // buffer(2) = c (output)
        encoder.set_buffer(3, Some(&buffer_count), 0); // buffer(3) = count

        // =====================================================================
        // Thread dispatch with non-power-of-2 data
        // =====================================================================
        //
        // We have 1000 elements. The GPU's SIMD width is typically 32.
        // dispatch_threads handles the rounding for us -- it will launch
        // enough threads to cover all 1000 elements, and our shader's
        // bounds check (if gid >= count) protects against the extra threads.
        let grid_size = MTLSize::new(num_elements as u64, 1, 1);
        let threadgroup_size = MTLSize::new(pipeline_state.thread_execution_width(), 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        // Submit and wait
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // =====================================================================
        // Read back and verify results
        // =====================================================================
        let results: &[f32] =
            unsafe { std::slice::from_raw_parts(buffer_c.contents() as *const f32, num_elements) };

        // Print a sample of results
        println!("Vector Addition: C[i] = A[i] + B[i]\n");
        println!("{:>5}  {:>8}  {:>8}  {:>8}", "i", "A[i]", "B[i]", "C[i]");
        println!("{}", "-".repeat(35));
        for i in [0, 1, 2, 3, 498, 499, 500, 997, 998, 999] {
            println!(
                "{:>5}  {:>8.1}  {:>8.1}  {:>8.1}",
                i, a_data[i], b_data[i], results[i]
            );
        }

        // Every result should be 1000.0
        let all_correct = results.iter().all(|&x| (x - 1000.0).abs() < f32::EPSILON);
        println!(
            "\nAll {} results equal 1000.0: {}",
            num_elements, all_correct
        );

        if all_correct {
            println!("\nThe GPU added two 1000-element vectors in parallel!");
            println!("Notice how the non-power-of-2 size (1000) worked fine");
            println!("thanks to the bounds check in the shader.");
        }
    });
}
