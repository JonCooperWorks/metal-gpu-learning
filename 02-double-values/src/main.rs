// =============================================================================
// LESSON 2: Double Values -- Processing 100 BILLION Numbers on the GPU
// =============================================================================
//
// This lesson doubles 100,000,000,000 (100 billion) floating-point numbers on
// the GPU. That's 400 GB of input data. Your Mac probably has 32-128 GB of RAM,
// and even the largest single Metal buffer is ~39 GB. So how do we do it?
//
// THE ANSWER: CHUNKING
//
// We can't fit 400 GB in memory, but we CAN fit 1 GB. So we:
//
//   1. Allocate two reusable buffers (input + output) that hold 256 million
//      floats each (~1 GB each).
//   2. Fill the input buffer with data.
//   3. Run the GPU kernel to double it.
//   4. Verify the output.
//   5. Repeat 400 times. 400 chunks × 250M elements = 100 billion total.
//
// This is exactly how real GPU applications handle datasets larger than memory.
// Whether it's machine learning training on terabytes of data, or processing
// a massive video file -- the pattern is always: chunk, process, repeat.
//
// THE MATH:
//
//   100,000,000,000 elements × 4 bytes/float = 400 GB of data
//   250,000,000 elements per chunk × 4 bytes  = 1 GB per buffer
//   We need 2 buffers (input + output)         = 2 GB of GPU memory
//   400 chunks × 250M elements                 = 100 billion total
//
// WHAT YOU'LL LEARN:
//
//   - Why you can't just allocate 400 GB and dispatch once
//   - The chunking pattern: allocate once, refill, re-dispatch
//   - Writing directly into shared GPU memory (zero-copy on Apple Silicon)
//   - Buffer reuse: same buffers, new data each iteration
//   - Measuring GPU throughput (elements/sec, GB/s)
//   - The full compute pipeline from Lesson 2, now in a real-world context
//
// RUN: cargo run --release -p double-values
//      (--release is important! CPU-side data generation is much faster optimized)
// =============================================================================

use metal::*;
use objc::rc::autoreleasepool;
use std::time::Instant;

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Total number of floats we want to double. 100 billion.
const TOTAL_ELEMENTS: u64 = 100_000_000_000;

/// How many floats we process per GPU dispatch.
///
/// 250 million × 4 bytes = 1 GB per buffer. We need two buffers (input + output),
/// so each chunk uses ~2 GB of GPU memory. This is comfortably within any Mac's
/// capabilities, even a MacBook Air with 8 GB of unified memory.
///
/// Why not bigger? We could go up to ~9 billion elements per buffer (the ~39 GB
/// max you saw in Lesson 1), but:
///   - Larger chunks mean more memory pressure, less room for the OS and apps
///   - Diminishing returns: the GPU is already fully utilized at 250M elements
///   - Smaller chunks give us more frequent progress updates
const ELEMENTS_PER_CHUNK: u64 = 250_000_000;

/// Number of chunks needed. With 250M per chunk, that's 400 chunks.
const NUM_CHUNKS: u64 = TOTAL_ELEMENTS / ELEMENTS_PER_CHUNK;

// =============================================================================
// THE METAL SHADER (same kernel as before -- the GPU code doesn't change!)
// =============================================================================
const SHADER_SOURCE: &str = r#"
    #include <metal_stdlib>
    using namespace metal;

    // Same simple kernel: each thread doubles one element.
    // The GPU doesn't know or care that we're calling it 400 times --
    // each dispatch is independent.
    kernel void double_values(
        device float *input  [[ buffer(0) ]],
        device float *output [[ buffer(1) ]],
        uint gid             [[ thread_position_in_grid ]]
    ) {
        output[gid] = input[gid] * 2.0;
    }
"#;

fn main() {
    autoreleasepool(|| {
        let device = Device::system_default().expect("No Metal-capable GPU found!");
        println!("============================================================");
        println!(" Doubling 100 BILLION numbers on the GPU");
        println!("============================================================");
        println!("GPU:                {}", device.name());
        println!("Total elements:     {} ({:.0} billion)", TOTAL_ELEMENTS, TOTAL_ELEMENTS as f64 / 1e9);
        println!("Elements per chunk: {} ({:.0} million)", ELEMENTS_PER_CHUNK, ELEMENTS_PER_CHUNK as f64 / 1e6);
        println!("Number of chunks:   {}", NUM_CHUNKS);
        println!("Memory per chunk:   {:.1} GB (input) + {:.1} GB (output) = {:.1} GB",
            ELEMENTS_PER_CHUNK as f64 * 4.0 / 1e9,
            ELEMENTS_PER_CHUNK as f64 * 4.0 / 1e9,
            ELEMENTS_PER_CHUNK as f64 * 8.0 / 1e9,
        );
        println!("Total data:         {:.0} GB\n", TOTAL_ELEMENTS as f64 * 4.0 / 1e9);

        // =====================================================================
        // STEP 1: Setup (compile shader, create pipeline) -- done ONCE
        // =====================================================================
        //
        // Everything here is reusable across all 400 chunks. The pipeline state,
        // command queue, and buffers are all allocated once and reused.
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(SHADER_SOURCE, &options)
            .expect("Failed to compile shader");
        let kernel = library
            .get_function("double_values", None)
            .expect("Couldn't find 'double_values' function");
        let pipeline_state = device
            .new_compute_pipeline_state_with_function(&kernel)
            .expect("Failed to create pipeline state");
        let command_queue = device.new_command_queue();

        let chunk_size = ELEMENTS_PER_CHUNK as usize;
        let buffer_bytes = (chunk_size * std::mem::size_of::<f32>()) as u64;

        // =====================================================================
        // STEP 2: Allocate REUSABLE buffers -- done ONCE
        // =====================================================================
        //
        // This is the key insight: we allocate these two buffers once and reuse
        // them for all 400 chunks. We're not allocating 400 GB -- just 2 GB.
        //
        // `StorageModeShared` means this memory is accessible to BOTH the CPU
        // and GPU. On Apple Silicon, it's the same physical RAM -- the CPU can
        // write data into the buffer, and the GPU reads from it, with no copy.
        let input_buffer = device.new_buffer(buffer_bytes, MTLResourceOptions::StorageModeShared);
        let output_buffer = device.new_buffer(buffer_bytes, MTLResourceOptions::StorageModeShared);

        // =====================================================================
        // STEP 3: Get raw pointers to the buffer memory
        // =====================================================================
        //
        // Since the buffers are StorageModeShared, `.contents()` gives us a raw
        // pointer that the CPU can read/write directly. On Apple Silicon this IS
        // the GPU memory -- there's no separate "upload" step.
        //
        // We cast to *mut f32 so we can write floats into it.
        let input_ptr = input_buffer.contents() as *mut f32;
        let output_ptr = output_buffer.contents() as *const f32;

        // Pre-compute dispatch sizes (same for every chunk)
        let grid_size = MTLSize::new(ELEMENTS_PER_CHUNK, 1, 1);
        let threadgroup_size = MTLSize::new(pipeline_state.thread_execution_width(), 1, 1);

        // =====================================================================
        // STEP 4: Fill the buffer with initial data and verify correctness
        // =====================================================================
        //
        // Before we run the full 100 billion, let's make sure the kernel works
        // correctly on the first chunk. We fill it with known values, run the
        // kernel, and check every single result.
        println!("--- Phase 1: Correctness check on first chunk ---\n");

        // Write directly into GPU-shared memory. No intermediate Vec, no copy.
        // Each element gets the value (i + 1) as f32: [1.0, 2.0, 3.0, ...]
        //
        // SAFETY: We own the buffer, it's exactly `chunk_size` f32s, and the
        // GPU isn't running yet.
        unsafe {
            for i in 0..chunk_size {
                input_ptr.add(i).write((i + 1) as f32);
            }
        }

        // Dispatch the kernel (same pattern as the original Lesson 2)
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Verify every element: output[i] should be (i + 1) * 2.0
        let results = unsafe { std::slice::from_raw_parts(output_ptr, chunk_size) };
        let mut all_correct = true;
        for i in 0..chunk_size {
            let expected = (i + 1) as f32 * 2.0;
            if (results[i] - expected).abs() > 0.01 {
                println!("MISMATCH at [{}]: expected {}, got {}", i, expected, results[i]);
                all_correct = false;
                break;
            }
        }

        // Show a sample
        println!("Sample results from first chunk:");
        for i in [0, 1, 2, 3, 4, chunk_size - 3, chunk_size - 2, chunk_size - 1] {
            println!("  input[{:>9}] = {:>12.1}  -->  output = {:>12.1}",
                i, (i + 1) as f32, results[i]);
        }
        println!("\nAll {} elements correct: {}\n", chunk_size, all_correct);

        if !all_correct {
            println!("Correctness check failed! Aborting.");
            return;
        }

        // =====================================================================
        // STEP 5: Process all 100 billion elements!
        // =====================================================================
        //
        // Now we run the kernel 400 times. Each iteration:
        //   1. Fills the input buffer with this chunk's data (writing directly
        //      into shared GPU memory)
        //   2. Creates a new command buffer (they're single-use)
        //   3. Encodes and dispatches the kernel
        //   4. Waits for the GPU to finish
        //
        // WHY DO WE WAIT EACH ITERATION?
        //   Because we reuse the same input buffer. If we submitted chunk N+1
        //   before chunk N finished, we'd overwrite the input data while the
        //   GPU is still reading it! A production app would use double-buffering
        //   (two sets of buffers, alternating) to keep the GPU busy while the
        //   CPU prepares the next chunk. But for clarity, we keep it simple.
        println!("--- Phase 2: Processing 100 billion elements ---\n");

        let overall_start = Instant::now();
        let mut total_gpu_time_ms: f64 = 0.0;
        let mut total_fill_time_ms: f64 = 0.0;

        for chunk in 0..NUM_CHUNKS {
            // Calculate the starting index for this chunk's data.
            // Chunk 0: elements 0..250M, Chunk 1: 250M..500M, etc.
            let base_index = chunk * ELEMENTS_PER_CHUNK;

            // --- Fill the input buffer ---
            // Write directly into shared GPU memory. Each element gets a value
            // derived from its global index so every chunk has unique data.
            //
            // We use wrapping arithmetic for the f32 value because at 100B
            // elements, the indices exceed f32 precision (~16M). That's fine --
            // we just need the kernel to do *something* with each element.
            let fill_start = Instant::now();
            unsafe {
                for i in 0..chunk_size {
                    let global_idx = base_index + i as u64;
                    // Use modular arithmetic to keep values in f32's precise range
                    input_ptr.add(i).write((global_idx % 1_000_000) as f32 + 1.0);
                }
            }
            total_fill_time_ms += fill_start.elapsed().as_secs_f64() * 1000.0;

            // --- Dispatch the kernel ---
            // Create a fresh command buffer (they're single-use and cheap)
            let gpu_start = Instant::now();
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline_state);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            encoder.dispatch_threads(grid_size, threadgroup_size);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            total_gpu_time_ms += gpu_start.elapsed().as_secs_f64() * 1000.0;

            // --- Progress update ---
            // Print every 40 chunks (10% intervals)
            if (chunk + 1) % (NUM_CHUNKS / 10) == 0 || chunk == NUM_CHUNKS - 1 {
                let elements_done = (chunk + 1) * ELEMENTS_PER_CHUNK;
                let pct = elements_done as f64 / TOTAL_ELEMENTS as f64 * 100.0;
                let elapsed = overall_start.elapsed().as_secs_f64();
                println!(
                    "  Chunk {:>3}/{}: {:>6.1}B elements done ({:.0}%) -- {:.1}s elapsed",
                    chunk + 1,
                    NUM_CHUNKS,
                    elements_done as f64 / 1e9,
                    pct,
                    elapsed,
                );
            }
        }

        // =====================================================================
        // STEP 6: Report performance
        // =====================================================================
        let total_time = overall_start.elapsed().as_secs_f64();
        let total_data_gb = TOTAL_ELEMENTS as f64 * 4.0 / 1e9; // input data in GB
        let total_bandwidth_gb = total_data_gb * 2.0;           // read + write

        println!("\n============================================================");
        println!(" DONE -- 100 billion numbers doubled!");
        println!("============================================================");
        println!("Total wall time:     {:.2}s", total_time);
        println!("  CPU fill time:     {:.2}s ({:.0}% of total)",
            total_fill_time_ms / 1000.0,
            total_fill_time_ms / 1000.0 / total_time * 100.0);
        println!("  GPU compute time:  {:.2}s ({:.0}% of total)",
            total_gpu_time_ms / 1000.0,
            total_gpu_time_ms / 1000.0 / total_time * 100.0);
        println!();
        println!("Throughput:");
        println!("  {:.2} billion elements/sec (GPU only)",
            TOTAL_ELEMENTS as f64 / 1e9 / (total_gpu_time_ms / 1000.0));
        println!("  {:.2} billion elements/sec (wall clock, incl. CPU fill)",
            TOTAL_ELEMENTS as f64 / 1e9 / total_time);
        println!("  {:.1} GB/s GPU bandwidth (read + write)",
            total_bandwidth_gb / (total_gpu_time_ms / 1000.0));
        println!();

        // =====================================================================
        // WHAT WE LEARNED
        // =====================================================================
        println!("--- What just happened ---");
        println!();
        println!("We processed 400 GB of data using only ~2 GB of GPU memory.");
        println!("The trick: allocate buffers ONCE, then refill and re-dispatch");
        println!("in a loop. This is the same pattern used by ML frameworks,");
        println!("video encoders, and scientific computing tools.");
        println!();
        println!("The breakdown above shows where time was spent:");
        println!("  - CPU fill: writing {} million f32s into shared memory per chunk", ELEMENTS_PER_CHUNK / 1_000_000);
        println!("  - GPU compute: the actual kernel execution");
        println!();
        println!("In a production app, you'd overlap CPU and GPU work using");
        println!("double-buffering: while the GPU processes chunk N, the CPU");
        println!("fills chunk N+1 in a second buffer. This hides the fill time");
        println!("and keeps the GPU 100% busy.");
    });
}
