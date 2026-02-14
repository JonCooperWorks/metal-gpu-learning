// =============================================================================
// LESSON 1: Hello GPU -- Device Discovery
// =============================================================================
//
// This is the absolute starting point for Metal programming. Before we can run
// any code on the GPU, we need to find it. In Metal, the GPU is represented by
// a "device" object (MTLDevice). This lesson:
//
//   1. Connects to the system's default Metal device (your Mac's GPU)
//   2. Queries and prints its properties
//   3. That's it! No shaders, no compute -- just "hello, GPU, are you there?"
//
// KEY CONCEPTS:
//   - MTLDevice: The object that represents a GPU. Everything in Metal starts
//     here. You use it to create buffers, compile shaders, build pipelines, etc.
//   - autoreleasepool: Metal's Rust bindings talk to Objective-C under the hood.
//     Objective-C uses reference counting for memory management, and many Metal
//     calls return "autoreleased" objects. The autoreleasepool block ensures
//     these objects are properly freed when the block ends. Always wrap your
//     Metal code in one.
//
// RUN: cargo run -p hello-gpu
// =============================================================================

use metal::*;
use objc::rc::autoreleasepool;

fn main() {
    // =========================================================================
    // STEP 1: Wrap everything in an autoreleasepool
    // =========================================================================
    //
    // Why? Metal is built on Objective-C. When Objective-C methods return
    // objects, they're often "autoreleased" -- meaning they'll be freed at the
    // end of the current autorelease pool. Without this pool, those objects
    // would leak. In a GUI app, the run loop provides one automatically, but
    // in a command-line app like this, we need to create our own.
    autoreleasepool(|| {
        // =====================================================================
        // STEP 2: Get the default Metal device
        // =====================================================================
        //
        // `Device::system_default()` returns the system's preferred GPU.
        // On a MacBook, this is typically the integrated or discrete GPU.
        // On Apple Silicon (M1/M2/M3/M4), it's the unified GPU built into
        // the chip.
        //
        // This is equivalent to calling MTLCreateSystemDefaultDevice() in
        // Objective-C or Swift.
        //
        // It returns an Option<Device> -- None if no Metal-capable GPU exists
        // (very unlikely on any modern Mac).
        let device = Device::system_default().expect("No Metal-capable GPU found!");

        // =====================================================================
        // STEP 3: Query and print device properties
        // =====================================================================
        //
        // The device object exposes many properties about the GPU's
        // capabilities. Let's look at the most important ones.

        println!("=== Metal Device Info ===\n");

        // --- Device name ---
        // A human-readable name like "Apple M2 Pro" or "AMD Radeon Pro 5500M".
        println!("Device name:            {}", device.name());

        // --- Registry ID ---
        // A unique identifier for this GPU in the system's I/O registry.
        // Useful if you have multiple GPUs and need to tell them apart.
        println!("Registry ID:            {}", device.registry_id());

        // --- Max buffer length ---
        // The maximum size (in bytes) of a single MTLBuffer you can allocate.
        // On Apple Silicon this is typically very large (several GB).
        // We convert to megabytes for readability.
        let max_buf_mb = device.max_buffer_length() / (1024 * 1024);
        println!("Max buffer length:      {} MB", max_buf_mb);

        // --- Max threads per threadgroup ---
        // When you dispatch a compute kernel, work is organized into
        // "threadgroups" (also called "workgroups" in other APIs like OpenCL).
        // Each threadgroup is a block of threads that can share memory and
        // synchronize with each other. This property tells you the maximum
        // number of threads you can put in a single threadgroup.
        //
        // The value is returned as an MTLSize with width/height/depth because
        // threadgroups can be 1D, 2D, or 3D. For compute, the total is what
        // matters: width * height * depth.
        let max_tpg = device.max_threads_per_threadgroup();
        println!(
            "Max threads/threadgroup: {} x {} x {} (= {} total)",
            max_tpg.width,
            max_tpg.height,
            max_tpg.depth,
            max_tpg.width * max_tpg.height * max_tpg.depth
        );

        // --- Is low power? ---
        // Returns true for integrated GPUs (which trade performance for power
        // efficiency). On Apple Silicon, this is always false because the GPU
        // is neither "integrated" nor "discrete" in the traditional sense --
        // it's unified.
        println!("Low power:              {}", device.is_low_power());

        // --- Has unified memory? ---
        // On Apple Silicon, the CPU and GPU share the same physical memory.
        // This means buffers created on the CPU are already accessible to the
        // GPU without any copying -- a huge performance advantage. On Intel
        // Macs with discrete GPUs, this would be false.
        println!("Unified memory:         {}", device.has_unified_memory());

        println!("\n=== GPU is ready! ===");
        println!("In the next lesson, we'll actually run code on it.");
    });
}
