// =============================================================================
// Lesson 6 Metal Kernels: Market Tick Aggregation
// =============================================================================
// Stage A (`transform_ticks`):
//   - one thread per tick
//   - derive fields used later by reduction
//
// Stage B (`reduce_partials`):
//   - each threadgroup reduces many ticks into one partial aggregate
//   - CPU folds threadgroup partials into one global result per chunk
//
// This two-stage design is a common production pattern because it avoids
// serial bottlenecks (e.g., every thread atomically hitting one global value).

#include <metal_stdlib>
using namespace metal;

// Scalar parameters shared by both kernels.
// `base_index` lets each chunk map back to a global synthetic timeline.
struct KernelParams {
    uint  count;
    uint  _pad;
    ulong base_index;
};

// Per-tick intermediate written by Stage A and consumed by Stage B.
struct TickDerived {
    float size;
    float notional;
    float return_proxy;
    float return_proxy_sq;
    float price;
};

// One reduction output per threadgroup.
// CPU combines all partials for the final chunk/global totals.
struct PartialAggregate {
    float sum_size;
    float sum_notional;
    float sum_return_proxy;
    float sum_return_proxy_sq;
    float min_price;
    float max_price;
    uint  tick_count;
};

// Deterministic synthetic generators keep runs reproducible across CPU/GPU.
inline float price_for_index(ulong idx) {
    return 90.0f + float((idx * 17ul + 13ul) % 5000ul) * 0.01f;
}

inline float size_for_index(ulong idx) {
    return 1.0f + float((idx * 29ul + 7ul) % 200ul) * 0.05f;
}

inline float baseline_price_for_index(ulong idx) {
    return 95.0f + float((idx * 31ul + 3ul) % 4000ul) * 0.01f;
}

// Stage A: embarrassingly parallel transform.
// Every lane handles one tick and writes one TickDerived record.
kernel void transform_ticks(
    device const float     *prices   [[ buffer(0) ]],
    device const float     *sizes    [[ buffer(1) ]],
    device TickDerived     *derived  [[ buffer(2) ]],
    constant KernelParams  &params   [[ buffer(3) ]],
    uint gid                         [[ thread_position_in_grid ]]
) {
    if (gid >= params.count) {
        return;
    }

    // Per-tick feature engineering (regular math, SIMT-friendly).
    float price = prices[gid];
    float size = sizes[gid];
    float notional = price * size;

    // Deterministic baseline avoids data dependency on previous tick.
    // That keeps this stage fully parallel while modeling drift from reference.
    ulong global_idx = params.base_index + ulong(gid);
    float baseline = baseline_price_for_index(global_idx);
    float return_proxy = fabs(price - baseline);

    TickDerived out;
    out.size = size;
    out.notional = notional;
    out.return_proxy = return_proxy;
    out.return_proxy_sq = return_proxy * return_proxy;
    out.price = price;

    derived[gid] = out;
}

// Stage B: threadgroup reduction.
// Each group accumulates local sums/min/max, then writes one partial record.
//
// Why this is better than one global atomic sum:
// - fewer global memory updates
// - less serialization
// - scales with threadgroup parallelism
kernel void reduce_partials(
    device const TickDerived       *derived   [[ buffer(0) ]],
    device PartialAggregate        *partials  [[ buffer(1) ]],
    constant KernelParams          &params    [[ buffer(2) ]],
    uint tid                                  [[ thread_index_in_threadgroup ]],
    uint gid                                  [[ thread_position_in_grid ]],
    uint tg_pos                               [[ threadgroup_position_in_grid ]],
    uint threads_per_tg                       [[ threads_per_threadgroup ]],
    uint tg_count                             [[ threadgroups_per_grid ]]
) {
    // Shared memory scratchpads (one slot per lane).
    threadgroup float sh_sum_size[256];
    threadgroup float sh_sum_notional[256];
    threadgroup float sh_sum_ret[256];
    threadgroup float sh_sum_ret_sq[256];
    threadgroup float sh_min_price[256];
    threadgroup float sh_max_price[256];
    threadgroup uint  sh_count[256];

    uint total_threads = threads_per_tg * tg_count;

    // Strided walk: one lane processes multiple ticks when count > grid size.
    float sum_size = 0.0f;
    float sum_notional = 0.0f;
    float sum_ret = 0.0f;
    float sum_ret_sq = 0.0f;
    float min_price = FLT_MAX;
    float max_price = -FLT_MAX;
    uint count = 0;

    for (uint i = gid; i < params.count; i += total_threads) {
        TickDerived d = derived[i];
        sum_size += d.size;
        sum_notional += d.notional;
        sum_ret += d.return_proxy;
        sum_ret_sq += d.return_proxy_sq;
        min_price = min(min_price, d.price);
        max_price = max(max_price, d.price);
        count += 1;
    }

    // Publish lane-local accumulators into threadgroup memory.
    sh_sum_size[tid] = sum_size;
    sh_sum_notional[tid] = sum_notional;
    sh_sum_ret[tid] = sum_ret;
    sh_sum_ret_sq[tid] = sum_ret_sq;
    sh_min_price[tid] = min_price;
    sh_max_price[tid] = max_price;
    sh_count[tid] = count;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction: halves active lanes each round.
    for (uint offset = threads_per_tg / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sh_sum_size[tid] += sh_sum_size[tid + offset];
            sh_sum_notional[tid] += sh_sum_notional[tid + offset];
            sh_sum_ret[tid] += sh_sum_ret[tid + offset];
            sh_sum_ret_sq[tid] += sh_sum_ret_sq[tid + offset];
            sh_min_price[tid] = min(sh_min_price[tid], sh_min_price[tid + offset]);
            sh_max_price[tid] = max(sh_max_price[tid], sh_max_price[tid + offset]);
            sh_count[tid] += sh_count[tid + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Lane 0 commits one compact partial aggregate for this threadgroup.
    if (tid == 0) {
        PartialAggregate out;
        out.sum_size = sh_sum_size[0];
        out.sum_notional = sh_sum_notional[0];
        out.sum_return_proxy = sh_sum_ret[0];
        out.sum_return_proxy_sq = sh_sum_ret_sq[0];
        out.min_price = sh_min_price[0];
        out.max_price = sh_max_price[0];
        out.tick_count = sh_count[0];
        partials[tg_pos] = out;
    }
}
