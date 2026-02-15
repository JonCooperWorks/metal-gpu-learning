// =============================================================================
// LESSON 10 METAL KERNELS (DETAILED TEACHING VERSION)
// =============================================================================
// This file contains 8 independent scoring kernels selected from CLI.
//
// Runtime contract with host (src/main.rs):
// - Host precomputes all per-row features on CPU.
// - Host binds buffers 0..10 exactly as documented below.
// - Each kernel reads one row index and writes one score.
//
// Thread model:
// - `gid` = global thread index within current dispatch chunk.
// - One thread maps to one row in chunk.
// - Row's global index = params.offset + gid.
//
// Why this design is beginner-friendly:
// - Kernels are stateless and branch-light.
// - All math is visible per row without reductions.
// - Every kernel has the same buffer signature, so dispatch code is uniform.
// =============================================================================

#include <metal_stdlib>
using namespace metal;

// Host-provided scalar parameters shared by all kernels.
struct KernelParams {
    uint count;     // number of valid rows in this chunk
    uint offset;    // global row offset of this chunk

    // Distribution stats for paid-based anomaly formulas.
    float mean;
    float std_dev;
    float median;
    float mad;

    // Auxiliary fields reused by specific kernels.
    // For kernel_4 these store paid-per-claim median and MAD.
    float aux_a;
    float aux_b;
};

// Safe division helper to avoid divide-by-zero / INF when denominators are tiny.
inline float safe_div(float num, float den) {
    return num / max(den, 1e-6f);
}

// Shared buffer binding contract (same for all kernels):
//   buffer(0)  paid
//   buffer(1)  claims
//   buffer(2)  beneficiaries
//   buffer(3)  ppc (paid_per_claim)
//   buffer(4)  mom_ratio
//   buffer(5)  drift_sigma
//   buffer(6)  rarity_weight
//   buffer(7)  distance_score
//   buffer(8)  mom_abs_delta
//   buffer(9)  KernelParams
//   buffer(10) out_scores

// -----------------------------------------------------------------------------
// KERNEL 1: Top-spenders base score
// -----------------------------------------------------------------------------
// Formula:
//   score = paid
//
// Interpretation:
// - Higher score => higher raw paid amount at row level.
// - Final top-spender report in host code aggregates by group_id.
// -----------------------------------------------------------------------------
kernel void kernel_1(
    device const float *paid         [[ buffer(0) ]],
    device const float *claims       [[ buffer(1) ]],
    device const float *beneficiaries[[ buffer(2) ]],
    device const float *ppc          [[ buffer(3) ]],
    device const float *mom_ratio    [[ buffer(4) ]],
    device const float *drift_sigma  [[ buffer(5) ]],
    device const float *rarity       [[ buffer(6) ]],
    device const float *distance     [[ buffer(7) ]],
    device const float *mom_abs_delta[[ buffer(8) ]],
    constant KernelParams &params    [[ buffer(9) ]],
    device float *out_scores         [[ buffer(10) ]],
    uint gid                         [[ thread_position_in_grid ]]
) {
    if (gid >= params.count) return;
    uint idx = params.offset + gid;
    out_scores[gid] = paid[idx];
}

// -----------------------------------------------------------------------------
// KERNEL 2: Z-score anomaly on TOTAL_PAID
// -----------------------------------------------------------------------------
// Formula:
//   z = |(x - mean) / std_dev|
// where x = paid[idx].
//
// Interpretation:
// - z ~= 0: close to mean
// - large z: many standard deviations away from mean
// -----------------------------------------------------------------------------
kernel void kernel_2(
    device const float *paid         [[ buffer(0) ]],
    device const float *claims       [[ buffer(1) ]],
    device const float *beneficiaries[[ buffer(2) ]],
    device const float *ppc          [[ buffer(3) ]],
    device const float *mom_ratio    [[ buffer(4) ]],
    device const float *drift_sigma  [[ buffer(5) ]],
    device const float *rarity       [[ buffer(6) ]],
    device const float *distance     [[ buffer(7) ]],
    device const float *mom_abs_delta[[ buffer(8) ]],
    constant KernelParams &params    [[ buffer(9) ]],
    device float *out_scores         [[ buffer(10) ]],
    uint gid                         [[ thread_position_in_grid ]]
) {
    if (gid >= params.count) return;
    uint idx = params.offset + gid;
    out_scores[gid] = fabs(safe_div(paid[idx] - params.mean, params.std_dev));
}

// -----------------------------------------------------------------------------
// KERNEL 3: Modified z-score anomaly on TOTAL_PAID (MAD-based)
// -----------------------------------------------------------------------------
// Formula:
//   mz = 0.6745 * |(x - median) / MAD|
// where x = paid[idx].
//
// Why MAD:
// - More robust than std_dev when outliers already distort the distribution.
// -----------------------------------------------------------------------------
kernel void kernel_3(
    device const float *paid         [[ buffer(0) ]],
    device const float *claims       [[ buffer(1) ]],
    device const float *beneficiaries[[ buffer(2) ]],
    device const float *ppc          [[ buffer(3) ]],
    device const float *mom_ratio    [[ buffer(4) ]],
    device const float *drift_sigma  [[ buffer(5) ]],
    device const float *rarity       [[ buffer(6) ]],
    device const float *distance     [[ buffer(7) ]],
    device const float *mom_abs_delta[[ buffer(8) ]],
    constant KernelParams &params    [[ buffer(9) ]],
    device float *out_scores         [[ buffer(10) ]],
    uint gid                         [[ thread_position_in_grid ]]
) {
    if (gid >= params.count) return;
    uint idx = params.offset + gid;
    float modified_z = 0.6745f * fabs(safe_div(paid[idx] - params.median, params.mad));
    out_scores[gid] = modified_z;
}

// -----------------------------------------------------------------------------
// KERNEL 4: Modified z-score anomaly on paid_per_claim
// -----------------------------------------------------------------------------
// Formula:
//   mz_ppc = 0.6745 * |(ppc - median_ppc) / MAD_ppc|
// where:
//   median_ppc = params.aux_a
//   MAD_ppc    = params.aux_b
// -----------------------------------------------------------------------------
kernel void kernel_4(
    device const float *paid         [[ buffer(0) ]],
    device const float *claims       [[ buffer(1) ]],
    device const float *beneficiaries[[ buffer(2) ]],
    device const float *ppc          [[ buffer(3) ]],
    device const float *mom_ratio    [[ buffer(4) ]],
    device const float *drift_sigma  [[ buffer(5) ]],
    device const float *rarity       [[ buffer(6) ]],
    device const float *distance     [[ buffer(7) ]],
    device const float *mom_abs_delta[[ buffer(8) ]],
    constant KernelParams &params    [[ buffer(9) ]],
    device float *out_scores         [[ buffer(10) ]],
    uint gid                         [[ thread_position_in_grid ]]
) {
    if (gid >= params.count) return;
    uint idx = params.offset + gid;
    float ppc_median = params.aux_a;
    float ppc_mad = max(params.aux_b, 1e-6f);
    float modified_z = 0.6745f * fabs((ppc[idx] - ppc_median) / ppc_mad);
    out_scores[gid] = modified_z;
}

// -----------------------------------------------------------------------------
// KERNEL 5: Month-over-month spike ratio
// -----------------------------------------------------------------------------
// Formula (precomputed on CPU):
//   score = paid_current / paid_previous
//
// Host later applies absolute-delta floor and threshold filtering.
// -----------------------------------------------------------------------------
kernel void kernel_5(
    device const float *paid         [[ buffer(0) ]],
    device const float *claims       [[ buffer(1) ]],
    device const float *beneficiaries[[ buffer(2) ]],
    device const float *ppc          [[ buffer(3) ]],
    device const float *mom_ratio    [[ buffer(4) ]],
    device const float *drift_sigma  [[ buffer(5) ]],
    device const float *rarity       [[ buffer(6) ]],
    device const float *distance     [[ buffer(7) ]],
    device const float *mom_abs_delta[[ buffer(8) ]],
    constant KernelParams &params    [[ buffer(9) ]],
    device float *out_scores         [[ buffer(10) ]],
    uint gid                         [[ thread_position_in_grid ]]
) {
    if (gid >= params.count) return;
    uint idx = params.offset + gid;
    out_scores[gid] = mom_ratio[idx];
}

// -----------------------------------------------------------------------------
// KERNEL 6: Drift anomaly
// -----------------------------------------------------------------------------
// Formula:
//   score = |drift_sigma|
// where drift_sigma is precomputed against rolling baseline on CPU.
// -----------------------------------------------------------------------------
kernel void kernel_6(
    device const float *paid         [[ buffer(0) ]],
    device const float *claims       [[ buffer(1) ]],
    device const float *beneficiaries[[ buffer(2) ]],
    device const float *ppc          [[ buffer(3) ]],
    device const float *mom_ratio    [[ buffer(4) ]],
    device const float *drift_sigma  [[ buffer(5) ]],
    device const float *rarity       [[ buffer(6) ]],
    device const float *distance     [[ buffer(7) ]],
    device const float *mom_abs_delta[[ buffer(8) ]],
    constant KernelParams &params    [[ buffer(9) ]],
    device float *out_scores         [[ buffer(10) ]],
    uint gid                         [[ thread_position_in_grid ]]
) {
    if (gid >= params.count) return;
    uint idx = params.offset + gid;
    out_scores[gid] = fabs(drift_sigma[idx]);
}

// -----------------------------------------------------------------------------
// KERNEL 7: HCPCS rarity-weighted anomaly
// -----------------------------------------------------------------------------
// Formula:
//   score = paid * rarity_weight
// where rarity_weight ~= 1 / frequency(hcpcs).
// -----------------------------------------------------------------------------
kernel void kernel_7(
    device const float *paid         [[ buffer(0) ]],
    device const float *claims       [[ buffer(1) ]],
    device const float *beneficiaries[[ buffer(2) ]],
    device const float *ppc          [[ buffer(3) ]],
    device const float *mom_ratio    [[ buffer(4) ]],
    device const float *drift_sigma  [[ buffer(5) ]],
    device const float *rarity       [[ buffer(6) ]],
    device const float *distance     [[ buffer(7) ]],
    device const float *mom_abs_delta[[ buffer(8) ]],
    constant KernelParams &params    [[ buffer(9) ]],
    device float *out_scores         [[ buffer(10) ]],
    uint gid                         [[ thread_position_in_grid ]]
) {
    if (gid >= params.count) return;
    uint idx = params.offset + gid;
    out_scores[gid] = paid[idx] * rarity[idx];
}

// -----------------------------------------------------------------------------
// KERNEL 8: Distance-based outlier score
// -----------------------------------------------------------------------------
// Formula (precomputed on CPU):
//   score = Euclidean distance from normalized centroid
// -----------------------------------------------------------------------------
kernel void kernel_8(
    device const float *paid         [[ buffer(0) ]],
    device const float *claims       [[ buffer(1) ]],
    device const float *beneficiaries[[ buffer(2) ]],
    device const float *ppc          [[ buffer(3) ]],
    device const float *mom_ratio    [[ buffer(4) ]],
    device const float *drift_sigma  [[ buffer(5) ]],
    device const float *rarity       [[ buffer(6) ]],
    device const float *distance     [[ buffer(7) ]],
    device const float *mom_abs_delta[[ buffer(8) ]],
    constant KernelParams &params    [[ buffer(9) ]],
    device float *out_scores         [[ buffer(10) ]],
    uint gid                         [[ thread_position_in_grid ]]
) {
    if (gid >= params.count) return;
    uint idx = params.offset + gid;
    out_scores[gid] = distance[idx];
}
