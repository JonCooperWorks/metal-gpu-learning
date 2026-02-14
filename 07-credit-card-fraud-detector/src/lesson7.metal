// =============================================================================
// Lesson 7 Metal Kernels: Credit Card Fraud Detector
// =============================================================================
//
// Teaching model:
// - `score_transactions` is the MAP stage:
//    one thread -> one transaction -> one ScoreOutput row.
// - `reduce_metrics` is the REDUCE stage:
//    many rows -> partial aggregates per threadgroup.
//
// CPU later folds partial aggregates into one global result.
// =============================================================================

#include <metal_stdlib>
using namespace metal;

constant uint MAX_SCORE_BINS = 64;

struct KernelParams {
    // Number of valid transactions in this chunk.
    uint count;
    // Histogram buckets (clamped to MAX_SCORE_BINS).
    uint score_bins;
    // Decision cutoff: score >= threshold => predicted fraud.
    float threshold;
    uint _pad0;
};

struct ScoreOutput {
    // Model output in [0, 1].
    float score;
    // Binary prediction from score and threshold.
    uint predicted_fraud;
    // Ground-truth label generated on CPU.
    uint actual_fraud;
    // Histogram bucket for this score.
    uint bin_index;
};

struct PartialMetrics {
    uint tp;
    uint fp;
    uint tn;
    uint fn;
    float sum_score_legit;
    float sum_score_fraud;
    uint count_legit;
    uint count_fraud;
};

// Teaching note:
// MAP kernel. Each thread independently computes score/prediction/bin for
// exactly one transaction index.
kernel void score_transactions(
    device const float *amounts           [[ buffer(0) ]],
    device const float *hours             [[ buffer(1) ]],
    device const float *distance_km       [[ buffer(2) ]],
    device const float *is_card_present   [[ buffer(3) ]],
    device const float *mcc_risk          [[ buffer(4) ]],
    device const float *velocity_1h       [[ buffer(5) ]],
    device const float *country_mismatch  [[ buffer(6) ]],
    device const float *device_change     [[ buffer(7) ]],
    device const uint  *actual_labels     [[ buffer(8) ]],
    device ScoreOutput *scores            [[ buffer(9) ]],
    constant KernelParams &params         [[ buffer(10) ]],
    uint gid                              [[ thread_position_in_grid ]]
) {
    // One GPU thread handles one transaction index (gid).
    if (gid >= params.count) {
        // Extra threads are possible because dispatch size is rounded by hardware.
        return;
    }

    // Normalize features to comparable scales before weighting.
    // This prevents a single raw unit (like dollars) from dominating due to magnitude.
    float amount_n = clamp(amounts[gid] / 2000.0f, 0.0f, 1.0f);
    // Night activity can be riskier in many fraud settings.
    float off_hours = (hours[gid] < 6.0f || hours[gid] > 22.0f) ? 1.0f : 0.0f;
    float distance_n = clamp(distance_km[gid] / 1500.0f, 0.0f, 1.0f);
    // Card-not-present transactions are often higher risk than in-person swipes.
    float card_not_present = 1.0f - is_card_present[gid];
    float velocity_n = clamp(velocity_1h[gid] / 24.0f, 0.0f, 1.0f);

    // Simple weighted rules engine: each signal contributes to risk.
    // Weights are hand-tuned for teaching, not model-calibrated.
    float score = 0.05f;
    // Amount matters, but not overwhelmingly.
    score += 0.20f * amount_n;
    // Time-of-day is a weaker contextual cue.
    score += 0.08f * off_hours;
    score += 0.14f * distance_n;
    score += 0.18f * card_not_present;
    score += 0.14f * velocity_n;
    score += 0.10f * mcc_risk[gid];
    // Country/device mismatch receive stronger weights as anomaly signals.
    score += 0.17f * country_mismatch[gid];
    score += 0.14f * device_change[gid];

    // Interaction bonuses keep this closer to real rule-engine behavior.
    // These capture "risk is greater when both signals happen together".
    score += 0.10f * country_mismatch[gid] * device_change[gid];
    score += 0.08f * card_not_present * velocity_n;
    score = clamp(score, 0.0f, 1.0f);

    // Final fraud flag from threshold.
    uint pred = score >= params.threshold ? 1u : 0u;

    uint bins = max(1u, min(params.score_bins, MAX_SCORE_BINS));
    uint bin = min((uint)(score * (float)bins), bins - 1u);

    ScoreOutput out;
    out.score = score;
    out.predicted_fraud = pred;
    out.actual_fraud = actual_labels[gid];
    out.bin_index = bin;
    scores[gid] = out;
}

// Teaching note:
// REDUCE kernel. Threads cooperatively aggregate ScoreOutput rows into
// per-threadgroup confusion counts, score sums, and histogram slices.
kernel void reduce_metrics(
    device const ScoreOutput *scores      [[ buffer(0) ]],
    device PartialMetrics *partials       [[ buffer(1) ]],
    device uint *partial_histograms       [[ buffer(2) ]],
    constant KernelParams &params         [[ buffer(3) ]],
    uint tid                              [[ thread_index_in_threadgroup ]],
    uint gid                              [[ thread_position_in_grid ]],
    uint tg_pos                           [[ threadgroup_position_in_grid ]],
    uint threads_per_tg                   [[ threads_per_threadgroup ]],
    uint tg_count                         [[ threadgroups_per_grid ]]
) {
    // Shared memory for this threadgroup. Much faster than global memory.
    // We first accumulate locally, then write one partial per group.
    threadgroup uint sh_tp[256];
    threadgroup uint sh_fp[256];
    threadgroup uint sh_tn[256];
    threadgroup uint sh_fn[256];
    threadgroup float sh_legit_sum[256];
    threadgroup float sh_fraud_sum[256];
    threadgroup uint sh_legit_count[256];
    threadgroup uint sh_fraud_count[256];
    threadgroup atomic_uint sh_hist[MAX_SCORE_BINS];

    uint bins = max(1u, min(params.score_bins, MAX_SCORE_BINS));

    // Zero shared histogram before use.
    if (tid < bins) {
        atomic_store_explicit(&sh_hist[tid], 0u, memory_order_relaxed);
    }
    // Make sure every histogram lane is reset before accumulation starts.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Total logical thread count across entire grid.
    uint total_threads = threads_per_tg * tg_count;

    uint tp = 0;
    uint fp = 0;
    uint tn = 0;
    uint fn = 0;
    float legit_sum = 0.0f;
    float fraud_sum = 0.0f;
    uint legit_count = 0;
    uint fraud_count = 0;

    // Grid-stride loop lets any grid size cover any chunk size.
    // Thread gid handles i = gid, gid + total_threads, ...
    for (uint i = gid; i < params.count; i += total_threads) {
        ScoreOutput s = scores[i];

        if (s.predicted_fraud == 1u && s.actual_fraud == 1u) tp += 1;
        if (s.predicted_fraud == 1u && s.actual_fraud == 0u) fp += 1;
        if (s.predicted_fraud == 0u && s.actual_fraud == 0u) tn += 1;
        if (s.predicted_fraud == 0u && s.actual_fraud == 1u) fn += 1;

        if (s.actual_fraud == 1u) {
            // Keep class-specific score sums for later mean calculations.
            fraud_sum += s.score;
            fraud_count += 1;
        } else {
            legit_sum += s.score;
            legit_count += 1;
        }

        // Shared histogram update is atomic because many threads hit same bins.
        atomic_fetch_add_explicit(
            &sh_hist[min(s.bin_index, bins - 1u)],
            1u,
            memory_order_relaxed
        );
    }

    sh_tp[tid] = tp;
    sh_fp[tid] = fp;
    sh_tn[tid] = tn;
    sh_fn[tid] = fn;
    sh_legit_sum[tid] = legit_sum;
    sh_fraud_sum[tid] = fraud_sum;
    sh_legit_count[tid] = legit_count;
    sh_fraud_count[tid] = fraud_count;

    // Ensure all per-thread values are visible before reduction.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction inside the threadgroup.
    // After loop, index 0 contains full group sums.
    for (uint offset = threads_per_tg / 2u; offset > 0u; offset >>= 1u) {
        if (tid < offset) {
            sh_tp[tid] += sh_tp[tid + offset];
            sh_fp[tid] += sh_fp[tid + offset];
            sh_tn[tid] += sh_tn[tid + offset];
            sh_fn[tid] += sh_fn[tid + offset];
            sh_legit_sum[tid] += sh_legit_sum[tid + offset];
            sh_fraud_sum[tid] += sh_fraud_sum[tid + offset];
            sh_legit_count[tid] += sh_legit_count[tid + offset];
            sh_fraud_count[tid] += sh_fraud_count[tid + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes this group's partial results to global memory.
    if (tid == 0u) {
        PartialMetrics out;
        out.tp = sh_tp[0];
        out.fp = sh_fp[0];
        out.tn = sh_tn[0];
        out.fn = sh_fn[0];
        out.sum_score_legit = sh_legit_sum[0];
        out.sum_score_fraud = sh_fraud_sum[0];
        out.count_legit = sh_legit_count[0];
        out.count_fraud = sh_fraud_count[0];
        partials[tg_pos] = out;

        // Histogram is stored as one row per threadgroup.
        uint hist_base = tg_pos * bins;
        for (uint b = 0; b < bins; b++) {
            partial_histograms[hist_base + b] = atomic_load_explicit(&sh_hist[b], memory_order_relaxed);
        }
    }
}
