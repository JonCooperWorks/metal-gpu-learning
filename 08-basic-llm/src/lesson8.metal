// =============================================================================
// LESSON 8: Basic LLM -- Single-Head Attention for One Autoregressive Step
// =============================================================================
//
// This kernel computes the attention result for the CURRENT token only.
// CPU has already built:
//   - q_current[d_model]
//   - k_all[seq_len][d_model]
//   - v_all[seq_len][d_model]
//
// The kernel computes:
//   1) scores[i] = dot(q_current, k_all[i]) * scale
//   2) weights = softmax(scores)
//   3) context[d] = sum_i weights[i] * v_all[i][d]
//
// Why one-step attention?
//   - It's the core operation in autoregressive generation.
//   - Keeping one step per dispatch makes the lesson easy to follow.
// =============================================================================

#include <metal_stdlib>
using namespace metal;

constant uint MAX_SEQ = 64;
constant uint MAX_D_MODEL = 64;

struct AttentionParams {
    uint seq_len;
    uint d_model;
    float scale;
    float _pad;
};

kernel void single_head_attention_step(
    device const float *q_current        [[ buffer(0) ]],
    device const float *k_all            [[ buffer(1) ]],
    device const float *v_all            [[ buffer(2) ]],
    device float *attn_weights_out       [[ buffer(3) ]],
    device float *context_out            [[ buffer(4) ]],
    constant AttentionParams &params     [[ buffer(5) ]],
    uint tid                             [[ thread_index_in_threadgroup ]]
) {
    if (params.seq_len == 0u || params.d_model == 0u) {
        return;
    }

    // Shared arrays keep intermediate values local to this one threadgroup.
    threadgroup float sh_scores[MAX_SEQ];
    threadgroup float sh_weights[MAX_SEQ];

    // -------------------------------------------------------------------------
    // Phase 1: compute raw attention score per position
    // -------------------------------------------------------------------------
    if (tid < params.seq_len) {
        float dot_qk = 0.0f;
        uint row_base = tid * params.d_model;
        for (uint d = 0u; d < params.d_model; d++) {
            dot_qk += q_current[d] * k_all[row_base + d];
        }
        sh_scores[tid] = dot_qk * params.scale;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // Phase 2: softmax over scores
    // -------------------------------------------------------------------------
    // We keep this in tid==0 for teaching clarity. This is not the fastest
    // possible softmax implementation, but it is very readable.
    if (tid == 0u) {
        float max_score = sh_scores[0];
        for (uint i = 1u; i < params.seq_len; i++) {
            max_score = max(max_score, sh_scores[i]);
        }

        float denom = 0.0f;
        for (uint i = 0u; i < params.seq_len; i++) {
            float w = exp(sh_scores[i] - max_score);
            sh_weights[i] = w;
            denom += w;
        }

        float inv_denom = 1.0f / max(denom, 1e-12f);
        for (uint i = 0u; i < params.seq_len; i++) {
            sh_weights[i] *= inv_denom;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // Phase 3: write attention weights for inspection/debugging
    // -------------------------------------------------------------------------
    if (tid < params.seq_len) {
        attn_weights_out[tid] = sh_weights[tid];
    }

    // -------------------------------------------------------------------------
    // Phase 4: weighted sum of V -> context vector
    // -------------------------------------------------------------------------
    // Each thread computes one context dimension.
    if (tid < params.d_model) {
        float acc = 0.0f;
        for (uint i = 0u; i < params.seq_len; i++) {
            uint idx = i * params.d_model + tid;
            acc += sh_weights[i] * v_all[idx];
        }
        context_out[tid] = acc;
    }
}
