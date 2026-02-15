// =============================================================================
// LESSON 9 METAL KERNELS: Attention + Logits for one generation step
// =============================================================================
// Beginner context:
// - CPU builds model state and supplies tensors.
// - GPU computes two dense math blocks fast:
//   1) attention for the last token in sequence,
//   2) projection from hidden state to vocabulary logits.
//
// Symbols used below:
//   d_model  = hidden width D
//   seq_len  = active context length T
//   q, k, v  = query/key/value vectors
//
// Core equations implemented here:
//   score_t   = (q · k_t) / sqrt(D)
//   prob_t    = softmax(score)_t
//   context_j = sum_t prob_t * v_t[j]
//   logit_j   = w_j · h + b_j
// =============================================================================

#include <metal_stdlib>
using namespace metal;

// Compile-time safety bounds used by this educational kernel.
constant uint MAX_SEQ = 64;
constant uint MAX_D_MODEL = 256;

// Host-provided scalar parameters for the attention kernel.
struct AttentionParams {
    uint seq_len;
    uint d_model;
    float scale; // should be 1/sqrt(d_model)
    float _pad;
};

kernel void attention_last_token(
    device const float *q_last            [[ buffer(0) ]],
    device const float *k_all             [[ buffer(1) ]],
    device const float *v_all             [[ buffer(2) ]],
    device float *weights_out             [[ buffer(3) ]],
    device float *context_out             [[ buffer(4) ]],
    constant AttentionParams &params      [[ buffer(5) ]],
    uint tid                              [[ thread_index_in_threadgroup ]]
) {
    if (params.seq_len == 0 || params.d_model == 0) {
        return;
    }

    // Threadgroup scratch arrays:
    // - scores[t] holds q·k_t/sqrt(D)
    // - probs[t] holds softmax(scores)[t]
    threadgroup float scores[MAX_SEQ];
    threadgroup float probs[MAX_SEQ];

    // Stage 1: each lane tid computes one score for time index t=tid.
    // score_t = (q · k_t) * scale where scale = 1/sqrt(D).
    if (tid < params.seq_len) {
        float dot_qk = 0.0f;
        uint base = tid * params.d_model;
        for (uint d = 0; d < params.d_model; d++) {
            dot_qk += q_last[d] * k_all[base + d];
        }
        scores[tid] = dot_qk * params.scale;
    }

    // Barrier ensures all scores are available before softmax.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 2: one lane computes stable softmax over scores.
    // Stable softmax trick: exp(score - max_score) to reduce overflow.
    if (tid == 0) {
        float max_score = scores[0];
        for (uint i = 1; i < params.seq_len; i++) {
            max_score = max(max_score, scores[i]);
        }

        float denom = 0.0f;
        for (uint i = 0; i < params.seq_len; i++) {
            float w = exp(scores[i] - max_score);
            probs[i] = w;
            denom += w;
        }

        // prob_i = exp_i / sum_j exp_j.
        float inv = 1.0f / max(denom, 1e-12f);
        for (uint i = 0; i < params.seq_len; i++) {
            probs[i] *= inv;
        }
    }

    // Barrier ensures probabilities are ready for all lanes.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Optional debug output: full attention weights.
    if (tid < params.seq_len) {
        weights_out[tid] = probs[tid];
    }

    // Stage 3: each lane tid computes one context dimension j=tid.
    // context_j = sum_t probs[t] * v_t[j].
    if (tid < params.d_model) {
        float acc = 0.0f;
        for (uint i = 0; i < params.seq_len; i++) {
            uint idx = i * params.d_model + tid;
            acc += probs[i] * v_all[idx];
        }
        context_out[tid] = acc;
    }
}

// Host-provided scalar parameters for logits kernel.
struct LogitsParams {
    uint d_model;
    uint vocab_size;
};

kernel void logits_projection(
    device const float *hidden            [[ buffer(0) ]],
    device const float *lm_head_weight    [[ buffer(1) ]],
    device const float *lm_head_bias      [[ buffer(2) ]],
    device float *logits_out              [[ buffer(3) ]],
    constant LogitsParams &params         [[ buffer(4) ]],
    uint tid                              [[ thread_position_in_grid ]]
) {
    if (tid >= params.vocab_size) {
        return;
    }

    // One thread computes one vocabulary logit:
    // logit_tid = bias[tid] + dot(weight[tid, :], hidden[:]).
    float acc = lm_head_bias[tid];
    uint row = tid * params.d_model;
    for (uint d = 0; d < params.d_model; d++) {
        acc += lm_head_weight[row + d] * hidden[d];
    }
    logits_out[tid] = acc;
}
