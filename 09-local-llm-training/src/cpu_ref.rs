// =============================================================================
// LESSON 9 CPU REFERENCE FORWARD PASS (WITH DEBUG TENSORS)
// =============================================================================
// Beginner context:
// - This is a pure-CPU implementation of the same transformer math used by GPU.
// - We use it as a correctness oracle: GPU outputs are compared against these
//   values during validation.
//
// Important equations in this file:
//   score_t = (q · k_t) / sqrt(D)
//   attn_t  = softmax(score)_t
//   ctx_j   = sum_t attn_t * v_t[j]
//   layernorm(x) = gamma * (x - mean) / sqrt(var + eps) + beta
// =============================================================================

use crate::model_json::ModelJson;
use anyhow::{bail, Result};

#[derive(Debug, Clone)]
pub struct LayerDebug {
    pub q_last: Vec<f32>,
    pub k_all: Vec<f32>,
    pub v_all: Vec<f32>,
    pub context_last: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct ForwardOutputs {
    pub logits: Vec<f32>,
    pub final_hidden_norm_last: Vec<f32>,
    pub layer_debug: Vec<LayerDebug>,
}

pub fn forward_with_debug(model: &ModelJson, token_ids: &[u32]) -> Result<ForwardOutputs> {
    let cfg = &model.model_config;
    let seq_len = token_ids.len();
    if seq_len == 0 {
        bail!("empty token sequence");
    }
    if seq_len > cfg.max_seq_len {
        bail!(
            "seq_len {} exceeds max_seq_len {}",
            seq_len,
            cfg.max_seq_len
        );
    }

    let d = cfg.d_model;
    // x stores the residual stream for all timesteps, row-major [T, D].
    let mut x = vec![0.0f32; seq_len * d];

    // Token + position embedding lookup:
    // x[t] = tok_emb[token_t] + pos_emb[t]
    for (t, &tok) in token_ids.iter().enumerate() {
        let tok_id = tok as usize;
        if tok_id >= cfg.vocab_size {
            bail!("token id {} out of range", tok_id);
        }
        let tok_off = tok_id * d;
        let pos_off = t * d;
        for i in 0..d {
            x[t * d + i] = model.weights.token_embedding.data[tok_off + i]
                + model.weights.position_embedding.data[pos_off + i];
        }
    }

    let mut layer_debug = Vec::with_capacity(cfg.n_layers);
    for layer in &model.weights.layers {
        // Pre-attention layer norm.
        let h_ln1 =
            apply_layer_norm_rows(&x, seq_len, d, &layer.ln1_weight.data, &layer.ln1_bias.data);

        // Linear projections into Q, K, V spaces (all [T, D]).
        let q_all = apply_linear_rows(
            &h_ln1,
            seq_len,
            d,
            d,
            &layer.wq_weight.data,
            &layer.wq_bias.data,
        );
        let k_all = apply_linear_rows(
            &h_ln1,
            seq_len,
            d,
            d,
            &layer.wk_weight.data,
            &layer.wk_bias.data,
        );
        let v_all = apply_linear_rows(
            &h_ln1,
            seq_len,
            d,
            d,
            &layer.wv_weight.data,
            &layer.wv_bias.data,
        );

        // Compute causal attention context for every timestep t using prefix [0..t].
        let mut ctx_all = vec![0.0f32; seq_len * d];
        for t in 0..seq_len {
            let q = &q_all[t * d..(t + 1) * d];
            let (weights, ctx) = attention_for_timestep(q, &k_all, &v_all, t + 1, d);
            let _ = weights; // Kept for readability; we store context debug only.
            ctx_all[t * d..(t + 1) * d].copy_from_slice(&ctx);
        }

        // Attention output projection + residual add.
        let attn_out = apply_linear_rows(
            &ctx_all,
            seq_len,
            d,
            d,
            &layer.wo_weight.data,
            &layer.wo_bias.data,
        );
        for i in 0..x.len() {
            x[i] += attn_out[i];
        }

        // Feed-forward network: LN -> Linear -> GELU -> Linear -> residual add.
        let h_ln2 =
            apply_layer_norm_rows(&x, seq_len, d, &layer.ln2_weight.data, &layer.ln2_bias.data);
        let ff1 = apply_linear_rows(
            &h_ln2,
            seq_len,
            d,
            model.model_config.ffn_hidden,
            &layer.ff1_weight.data,
            &layer.ff1_bias.data,
        );
        let ff1 = ff1.into_iter().map(gelu).collect::<Vec<_>>();
        let ff2 = apply_linear_rows(
            &ff1,
            seq_len,
            model.model_config.ffn_hidden,
            d,
            &layer.ff2_weight.data,
            &layer.ff2_bias.data,
        );
        for i in 0..x.len() {
            x[i] += ff2[i];
        }

        // Expose last-token debug tensors so GPU path can be compared layer-wise.
        let t_last = seq_len - 1;
        let q_last = q_all[t_last * d..(t_last + 1) * d].to_vec();
        let (_, context_last) = attention_for_timestep(&q_last, &k_all, &v_all, seq_len, d);
        layer_debug.push(LayerDebug {
            q_last,
            k_all,
            v_all,
            context_last,
        });
    }

    // Final layer norm and logits for last position.
    let x_norm = apply_layer_norm_rows(
        &x,
        seq_len,
        d,
        &model.weights.ln_f_weight.data,
        &model.weights.ln_f_bias.data,
    );
    let last = x_norm[(seq_len - 1) * d..seq_len * d].to_vec();
    let logits = apply_linear_single(
        &last,
        d,
        model.model_config.vocab_size,
        &model.weights.lm_head_weight.data,
        &model.weights.lm_head_bias.data,
    );

    Ok(ForwardOutputs {
        logits,
        final_hidden_norm_last: last,
        layer_debug,
    })
}

fn attention_for_timestep(
    q: &[f32],
    k_all: &[f32],
    v_all: &[f32],
    prefix_len: usize,
    d: usize,
) -> (Vec<f32>, Vec<f32>) {
    // scale = 1/sqrt(D) keeps score magnitudes stable as D grows.
    let scale = 1.0f32 / (d as f32).sqrt();
    let mut scores = vec![0.0f32; prefix_len];
    for t in 0..prefix_len {
        let k = &k_all[t * d..(t + 1) * d];
        scores[t] = dot(q, k) * scale;
    }

    let weights = softmax(&scores);

    // Context vector is weighted sum of value vectors.
    let mut ctx = vec![0.0f32; d];
    for t in 0..prefix_len {
        let v = &v_all[t * d..(t + 1) * d];
        for i in 0..d {
            ctx[i] += weights[t] * v[i];
        }
    }
    (weights, ctx)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

fn softmax(xs: &[f32]) -> Vec<f32> {
    // Numerically stable softmax:
    // softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x)).
    let mut max_v = f32::NEG_INFINITY;
    for &v in xs {
        if v > max_v {
            max_v = v;
        }
    }

    let mut out = vec![0.0f32; xs.len()];
    let mut sum = 0.0f32;
    for (i, &v) in xs.iter().enumerate() {
        let e = (v - max_v).exp();
        out[i] = e;
        sum += e;
    }
    let inv = 1.0f32 / sum.max(1e-12);
    for v in &mut out {
        *v *= inv;
    }
    out
}

fn gelu(x: f32) -> f32 {
    // GELU approximation used in many transformer implementations.
    // Exact GELU uses erf; this tanh form is faster and close numerically.
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
}

fn apply_layer_norm_rows(
    input: &[f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    beta: &[f32],
) -> Vec<f32> {
    // Per-row normalization over feature axis:
    // y = gamma * (x - mean)/sqrt(var + eps) + beta
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &input[r * cols..(r + 1) * cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let mut var = 0.0f32;
        for &v in row {
            let d = v - mean;
            var += d * d;
        }
        var /= cols as f32;
        let inv_std = 1.0f32 / (var + 1e-5).sqrt();
        for c in 0..cols {
            let norm = (row[c] - mean) * inv_std;
            out[r * cols + c] = norm * gamma[c] + beta[c];
        }
    }
    out
}

fn apply_linear_rows(
    input: &[f32],
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    weight: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    // Matrix-vector application row by row.
    let mut out = vec![0.0f32; rows * out_dim];
    for r in 0..rows {
        let row = &input[r * in_dim..(r + 1) * in_dim];
        let y = apply_linear_single(row, in_dim, out_dim, weight, bias);
        out[r * out_dim..(r + 1) * out_dim].copy_from_slice(&y);
    }
    out
}

fn apply_linear_single(
    input: &[f32],
    in_dim: usize,
    out_dim: usize,
    weight: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    // y_o = b_o + sum_i W[o, i] * x_i
    let mut out = vec![0.0f32; out_dim];
    for o in 0..out_dim {
        let mut acc = bias[o];
        let row = &weight[o * in_dim..(o + 1) * in_dim];
        for i in 0..in_dim {
            acc += row[i] * input[i];
        }
        out[o] = acc;
    }
    out
}
