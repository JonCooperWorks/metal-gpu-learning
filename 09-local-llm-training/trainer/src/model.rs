// =============================================================================
// LESSON 9 TRAINER MODEL: Tiny causal transformer in Candle
// =============================================================================
// Beginner context:
// - This file defines the trainable neural network used by Lesson 9.
// - Training runs this forward pass many times and updates weights.
//
// Shape notation used throughout:
//   B = batch size
//   T = sequence length
//   D = d_model
//   V = vocab size
//
// Core math in this file:
//   1) Attention score: score = (Q K^T) / sqrt(D)
//   2) Causal mask: future positions get -1e9 before softmax
//   3) Attention probs: softmax(score)
//   4) Context: softmax(score) V
//   5) Residual blocks + LayerNorm + MLP (GELU)
// =============================================================================

use candle_core::{Device, Result, Tensor, D};
use candle_nn::{self as nn, Embedding, LayerNorm, Linear, Module, VarBuilder};

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub ffn_hidden: usize,
}

#[derive(Debug)]
pub struct TransformerBlock {
    pub ln1: LayerNorm,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub ln2: LayerNorm,
    pub ff1: Linear,
    pub ff2: Linear,
}

#[derive(Debug)]
pub struct TinyTransformer {
    pub cfg: ModelConfig,
    pub tok_emb: Embedding,
    pub pos_emb: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub ln_f: LayerNorm,
    pub lm_head: Linear,
}

impl TinyTransformer {
    pub fn new(cfg: ModelConfig, vb: VarBuilder) -> Result<Self> {
        // Embedding tables:
        // tok_emb[token_id] and pos_emb[position] are both length D vectors.
        let tok_emb = nn::embedding(cfg.vocab_size, cfg.d_model, vb.pp("tok_emb"))?;
        let pos_emb = nn::embedding(cfg.max_seq_len, cfg.d_model, vb.pp("pos_emb"))?;

        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let bvb = vb.pp(format!("blocks.{i}"));
            blocks.push(TransformerBlock {
                ln1: nn::layer_norm(cfg.d_model, 1e-5, bvb.pp("ln1"))?,
                wq: nn::linear(cfg.d_model, cfg.d_model, bvb.pp("wq"))?,
                wk: nn::linear(cfg.d_model, cfg.d_model, bvb.pp("wk"))?,
                wv: nn::linear(cfg.d_model, cfg.d_model, bvb.pp("wv"))?,
                wo: nn::linear(cfg.d_model, cfg.d_model, bvb.pp("wo"))?,
                ln2: nn::layer_norm(cfg.d_model, 1e-5, bvb.pp("ln2"))?,
                ff1: nn::linear(cfg.d_model, cfg.ffn_hidden, bvb.pp("ff1"))?,
                ff2: nn::linear(cfg.ffn_hidden, cfg.d_model, bvb.pp("ff2"))?,
            });
        }

        let ln_f = nn::layer_norm(cfg.d_model, 1e-5, vb.pp("ln_f"))?;
        // Final projection from hidden vector h (len D) to logits (len V).
        let lm_head = nn::linear(cfg.d_model, cfg.vocab_size, vb.pp("lm_head"))?;

        Ok(Self {
            cfg,
            tok_emb,
            pos_emb,
            blocks,
            ln_f,
            lm_head,
        })
    }

    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        let (_b_sz, seq_len) = token_ids.dims2()?;
        if seq_len > self.cfg.max_seq_len {
            candle_core::bail!(
                "sequence length {} exceeds max_seq_len {}",
                seq_len,
                self.cfg.max_seq_len
            );
        }

        // token_ids: [B, T] -> tok embeddings: [B, T, D]
        let tok = self.tok_emb.forward(token_ids)?;

        // Position embeddings: [T, D] then unsqueeze -> [1, T, D], broadcast to batch.
        let pos_ids = Tensor::arange(0u32, seq_len as u32, token_ids.device())?;
        let pos = self.pos_emb.forward(&pos_ids)?.unsqueeze(0)?;

        // Residual stream x starts as token + position embedding.
        let mut x = tok.broadcast_add(&pos)?;

        for block in &self.blocks {
            // Pre-norm attention block.
            let h = block.ln1.forward(&x)?;
            let q = block.wq.forward(&h)?;
            let k = block.wk.forward(&h)?;
            let v = block.wv.forward(&h)?;

            // Q: [B,T,D], K^T: [B,D,T] -> scores: [B,T,T]
            // score_{t,s} = dot(q_t, k_s) / sqrt(D)
            let k_t = k.transpose(1, 2)?;
            let scale = 1.0f64 / (self.cfg.d_model as f64).sqrt();
            let mut scores = (q.matmul(&k_t)? * scale)?;

            // Causal mask prevents looking into the future.
            // Future entries get large negative values, so softmax ~ 0 there.
            let mask = causal_mask(seq_len, token_ids.device())?;
            scores = scores.broadcast_add(&mask)?;

            // Softmax on last axis (the source timestep axis s).
            let attn = nn::ops::softmax(&scores, D::Minus1)?;
            // Context: [B,T,T] x [B,T,D] -> [B,T,D]
            let ctx = attn.matmul(&v)?;
            let attn_out = block.wo.forward(&ctx)?;

            // Residual connection: x <- x + attention_output
            x = (x + attn_out)?;

            // Feed-forward block (two linear layers + GELU nonlinearity).
            let h2 = block.ln2.forward(&x)?;
            let ff_hidden = block.ff1.forward(&h2)?.apply(&nn::Activation::Gelu)?;
            let ff = block.ff2.forward(&ff_hidden)?;
            // Residual connection: x <- x + ff_output
            x = (x + ff)?;
        }

        // Final norm + lm head produces logits [B, T, V].
        let x = self.ln_f.forward(&x)?;
        self.lm_head.forward(&x)
    }
}

pub fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    // Mask semantics:
    // - 0.0 means allowed attention.
    // - -1e9 means forbidden future attention (softmax pushes to ~0).
    let mut data = vec![0f32; seq_len * seq_len];
    for t in 0..seq_len {
        for s in (t + 1)..seq_len {
            data[t * seq_len + s] = -1e9;
        }
    }
    // Add a leading singleton batch axis to broadcast across B: [1, T, T].
    Tensor::from_slice(&data, (1, seq_len, seq_len), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_is_causal() -> Result<()> {
        let mask = causal_mask(4, &Device::Cpu)?;
        let vals = mask.squeeze(0)?.to_vec2::<f32>()?;
        assert_eq!(vals[0][0], 0.0);
        assert!(vals[0][1] < -1e8);
        assert_eq!(vals[2][1], 0.0);
        assert!(vals[1][3] < -1e8);
        Ok(())
    }
}
