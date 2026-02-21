// =============================================================================
// LESSON 9 MODEL JSON LOADER/VALIDATOR
// =============================================================================
// Beginner context:
// - Training produces a checkpoint (binary tensors).
// - export_json converts that checkpoint to a portable JSON artifact.
// - This file loads that JSON, verifies checksum integrity, and validates
//   tensor shapes before inference starts.
//
// Why this matters:
// - GPU kernels assume specific shapes and contiguous layouts.
// - If one shape is wrong, math becomes invalid or memory access can break.
// =============================================================================

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ArrayF32 {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LayerWeights {
    pub ln1_weight: ArrayF32,
    pub ln1_bias: ArrayF32,
    pub wq_weight: ArrayF32,
    pub wq_bias: ArrayF32,
    pub wk_weight: ArrayF32,
    pub wk_bias: ArrayF32,
    pub wv_weight: ArrayF32,
    pub wv_bias: ArrayF32,
    pub wo_weight: ArrayF32,
    pub wo_bias: ArrayF32,
    pub ln2_weight: ArrayF32,
    pub ln2_bias: ArrayF32,
    pub ff1_weight: ArrayF32,
    pub ff1_bias: ArrayF32,
    pub ff2_weight: ArrayF32,
    pub ff2_bias: ArrayF32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Weights {
    pub token_embedding: ArrayF32,
    pub position_embedding: ArrayF32,
    pub layers: Vec<LayerWeights>,
    pub ln_f_weight: ArrayF32,
    pub ln_f_bias: ArrayF32,
    pub lm_head_weight: ArrayF32,
    pub lm_head_bias: ArrayF32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SpecialTokenIds {
    pub pad: u32,
    pub bos: u32,
    pub eos: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelConfigJson {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub ffn_hidden: usize,
    pub special_token_ids: SpecialTokenIds,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainingMetadata {
    pub steps: usize,
    pub seed: u64,
    pub final_train_loss: f32,
    pub final_val_loss: f32,
    pub train_tokens_seen: usize,
    pub exported_unix_seconds: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TokenizerConfig {
    pub r#type: String,
    pub vocab_size: usize,
    pub pad_id: u32,
    pub bos_id: u32,
    pub eos_id: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelJson {
    pub format_version: String,
    pub model_config: ModelConfigJson,
    pub tokenizer: TokenizerConfig,
    pub training_metadata: TrainingMetadata,
    pub weights: Weights,
    pub checksum: String,
}

impl ModelJson {
    /// Verifies checksum over canonical JSON bytes of `weights`.
    ///
    /// If this fails, the file may be corrupted or edited after export.
    pub fn verify_checksum(&self) -> Result<()> {
        let canonical = serde_json::to_vec(&self.weights)?;
        let mut hasher = Sha256::new();
        hasher.update(&canonical);
        let actual = format!("{:x}", hasher.finalize());
        if actual != self.checksum {
            bail!(
                "checksum mismatch: expected {}, got {}",
                self.checksum,
                actual
            );
        }
        Ok(())
    }

    /// Validates tensor shapes against model config.
    ///
    /// This catches mismatches such as wrong vocab size, wrong layer count,
    /// or wrong projection dimensions before runtime compute starts.
    pub fn validate_shapes(&self) -> Result<()> {
        let cfg = &self.model_config;
        expect_shape(
            &self.weights.token_embedding,
            &[cfg.vocab_size, cfg.d_model],
            "token_embedding",
        )?;
        expect_shape(
            &self.weights.position_embedding,
            &[cfg.max_seq_len, cfg.d_model],
            "position_embedding",
        )?;
        if self.weights.layers.len() != cfg.n_layers {
            bail!(
                "layer count mismatch: cfg={} json={}",
                cfg.n_layers,
                self.weights.layers.len()
            );
        }
        for (i, layer) in self.weights.layers.iter().enumerate() {
            let p = format!("layers[{i}]");
            expect_shape(
                &layer.ln1_weight,
                &[cfg.d_model],
                &format!("{p}.ln1_weight"),
            )?;
            expect_shape(&layer.ln1_bias, &[cfg.d_model], &format!("{p}.ln1_bias"))?;
            expect_shape(
                &layer.wq_weight,
                &[cfg.d_model, cfg.d_model],
                &format!("{p}.wq_weight"),
            )?;
            expect_shape(&layer.wq_bias, &[cfg.d_model], &format!("{p}.wq_bias"))?;
            expect_shape(
                &layer.wk_weight,
                &[cfg.d_model, cfg.d_model],
                &format!("{p}.wk_weight"),
            )?;
            expect_shape(&layer.wk_bias, &[cfg.d_model], &format!("{p}.wk_bias"))?;
            expect_shape(
                &layer.wv_weight,
                &[cfg.d_model, cfg.d_model],
                &format!("{p}.wv_weight"),
            )?;
            expect_shape(&layer.wv_bias, &[cfg.d_model], &format!("{p}.wv_bias"))?;
            expect_shape(
                &layer.wo_weight,
                &[cfg.d_model, cfg.d_model],
                &format!("{p}.wo_weight"),
            )?;
            expect_shape(&layer.wo_bias, &[cfg.d_model], &format!("{p}.wo_bias"))?;
            expect_shape(
                &layer.ln2_weight,
                &[cfg.d_model],
                &format!("{p}.ln2_weight"),
            )?;
            expect_shape(&layer.ln2_bias, &[cfg.d_model], &format!("{p}.ln2_bias"))?;
            expect_shape(
                &layer.ff1_weight,
                &[cfg.ffn_hidden, cfg.d_model],
                &format!("{p}.ff1_weight"),
            )?;
            expect_shape(&layer.ff1_bias, &[cfg.ffn_hidden], &format!("{p}.ff1_bias"))?;
            expect_shape(
                &layer.ff2_weight,
                &[cfg.d_model, cfg.ffn_hidden],
                &format!("{p}.ff2_weight"),
            )?;
            expect_shape(&layer.ff2_bias, &[cfg.d_model], &format!("{p}.ff2_bias"))?;
        }
        expect_shape(&self.weights.ln_f_weight, &[cfg.d_model], "ln_f_weight")?;
        expect_shape(&self.weights.ln_f_bias, &[cfg.d_model], "ln_f_bias")?;
        expect_shape(
            &self.weights.lm_head_weight,
            &[cfg.vocab_size, cfg.d_model],
            "lm_head_weight",
        )?;
        expect_shape(
            &self.weights.lm_head_bias,
            &[cfg.vocab_size],
            "lm_head_bias",
        )?;
        Ok(())
    }
}

fn expect_shape(arr: &ArrayF32, expected: &[usize], name: &str) -> Result<()> {
    if arr.shape != expected {
        bail!(
            "shape mismatch for {name}: expected {:?} got {:?}",
            expected,
            arr.shape
        );
    }
    let expected_len: usize = expected.iter().product();
    if arr.data.len() != expected_len {
        bail!(
            "data length mismatch for {name}: expected {} got {}",
            expected_len,
            arr.data.len()
        );
    }
    Ok(())
}

pub fn load_model_json(path: &Path) -> Result<ModelJson> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let model: ModelJson = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {} as model JSON", path.display()))?;
    model.verify_checksum()?;
    model.validate_shapes()?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksum_round_trip() {
        let weights = Weights {
            token_embedding: ArrayF32 {
                shape: vec![1, 1],
                data: vec![1.0],
            },
            position_embedding: ArrayF32 {
                shape: vec![1, 1],
                data: vec![2.0],
            },
            layers: vec![],
            ln_f_weight: ArrayF32 {
                shape: vec![1],
                data: vec![1.0],
            },
            ln_f_bias: ArrayF32 {
                shape: vec![1],
                data: vec![0.0],
            },
            lm_head_weight: ArrayF32 {
                shape: vec![1, 1],
                data: vec![1.0],
            },
            lm_head_bias: ArrayF32 {
                shape: vec![1],
                data: vec![0.0],
            },
        };
        let canonical = serde_json::to_vec(&weights).unwrap();
        let mut hasher = Sha256::new();
        hasher.update(&canonical);
        let checksum = format!("{:x}", hasher.finalize());

        let model = ModelJson {
            format_version: "lesson9.v1".into(),
            model_config: ModelConfigJson {
                vocab_size: 1,
                max_seq_len: 1,
                d_model: 1,
                n_layers: 0,
                ffn_hidden: 1,
                special_token_ids: SpecialTokenIds {
                    pad: 256,
                    bos: 257,
                    eos: 258,
                },
            },
            tokenizer: TokenizerConfig {
                r#type: "byte_level".into(),
                vocab_size: 1,
                pad_id: 256,
                bos_id: 257,
                eos_id: 258,
            },
            training_metadata: TrainingMetadata {
                steps: 1,
                seed: 1,
                final_train_loss: 0.0,
                final_val_loss: 0.0,
                train_tokens_seen: 1,
                exported_unix_seconds: 1,
            },
            weights,
            checksum,
        };

        model.verify_checksum().unwrap();
    }
}
