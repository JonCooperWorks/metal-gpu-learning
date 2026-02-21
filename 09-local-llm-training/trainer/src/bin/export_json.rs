// =============================================================================
// LESSON 9 CHECKPOINT -> JSON EXPORTER
// =============================================================================
// Purpose:
// - Load trained tensors from Candle checkpoint.
// - Repackage them into lesson JSON format used by Rust/Metal inference.
// - Compute checksum over weights payload for integrity verification.
//
// Why checksum exists:
// - Inference validates that weights were not corrupted or edited.
// =============================================================================

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use lesson9_trainer::model::{ModelConfig, TinyTransformer};
use lesson9_trainer::tokenizer::{BOS_ID, EOS_ID, PAD_ID, VOCAB_SIZE};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Serialize, Deserialize)]
struct CheckpointMeta {
    vocab_size: usize,
    max_seq_len: usize,
    d_model: usize,
    n_layers: usize,
    ffn_hidden: usize,
    steps: usize,
    seed: u64,
    final_train_loss: f32,
    final_val_loss: f32,
    train_tokens_seen: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ArrayF32 {
    shape: Vec<usize>,
    data: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct LayerWeights {
    ln1_weight: ArrayF32,
    ln1_bias: ArrayF32,
    wq_weight: ArrayF32,
    wq_bias: ArrayF32,
    wk_weight: ArrayF32,
    wk_bias: ArrayF32,
    wv_weight: ArrayF32,
    wv_bias: ArrayF32,
    wo_weight: ArrayF32,
    wo_bias: ArrayF32,
    ln2_weight: ArrayF32,
    ln2_bias: ArrayF32,
    ff1_weight: ArrayF32,
    ff1_bias: ArrayF32,
    ff2_weight: ArrayF32,
    ff2_bias: ArrayF32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Weights {
    token_embedding: ArrayF32,
    position_embedding: ArrayF32,
    layers: Vec<LayerWeights>,
    ln_f_weight: ArrayF32,
    ln_f_bias: ArrayF32,
    lm_head_weight: ArrayF32,
    lm_head_bias: ArrayF32,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelConfigJson {
    vocab_size: usize,
    max_seq_len: usize,
    d_model: usize,
    n_layers: usize,
    ffn_hidden: usize,
    special_token_ids: SpecialTokenIds,
}

#[derive(Debug, Serialize, Deserialize)]
struct SpecialTokenIds {
    pad: u32,
    bos: u32,
    eos: u32,
}

#[derive(Debug, Serialize, Deserialize)]
struct TrainingMetadata {
    steps: usize,
    seed: u64,
    final_train_loss: f32,
    final_val_loss: f32,
    train_tokens_seen: usize,
    exported_unix_seconds: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelJson {
    format_version: String,
    model_config: ModelConfigJson,
    tokenizer: TokenizerConfig,
    training_metadata: TrainingMetadata,
    weights: Weights,
    checksum: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct TokenizerConfig {
    r#type: String,
    vocab_size: usize,
    pad_id: u32,
    bos_id: u32,
    eos_id: u32,
}

fn usage() {
    eprintln!("Usage: export_json --checkpoint <checkpoint.safetensors> --out <model.json>");
}

fn parse_args() -> Result<(PathBuf, PathBuf)> {
    let args: Vec<String> = env::args().collect();
    let mut checkpoint = None;
    let mut out = None;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--checkpoint" => {
                checkpoint = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--out" => {
                out = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            _ => {
                usage();
                bail!("unknown argument {}", args[i]);
            }
        }
    }

    Ok((
        checkpoint.context("missing --checkpoint")?,
        out.context("missing --out")?,
    ))
}

fn tensor_to_array(t: &Tensor) -> Result<ArrayF32> {
    let t = t.to_dtype(DType::F32)?;
    let shape = t.dims().to_vec();
    let data = t.flatten_all()?.to_vec1::<f32>()?;
    Ok(ArrayF32 { shape, data })
}

fn tensor_by_name(vars: &HashMap<String, candle_core::Var>, name: &str) -> Result<ArrayF32> {
    let var = vars
        .get(name)
        .with_context(|| format!("missing tensor in checkpoint: {name}"))?;
    tensor_to_array(var.as_tensor())
}

fn main() -> Result<()> {
    // This exporter turns Candle checkpoint tensors into the lesson JSON schema.
    let (checkpoint_path, out_path) = parse_args()?;

    let meta_path = PathBuf::from(format!("{}.meta.json", checkpoint_path.display()));
    let ckpt_meta: CheckpointMeta =
        serde_json::from_slice(&fs::read(&meta_path).with_context(|| {
            format!("missing checkpoint metadata file {}", meta_path.display())
        })?)?;

    if ckpt_meta.vocab_size != VOCAB_SIZE {
        bail!(
            "checkpoint vocab size mismatch: {} != {}",
            ckpt_meta.vocab_size,
            VOCAB_SIZE
        );
    }

    let mut varmap = VarMap::new();
    let cfg = ModelConfig {
        vocab_size: ckpt_meta.vocab_size,
        max_seq_len: ckpt_meta.max_seq_len,
        d_model: ckpt_meta.d_model,
        n_layers: ckpt_meta.n_layers,
        ffn_hidden: ckpt_meta.ffn_hidden,
    };
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    let _model = TinyTransformer::new(cfg.clone(), vb)?;
    varmap.load(&checkpoint_path)?;

    let vars_guard = varmap.data().lock().unwrap();

    // Collect per-layer tensors into JSON struct.
    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let p = format!("blocks.{i}");
        layers.push(LayerWeights {
            ln1_weight: tensor_by_name(&vars_guard, &format!("{p}.ln1.weight"))?,
            ln1_bias: tensor_by_name(&vars_guard, &format!("{p}.ln1.bias"))?,
            wq_weight: tensor_by_name(&vars_guard, &format!("{p}.wq.weight"))?,
            wq_bias: tensor_by_name(&vars_guard, &format!("{p}.wq.bias"))?,
            wk_weight: tensor_by_name(&vars_guard, &format!("{p}.wk.weight"))?,
            wk_bias: tensor_by_name(&vars_guard, &format!("{p}.wk.bias"))?,
            wv_weight: tensor_by_name(&vars_guard, &format!("{p}.wv.weight"))?,
            wv_bias: tensor_by_name(&vars_guard, &format!("{p}.wv.bias"))?,
            wo_weight: tensor_by_name(&vars_guard, &format!("{p}.wo.weight"))?,
            wo_bias: tensor_by_name(&vars_guard, &format!("{p}.wo.bias"))?,
            ln2_weight: tensor_by_name(&vars_guard, &format!("{p}.ln2.weight"))?,
            ln2_bias: tensor_by_name(&vars_guard, &format!("{p}.ln2.bias"))?,
            ff1_weight: tensor_by_name(&vars_guard, &format!("{p}.ff1.weight"))?,
            ff1_bias: tensor_by_name(&vars_guard, &format!("{p}.ff1.bias"))?,
            ff2_weight: tensor_by_name(&vars_guard, &format!("{p}.ff2.weight"))?,
            ff2_bias: tensor_by_name(&vars_guard, &format!("{p}.ff2.bias"))?,
        });
    }

    let weights = Weights {
        token_embedding: tensor_by_name(&vars_guard, "tok_emb.weight")?,
        position_embedding: tensor_by_name(&vars_guard, "pos_emb.weight")?,
        layers,
        ln_f_weight: tensor_by_name(&vars_guard, "ln_f.weight")?,
        ln_f_bias: tensor_by_name(&vars_guard, "ln_f.bias")?,
        lm_head_weight: tensor_by_name(&vars_guard, "lm_head.weight")?,
        lm_head_bias: tensor_by_name(&vars_guard, "lm_head.bias")?,
    };

    // Checksum over canonical serialized weights bytes.
    let checksum = {
        let canonical = serde_json::to_vec(&weights)?;
        let mut hasher = Sha256::new();
        hasher.update(&canonical);
        format!("{:x}", hasher.finalize())
    };

    let exported_unix_seconds = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let json = ModelJson {
        format_version: "lesson9.v1".to_string(),
        model_config: ModelConfigJson {
            vocab_size: cfg.vocab_size,
            max_seq_len: cfg.max_seq_len,
            d_model: cfg.d_model,
            n_layers: cfg.n_layers,
            ffn_hidden: cfg.ffn_hidden,
            special_token_ids: SpecialTokenIds {
                pad: PAD_ID,
                bos: BOS_ID,
                eos: EOS_ID,
            },
        },
        tokenizer: TokenizerConfig {
            r#type: "byte_level".to_string(),
            vocab_size: VOCAB_SIZE,
            pad_id: PAD_ID,
            bos_id: BOS_ID,
            eos_id: EOS_ID,
        },
        training_metadata: TrainingMetadata {
            steps: ckpt_meta.steps,
            seed: ckpt_meta.seed,
            final_train_loss: ckpt_meta.final_train_loss,
            final_val_loss: ckpt_meta.final_val_loss,
            train_tokens_seen: ckpt_meta.train_tokens_seen,
            exported_unix_seconds,
        },
        weights,
        checksum,
    };

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out_path, serde_json::to_vec_pretty(&json)?)?;
    println!("Exported JSON model: {}", out_path.display());

    Ok(())
}
