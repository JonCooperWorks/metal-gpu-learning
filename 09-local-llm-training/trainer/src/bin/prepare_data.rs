// =============================================================================
// LESSON 9 DATA PREP BINARY
// =============================================================================
// Purpose:
// - Convert raw text lines into token IDs for training/validation.
// - Persist compact binary token files train.bin and val.bin.
//
// Data format:
// - Each token is stored as little-endian u32 (4 bytes).
// - Metadata JSON records paths and token counts.
// =============================================================================

use anyhow::{bail, Context, Result};
use lesson9_trainer::tokenizer::{encode_with_specials, BOS_ID, EOS_ID, PAD_ID, VOCAB_SIZE};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize, Deserialize)]
struct DataMeta {
    vocab_size: usize,
    pad_id: u32,
    bos_id: u32,
    eos_id: u32,
    train_tokens: usize,
    val_tokens: usize,
    train_path: String,
    val_path: String,
}

fn usage() {
    eprintln!("Usage: prepare_data --input <text-file> --out <folder>");
}

fn parse_args() -> Result<(PathBuf, PathBuf)> {
    let mut input = None;
    let mut out = None;

    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                input = args.get(i + 1).map(PathBuf::from);
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

    let input = input.context("missing --input")?;
    let out = out.context("missing --out")?;
    Ok((input, out))
}

fn normalize_text(input: &str) -> String {
    input
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn write_tokens(path: &Path, tokens: &[u32]) -> Result<()> {
    let mut file = fs::File::create(path)?;
    for &v in tokens {
        file.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn main() -> Result<()> {
    // This step creates next-token training corpora from plain text.
    let (input_path, out_dir) = parse_args()?;
    fs::create_dir_all(&out_dir)?;

    let raw = fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    let norm = normalize_text(&raw);

    let mut all_tokens = Vec::new();
    for line in norm.lines() {
        all_tokens.extend(encode_with_specials(line));
    }

    if all_tokens.len() < 1024 {
        bail!(
            "dataset too small after tokenization: {} tokens",
            all_tokens.len()
        );
    }

    // 90/10 split for train/validation.
    let split = (all_tokens.len() as f32 * 0.9) as usize;
    let train = &all_tokens[..split];
    let val = &all_tokens[split..];

    let train_path = out_dir.join("train.bin");
    let val_path = out_dir.join("val.bin");
    let meta_path = out_dir.join("meta.json");

    write_tokens(&train_path, train)?;
    write_tokens(&val_path, val)?;

    let meta = DataMeta {
        vocab_size: VOCAB_SIZE,
        pad_id: PAD_ID,
        bos_id: BOS_ID,
        eos_id: EOS_ID,
        train_tokens: train.len(),
        val_tokens: val.len(),
        train_path: train_path.display().to_string(),
        val_path: val_path.display().to_string(),
    };

    fs::write(&meta_path, serde_json::to_vec_pretty(&meta)?)?;
    println!("Prepared dataset:");
    println!("  train tokens: {}", train.len());
    println!("  val tokens:   {}", val.len());
    println!("  meta:         {}", meta_path.display());

    Ok(())
}
