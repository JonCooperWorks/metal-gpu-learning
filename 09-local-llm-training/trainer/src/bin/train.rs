// =============================================================================
// LESSON 9 TRAINING BINARY (Candle)
// =============================================================================
// Beginner context:
// - This executable performs end-to-end local training for a tiny transformer.
// - Each optimization step does:
//   1) forward pass -> logits,
//   2) cross-entropy loss,
//   3) backward pass,
//   4) AdamW parameter update.
//
// Core math used in this file:
// - Cross-entropy for next-token target y:
//     loss = -log(softmax(logits)[y])
// - Learning-rate schedule:
//     warmup (linear ramp) + cosine decay.
// =============================================================================

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{loss, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use lesson9_trainer::model::{ModelConfig, TinyTransformer};
use lesson9_trainer::tokenizer::{BOS_ID, EOS_ID, PAD_ID, VOCAB_SIZE};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

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
    device: String,
    hardware_profile: String,
    detected_mem_gb: f32,
    detected_perf_cores: usize,
}

#[derive(Debug)]
struct Config {
    data_dir: PathBuf,
    out: PathBuf,
    steps: usize,
    seq_len: usize,
    batch_size: usize,
    d_model: usize,
    layers: usize,
    lr: f64,
    min_lr: f64,
    warmup_steps: usize,
    weight_decay: f64,
    seed: u64,
    eval_every: usize,
    eval_batches: usize,
    time_budget_seconds: u64,
    auto_hardware: bool,
    profile: String,
}

#[derive(Debug, Default)]
struct OverrideFlags {
    steps: bool,
    seq_len: bool,
    batch_size: bool,
    d_model: bool,
    layers: bool,
    lr: bool,
    min_lr: bool,
    warmup_steps: bool,
    eval_every: bool,
    eval_batches: bool,
    time_budget_seconds: bool,
    profile: bool,
}

#[derive(Debug, Clone)]
struct HardwareInfo {
    mem_gb: f32,
    perf_cores: usize,
    logical_cores: usize,
}

fn usage() {
    eprintln!("Usage: train --data <folder> --out <checkpoint.safetensors> [options]");
    eprintln!("Options: --steps --seq-len --batch-size --d-model --layers --lr --min-lr --warmup-steps --eval-every --eval-batches --time-budget-seconds --auto-hardware on|off --profile auto|baseline|balanced|max");
}

fn parse_on_off(v: &str) -> Result<bool> {
    match v {
        "on" => Ok(true),
        "off" => Ok(false),
        _ => bail!("expected on/off, got {v}"),
    }
}

fn parse_args() -> Result<(Config, OverrideFlags)> {
    let mut cfg = Config {
        data_dir: PathBuf::new(),
        out: PathBuf::new(),
        steps: 1200,
        seq_len: 64,
        batch_size: 24,
        d_model: 128,
        layers: 2,
        lr: 3e-4,
        min_lr: 3e-5,
        warmup_steps: 80,
        weight_decay: 0.01,
        seed: 1337,
        eval_every: 100,
        eval_batches: 12,
        time_budget_seconds: 300,
        auto_hardware: true,
        profile: "auto".to_string(),
    };
    let mut flags = OverrideFlags::default();

    let args: Vec<String> = env::args().collect();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                usage();
                std::process::exit(0);
            }
            "--data" => {
                cfg.data_dir = PathBuf::from(args.get(i + 1).context("missing value for --data")?);
                i += 2;
            }
            "--out" => {
                cfg.out = PathBuf::from(args.get(i + 1).context("missing value for --out")?);
                i += 2;
            }
            "--steps" => {
                cfg.steps = args.get(i + 1).context("missing --steps value")?.parse()?;
                flags.steps = true;
                i += 2;
            }
            "--seq-len" => {
                cfg.seq_len = args
                    .get(i + 1)
                    .context("missing --seq-len value")?
                    .parse()?;
                flags.seq_len = true;
                i += 2;
            }
            "--batch-size" => {
                cfg.batch_size = args
                    .get(i + 1)
                    .context("missing --batch-size value")?
                    .parse()?;
                flags.batch_size = true;
                i += 2;
            }
            "--d-model" => {
                cfg.d_model = args
                    .get(i + 1)
                    .context("missing --d-model value")?
                    .parse()?;
                flags.d_model = true;
                i += 2;
            }
            "--layers" => {
                cfg.layers = args.get(i + 1).context("missing --layers value")?.parse()?;
                flags.layers = true;
                i += 2;
            }
            "--lr" => {
                cfg.lr = args.get(i + 1).context("missing --lr value")?.parse()?;
                flags.lr = true;
                i += 2;
            }
            "--min-lr" => {
                cfg.min_lr = args.get(i + 1).context("missing --min-lr value")?.parse()?;
                flags.min_lr = true;
                i += 2;
            }
            "--warmup-steps" => {
                cfg.warmup_steps = args
                    .get(i + 1)
                    .context("missing --warmup-steps value")?
                    .parse()?;
                flags.warmup_steps = true;
                i += 2;
            }
            "--eval-every" => {
                cfg.eval_every = args
                    .get(i + 1)
                    .context("missing --eval-every value")?
                    .parse()?;
                flags.eval_every = true;
                i += 2;
            }
            "--eval-batches" => {
                cfg.eval_batches = args
                    .get(i + 1)
                    .context("missing --eval-batches value")?
                    .parse()?;
                flags.eval_batches = true;
                i += 2;
            }
            "--seed" => {
                cfg.seed = args.get(i + 1).context("missing --seed value")?.parse()?;
                i += 2;
            }
            "--time-budget-seconds" => {
                cfg.time_budget_seconds = args
                    .get(i + 1)
                    .context("missing --time-budget-seconds value")?
                    .parse()?;
                flags.time_budget_seconds = true;
                i += 2;
            }
            "--auto-hardware" => {
                cfg.auto_hardware =
                    parse_on_off(args.get(i + 1).context("missing --auto-hardware value")?)?;
                i += 2;
            }
            "--profile" => {
                cfg.profile = args
                    .get(i + 1)
                    .context("missing --profile value")?
                    .to_string();
                flags.profile = true;
                i += 2;
            }
            flag => {
                usage();
                bail!("unknown argument: {flag}");
            }
        }
    }

    if cfg.data_dir.as_os_str().is_empty() || cfg.out.as_os_str().is_empty() {
        usage();
        bail!("--data and --out are required");
    }
    if cfg.seq_len == 0 || cfg.batch_size == 0 || cfg.steps == 0 {
        bail!("--seq-len, --batch-size and --steps must be > 0");
    }

    Ok((cfg, flags))
}

fn read_u32_tokens(path: &Path) -> Result<Vec<u32>> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    if bytes.len() % 4 != 0 {
        bail!("token file {} has invalid size", path.display());
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn sample_batch(
    tokens: &[u32],
    batch_size: usize,
    seq_len: usize,
    rng: &mut StdRng,
) -> Result<(Vec<u32>, Vec<u32>)> {
    let needed = seq_len + 1;
    if tokens.len() <= needed {
        bail!("token stream too short for requested seq-len");
    }

    let max_start = tokens.len() - needed;
    let mut x = Vec::with_capacity(batch_size * seq_len);
    let mut y = Vec::with_capacity(batch_size * seq_len);

    for _ in 0..batch_size {
        let start = rng.gen_range(0..=max_start);
        x.extend_from_slice(&tokens[start..start + seq_len]);
        y.extend_from_slice(&tokens[start + 1..start + 1 + seq_len]);
    }

    Ok((x, y))
}

// Learning-rate scheduler:
// - Warmup: linearly increase LR to max_lr.
// - Decay: cosine curve down toward min_lr.
fn lr_for_step(
    step: usize,
    total_steps: usize,
    warmup_steps: usize,
    max_lr: f64,
    min_lr: f64,
) -> f64 {
    if step < warmup_steps {
        let pct = (step as f64 + 1.0) / warmup_steps.max(1) as f64;
        return max_lr * pct;
    }
    let remain = (total_steps.saturating_sub(warmup_steps)).max(1);
    let idx = step.saturating_sub(warmup_steps);
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * idx as f64 / remain as f64).cos());
    min_lr + (max_lr - min_lr) * cosine
}

fn eval_loss(
    model: &TinyTransformer,
    val_tokens: &[u32],
    batch_size: usize,
    seq_len: usize,
    eval_batches: usize,
    device: &Device,
    rng: &mut StdRng,
) -> Result<f32> {
    // Validation computes average cross-entropy over sampled mini-batches.
    // Shapes:
    // - x: [B, T]
    // - logits: [B, T, V]
    // - logits2d: [B*T, V]
    // - y: [B*T]
    let mut sum = 0.0f32;
    for _ in 0..eval_batches {
        let (x_ids, y_ids) = sample_batch(val_tokens, batch_size, seq_len, rng)?;
        let x = Tensor::from_slice(&x_ids, (batch_size, seq_len), device)?;
        let y = Tensor::from_slice(&y_ids, (batch_size * seq_len,), device)?;
        let logits = model.forward(&x)?;
        let logits2d = logits.reshape((batch_size * seq_len, model.cfg.vocab_size))?;
        let loss_t = loss::cross_entropy(&logits2d, &y)?;
        sum += loss_t.to_scalar::<f32>()?;
    }
    Ok(sum / eval_batches as f32)
}

fn choose_device() -> Device {
    Device::metal_if_available(0).unwrap_or(Device::Cpu)
}

fn sysctl_u64(name: &str) -> Option<u64> {
    let out = Command::new("sysctl").arg("-n").arg(name).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    s.trim().parse::<u64>().ok()
}

fn detect_hardware() -> HardwareInfo {
    let mem_bytes = sysctl_u64("hw.memsize").unwrap_or(0);
    let perf_cores = sysctl_u64("hw.perflevel0.physicalcpu").unwrap_or(0) as usize;
    let logical = sysctl_u64("hw.logicalcpu").unwrap_or(0) as usize;

    HardwareInfo {
        mem_gb: (mem_bytes as f64 / (1024.0 * 1024.0 * 1024.0)) as f32,
        perf_cores,
        logical_cores: logical,
    }
}

fn pick_profile(cfg: &Config, device: &Device, hw: &HardwareInfo) -> String {
    if cfg.profile != "auto" {
        return cfg.profile.clone();
    }
    if !device.is_metal() {
        return "baseline".to_string();
    }
    if hw.mem_gb >= 48.0 && hw.perf_cores >= 10 {
        "max".to_string()
    } else if hw.mem_gb >= 24.0 {
        "balanced".to_string()
    } else {
        "baseline".to_string()
    }
}

fn apply_hardware_profile(cfg: &mut Config, flags: &OverrideFlags, profile: &str) {
    let set_usize = |slot: &mut usize, val: usize, locked: bool| {
        if !locked {
            *slot = val;
        }
    };
    let set_u64 = |slot: &mut u64, val: u64, locked: bool| {
        if !locked {
            *slot = val;
        }
    };
    let set_f64 = |slot: &mut f64, val: f64, locked: bool| {
        if !locked {
            *slot = val;
        }
    };

    match profile {
        "max" => {
            // Aggressive profile for machines like M4 Max 64GB.
            set_usize(&mut cfg.steps, 30_000, flags.steps);
            set_usize(&mut cfg.seq_len, 64, flags.seq_len);
            set_usize(&mut cfg.batch_size, 128, flags.batch_size);
            set_usize(&mut cfg.d_model, 192, flags.d_model);
            set_usize(&mut cfg.layers, 4, flags.layers);
            set_f64(&mut cfg.lr, 2e-4, flags.lr);
            set_f64(&mut cfg.min_lr, 2e-5, flags.min_lr);
            set_usize(&mut cfg.warmup_steps, 600, flags.warmup_steps);
            set_usize(&mut cfg.eval_every, 300, flags.eval_every);
            set_usize(&mut cfg.eval_batches, 24, flags.eval_batches);
            set_u64(
                &mut cfg.time_budget_seconds,
                2700,
                flags.time_budget_seconds,
            );
        }
        "balanced" => {
            set_usize(&mut cfg.steps, 12_000, flags.steps);
            set_usize(&mut cfg.seq_len, 64, flags.seq_len);
            set_usize(&mut cfg.batch_size, 64, flags.batch_size);
            set_usize(&mut cfg.d_model, 160, flags.d_model);
            set_usize(&mut cfg.layers, 3, flags.layers);
            set_f64(&mut cfg.lr, 2.5e-4, flags.lr);
            set_f64(&mut cfg.min_lr, 2.5e-5, flags.min_lr);
            set_usize(&mut cfg.warmup_steps, 300, flags.warmup_steps);
            set_usize(&mut cfg.eval_every, 200, flags.eval_every);
            set_usize(&mut cfg.eval_batches, 16, flags.eval_batches);
            set_u64(
                &mut cfg.time_budget_seconds,
                1200,
                flags.time_budget_seconds,
            );
        }
        "baseline" => {
            set_usize(&mut cfg.steps, 1200, flags.steps);
            set_usize(&mut cfg.seq_len, 64, flags.seq_len);
            set_usize(&mut cfg.batch_size, 24, flags.batch_size);
            set_usize(&mut cfg.d_model, 128, flags.d_model);
            set_usize(&mut cfg.layers, 2, flags.layers);
            set_f64(&mut cfg.lr, 3e-4, flags.lr);
            set_f64(&mut cfg.min_lr, 3e-5, flags.min_lr);
            set_usize(&mut cfg.warmup_steps, 80, flags.warmup_steps);
            set_usize(&mut cfg.eval_every, 100, flags.eval_every);
            set_usize(&mut cfg.eval_batches, 12, flags.eval_batches);
            set_u64(&mut cfg.time_budget_seconds, 300, flags.time_budget_seconds);
        }
        other => {
            eprintln!("Unknown profile '{other}', falling back to baseline");
            apply_hardware_profile(cfg, flags, "baseline");
        }
    }
}

fn main() -> Result<()> {
    // Full training loop:
    // forward -> loss -> backward -> optimizer step.
    let (mut cfg, flags) = parse_args()?;
    let data_meta: DataMeta = serde_json::from_slice(
        &fs::read(cfg.data_dir.join("meta.json")).context("missing data meta.json")?,
    )?;

    if data_meta.vocab_size != VOCAB_SIZE
        || data_meta.pad_id != PAD_ID
        || data_meta.bos_id != BOS_ID
        || data_meta.eos_id != EOS_ID
    {
        bail!("dataset metadata does not match lesson tokenizer constants");
    }

    let train_tokens = read_u32_tokens(Path::new(&data_meta.train_path))?;
    let val_tokens = read_u32_tokens(Path::new(&data_meta.val_path))?;

    let device = choose_device();
    let dev_name = if device.is_metal() { "metal" } else { "cpu" };
    let hw = detect_hardware();

    let selected_profile = if cfg.auto_hardware {
        let p = pick_profile(&cfg, &device, &hw);
        apply_hardware_profile(&mut cfg, &flags, &p);
        p
    } else {
        "manual".to_string()
    };

    println!("Training device: {dev_name}");
    println!(
        "Detected hardware: mem={:.1} GB perf_cores={} logical_cores={}",
        hw.mem_gb, hw.perf_cores, hw.logical_cores
    );
    println!("Hardware profile: {}", selected_profile);
    println!(
        "Config: steps={} seq_len={} batch={} d_model={} layers={} time_budget={}s",
        cfg.steps, cfg.seq_len, cfg.batch_size, cfg.d_model, cfg.layers, cfg.time_budget_seconds
    );

    let mut rng = StdRng::seed_from_u64(cfg.seed);

    let model_cfg = ModelConfig {
        vocab_size: VOCAB_SIZE,
        max_seq_len: cfg.seq_len,
        d_model: cfg.d_model,
        n_layers: cfg.layers,
        ffn_hidden: cfg.d_model * 4,
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = TinyTransformer::new(model_cfg.clone(), vb)?;

    let params = ParamsAdamW {
        lr: cfg.lr,
        weight_decay: cfg.weight_decay,
        ..ParamsAdamW::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), params)?;

    let batch_tokens = cfg.batch_size * cfg.seq_len;
    let train_start = Instant::now();
    let mut train_tokens_seen = 0usize;
    let mut final_train_loss = 0.0f32;

    // Warm-up probe estimates token throughput.
    // This allows adaptive step capping against a user time budget.
    let warmup_probe = cfg.steps.min(32);
    let probe_start = Instant::now();
    for _ in 0..warmup_probe {
        let (x_ids, y_ids) = sample_batch(&train_tokens, cfg.batch_size, cfg.seq_len, &mut rng)?;
        let x = Tensor::from_slice(&x_ids, (cfg.batch_size, cfg.seq_len), &device)?;
        let y = Tensor::from_slice(&y_ids, (cfg.batch_size * cfg.seq_len,), &device)?;
        let logits = model.forward(&x)?;
        let logits2d = logits.reshape((cfg.batch_size * cfg.seq_len, model_cfg.vocab_size))?;
        // cross_entropy expects class logits [N, C] and targets [N].
        let loss_t = loss::cross_entropy(&logits2d, &y)?;
        opt.backward_step(&loss_t)?;
        final_train_loss = loss_t.to_scalar::<f32>()?;
        train_tokens_seen += batch_tokens;
    }

    let probe_s = probe_start.elapsed().as_secs_f64().max(1e-6);
    let toks_per_s = train_tokens_seen as f64 / probe_s;
    let budgeted_steps =
        ((cfg.time_budget_seconds as f64 * toks_per_s) as usize / batch_tokens).max(1);
    let total_steps = cfg.steps.min(budgeted_steps.max(warmup_probe));
    println!(
        "steps requested={} adaptive_cap={} running_steps={} (~{:.1} tok/s)",
        cfg.steps, budgeted_steps, total_steps, toks_per_s
    );

    let mut step = warmup_probe;
    while step < total_steps {
        // Update LR according to warmup + cosine schedule.
        let lr = lr_for_step(step, total_steps, cfg.warmup_steps, cfg.lr, cfg.min_lr);
        opt.set_learning_rate(lr);

        let (x_ids, y_ids) = sample_batch(&train_tokens, cfg.batch_size, cfg.seq_len, &mut rng)?;
        let x = Tensor::from_slice(&x_ids, (cfg.batch_size, cfg.seq_len), &device)?;
        let y = Tensor::from_slice(&y_ids, (cfg.batch_size * cfg.seq_len,), &device)?;

        let logits = model.forward(&x)?;
        let logits2d = logits.reshape((cfg.batch_size * cfg.seq_len, model_cfg.vocab_size))?;
        let loss_t = loss::cross_entropy(&logits2d, &y)?;
        opt.backward_step(&loss_t)?;

        final_train_loss = loss_t.to_scalar::<f32>()?;
        train_tokens_seen += batch_tokens;

        if step % cfg.eval_every == 0 || step + 1 == total_steps {
            let val = eval_loss(
                &model,
                &val_tokens,
                cfg.batch_size,
                cfg.seq_len,
                cfg.eval_batches,
                &device,
                &mut rng,
            )?;
            let elapsed = train_start.elapsed().as_secs_f64();
            let speed = train_tokens_seen as f64 / elapsed.max(1e-6);
            println!(
                "step {:>5}/{:>5} train_loss={:.4} val_loss={:.4} lr={:.6} tok/s={:.1}",
                step + 1,
                total_steps,
                final_train_loss,
                val,
                lr,
                speed
            );
        }

        step += 1;
    }

    let final_val_loss = eval_loss(
        &model,
        &val_tokens,
        cfg.batch_size,
        cfg.seq_len,
        cfg.eval_batches,
        &device,
        &mut rng,
    )?;

    if let Some(parent) = cfg.out.parent() {
        fs::create_dir_all(parent)?;
    }
    varmap.save(&cfg.out)?;

    let meta = CheckpointMeta {
        vocab_size: model_cfg.vocab_size,
        max_seq_len: model_cfg.max_seq_len,
        d_model: model_cfg.d_model,
        n_layers: model_cfg.n_layers,
        ffn_hidden: model_cfg.ffn_hidden,
        steps: total_steps,
        seed: cfg.seed,
        final_train_loss,
        final_val_loss,
        train_tokens_seen,
        device: dev_name.to_string(),
        hardware_profile: selected_profile,
        detected_mem_gb: hw.mem_gb,
        detected_perf_cores: hw.perf_cores,
    };

    let meta_path = PathBuf::from(format!("{}.meta.json", cfg.out.display()));
    fs::write(&meta_path, serde_json::to_vec_pretty(&meta)?)?;

    println!("Saved checkpoint: {}", cfg.out.display());
    println!("Saved metadata:   {}", meta_path.display());
    println!(
        "Final losses: train={:.4} val={:.4}",
        final_train_loss, final_val_loss
    );

    Ok(())
}
