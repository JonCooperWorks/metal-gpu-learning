// =============================================================================
// LESSON 9 INFERENCE RUNTIME (Rust + Metal)
// =============================================================================
// Beginner context:
// - This binary performs autoregressive generation one token at a time.
// - CPU computes full reference tensors (for clarity + validation).
// - GPU executes two focused kernels per generation step:
//   1) attention for last token,
//   2) logits projection from hidden state to vocabulary.
//
// Why this split exists in the lesson:
// - It keeps GPU code small and inspectable.
// - It lets us compare GPU values against CPU reference every step.
// =============================================================================

mod cpu_ref;
mod model_json;

use anyhow::{bail, Context, Result};
use metal::*;
use model_json::load_model_json;
use objc::rc::autoreleasepool;
use std::env;
use std::mem;
use std::path::PathBuf;

const SHADER_SOURCE: &str = include_str!("lesson9.metal");

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct AttentionParams {
    seq_len: u32,
    d_model: u32,
    scale: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct LogitsParams {
    d_model: u32,
    vocab_size: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ValidateMode {
    On,
    Off,
}

#[derive(Debug)]
struct Config {
    model_json: PathBuf,
    prompt: String,
    generate_tokens: usize,
    max_seq: usize,
    validate: ValidateMode,
}

fn usage() {
    eprintln!(
        "Usage: local-llm-training --model-json <path> [--prompt TEXT] [--generate-tokens N] [--max-seq N] [--validate on|off]"
    );
}

// -----------------------------------------------------------------------------
// CLI parsing / validation
// -----------------------------------------------------------------------------
fn parse_args() -> Result<Config> {
    let mut model_json = None;
    let mut prompt = "hello gpu".to_string();
    let mut generate_tokens = 32usize;
    let mut max_seq = 64usize;
    let mut validate = ValidateMode::On;

    let args: Vec<String> = env::args().collect();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                usage();
                std::process::exit(0);
            }
            "--model-json" => {
                model_json = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--prompt" => {
                prompt = args.get(i + 1).context("missing --prompt value")?.to_string();
                i += 2;
            }
            "--generate-tokens" => {
                generate_tokens = args
                    .get(i + 1)
                    .context("missing --generate-tokens value")?
                    .parse()?;
                i += 2;
            }
            "--max-seq" => {
                max_seq = args.get(i + 1).context("missing --max-seq value")?.parse()?;
                i += 2;
            }
            "--validate" => {
                validate = match args.get(i + 1).context("missing --validate value")?.as_str() {
                    "on" => ValidateMode::On,
                    "off" => ValidateMode::Off,
                    bad => bail!("invalid validate mode: {bad}"),
                };
                i += 2;
            }
            flag => {
                usage();
                bail!("unknown argument: {flag}");
            }
        }
    }

    let model_json = model_json.context("missing --model-json")?;
    if generate_tokens == 0 || max_seq == 0 {
        bail!("--generate-tokens and --max-seq must be > 0");
    }

    Ok(Config {
        model_json,
        prompt,
        generate_tokens,
        max_seq,
        validate,
    })
}

fn encode_prompt(prompt: &str, bos_id: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(prompt.len() + 1);
    out.push(bos_id);
    out.extend(prompt.as_bytes().iter().map(|b| *b as u32));
    out
}

fn decode_generated(ids: &[u32], eos_id: u32) -> String {
    let mut bytes = Vec::with_capacity(ids.len());
    for &id in ids {
        if id == eos_id {
            break;
        }
        if id <= 255 {
            bytes.push(id as u8);
        }
    }
    String::from_utf8_lossy(&bytes).to_string()
}

// Greedy decoding picks the token with largest logit.
fn argmax(xs: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = xs[0];
    for (i, &v) in xs.iter().enumerate().skip(1) {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    let mut m = 0.0f32;
    for i in 0..a.len() {
        m = m.max((a[i] - b[i]).abs());
    }
    m
}

// SAFETY: caller must ensure the Metal buffer length matches requested element
// count and type T.
fn as_mut_slice<T>(buffer: &Buffer, len: usize) -> &mut [T] {
    unsafe { std::slice::from_raw_parts_mut(buffer.contents() as *mut T, len) }
}

fn as_slice<T>(buffer: &Buffer, len: usize) -> &[T] {
    unsafe { std::slice::from_raw_parts(buffer.contents() as *const T, len) }
}

fn main() {
    autoreleasepool(|| {
        if let Err(e) = run() {
            eprintln!("Error: {e:#}");
            std::process::exit(1);
        }
    });
}

fn run() -> Result<()> {
    let cfg = parse_args()?;
    let model = load_model_json(&cfg.model_json)?;

    let model_cfg = &model.model_config;
    if model_cfg.max_seq_len > 64 {
        bail!("kernel MAX_SEQ is 64, model has {}", model_cfg.max_seq_len);
    }
    if model_cfg.d_model > 256 {
        bail!("kernel MAX_D_MODEL is 256, model has {}", model_cfg.d_model);
    }
    if cfg.max_seq > model_cfg.max_seq_len {
        bail!("--max-seq exceeds model max_seq_len");
    }

    // -------------------------------------------------------------------------
    // Metal setup: compile shaders and create compute pipelines.
    // -------------------------------------------------------------------------
    let device = Device::system_default().context("no Metal-capable GPU found")?;
    println!("Using GPU: {}", device.name());

    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(SHADER_SOURCE, &options)
        .map_err(|e| anyhow::anyhow!("failed to compile lesson9.metal: {e}"))?;

    let attention_fn = library
        .get_function("attention_last_token", None)
        .map_err(|e| anyhow::anyhow!("missing kernel attention_last_token: {e}"))?;
    let logits_fn = library
        .get_function("logits_projection", None)
        .map_err(|e| anyhow::anyhow!("missing kernel logits_projection: {e}"))?;

    let attention_pipeline = device
        .new_compute_pipeline_state_with_function(&attention_fn)
        .map_err(|e| anyhow::anyhow!("failed to create attention pipeline: {e}"))?;
    let logits_pipeline = device
        .new_compute_pipeline_state_with_function(&logits_fn)
        .map_err(|e| anyhow::anyhow!("failed to create logits pipeline: {e}"))?;

    let queue = device.new_command_queue();

    let d = model_cfg.d_model;
    let vocab = model_cfg.vocab_size;
    let max_seq = cfg.max_seq;

    // -------------------------------------------------------------------------
    // GPU buffers
    // -------------------------------------------------------------------------
    let q_buf = device.new_buffer((d * mem::size_of::<f32>()) as u64, MTLResourceOptions::StorageModeShared);
    let k_buf = device.new_buffer(
        (max_seq * d * mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let v_buf = device.new_buffer(
        (max_seq * d * mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let attn_w_buf = device.new_buffer((max_seq * mem::size_of::<f32>()) as u64, MTLResourceOptions::StorageModeShared);
    let context_buf = device.new_buffer((d * mem::size_of::<f32>()) as u64, MTLResourceOptions::StorageModeShared);
    let attn_params_buf = device.new_buffer(
        mem::size_of::<AttentionParams>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let hidden_buf = device.new_buffer((d * mem::size_of::<f32>()) as u64, MTLResourceOptions::StorageModeShared);
    let lm_head_w_buf = device.new_buffer(
        (vocab * d * mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let lm_head_b_buf = device.new_buffer((vocab * mem::size_of::<f32>()) as u64, MTLResourceOptions::StorageModeShared);
    let logits_buf = device.new_buffer((vocab * mem::size_of::<f32>()) as u64, MTLResourceOptions::StorageModeShared);
    let logits_params_buf = device.new_buffer(
        mem::size_of::<LogitsParams>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // LM head remains constant during inference, so we copy once.
    as_mut_slice::<f32>(&lm_head_w_buf, vocab * d).copy_from_slice(&model.weights.lm_head_weight.data);
    as_mut_slice::<f32>(&lm_head_b_buf, vocab).copy_from_slice(&model.weights.lm_head_bias.data);

    let mut token_ids = encode_prompt(&cfg.prompt, model_cfg.special_token_ids.bos);
    if token_ids.len() + cfg.generate_tokens > max_seq {
        bail!("prompt length + generation exceeds max seq cap");
    }

    println!("Prompt: {:?}", cfg.prompt);
    println!("Prompt token count (incl <bos>): {}", token_ids.len());

    let mut generated = Vec::new();

    // -------------------------------------------------------------------------
    // Autoregressive loop: each step predicts exactly one next token.
    // -------------------------------------------------------------------------
    for step in 0..cfg.generate_tokens {
        // CPU reference forward pass for current prefix.
        let cpu = cpu_ref::forward_with_debug(&model, &token_ids)?;

        // GPU attention validation layer-by-layer.
        for layer_idx in 0..model_cfg.n_layers {
            let debug = &cpu.layer_debug[layer_idx];
            let seq_len = token_ids.len();

            as_mut_slice::<f32>(&q_buf, d).copy_from_slice(&debug.q_last);
            as_mut_slice::<f32>(&k_buf, seq_len * d).copy_from_slice(&debug.k_all[..seq_len * d]);
            as_mut_slice::<f32>(&v_buf, seq_len * d).copy_from_slice(&debug.v_all[..seq_len * d]);

            // Math callout: scale = 1/sqrt(D) stabilizes dot-product magnitudes.
            as_mut_slice::<AttentionParams>(&attn_params_buf, 1)[0] = AttentionParams {
                seq_len: seq_len as u32,
                d_model: d as u32,
                scale: 1.0 / (d as f32).sqrt(),
                _pad: 0.0,
            };

            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&attention_pipeline);
            enc.set_buffer(0, Some(&q_buf), 0);
            enc.set_buffer(1, Some(&k_buf), 0);
            enc.set_buffer(2, Some(&v_buf), 0);
            enc.set_buffer(3, Some(&attn_w_buf), 0);
            enc.set_buffer(4, Some(&context_buf), 0);
            enc.set_buffer(5, Some(&attn_params_buf), 0);

            // One threadgroup is enough for this educational kernel because
            // seq_len and d_model are capped small.
            let threads = d.max(seq_len).min(256) as u64;
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(threads, 1, 1));
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            if cfg.validate == ValidateMode::On {
                let gpu_ctx = as_slice::<f32>(&context_buf, d);
                let err = max_abs_diff(gpu_ctx, &debug.context_last);
                if err > 2e-3 {
                    bail!(
                        "layer {} attention mismatch at step {}: max_abs_diff={:.6e}",
                        layer_idx,
                        step,
                        err
                    );
                }
            }
        }

        // GPU logits projection for current last hidden state.
        as_mut_slice::<f32>(&hidden_buf, d).copy_from_slice(&cpu.final_hidden_norm_last);
        as_mut_slice::<LogitsParams>(&logits_params_buf, 1)[0] = LogitsParams {
            d_model: d as u32,
            vocab_size: vocab as u32,
        };

        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&logits_pipeline);
        enc.set_buffer(0, Some(&hidden_buf), 0);
        enc.set_buffer(1, Some(&lm_head_w_buf), 0);
        enc.set_buffer(2, Some(&lm_head_b_buf), 0);
        enc.set_buffer(3, Some(&logits_buf), 0);
        enc.set_buffer(4, Some(&logits_params_buf), 0);

        let tg = logits_pipeline.max_total_threads_per_threadgroup().min(256) as u64;
        enc.dispatch_threads(MTLSize::new(vocab as u64, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let gpu_logits = as_slice::<f32>(&logits_buf, vocab).to_vec();
        if cfg.validate == ValidateMode::On {
            let err = max_abs_diff(&gpu_logits, &cpu.logits);
            if err > 2e-3 {
                bail!("logits mismatch at step {}: max_abs_diff={:.6e}", step, err);
            }
        }

        // Greedy choice: next token is argmax(logits).
        let next = argmax(&gpu_logits) as u32;
        token_ids.push(next);
        generated.push(next);

        let preview = if next <= 255 {
            char::from_u32(next).unwrap_or('?').to_string()
        } else if next == model_cfg.special_token_ids.eos {
            "<eos>".to_string()
        } else {
            format!("<{}>", next)
        };
        println!("step {:>2}: next_token={} {}", step + 1, next, preview);

        if next == model_cfg.special_token_ids.eos {
            println!("Stopping early on <eos>.");
            break;
        }
    }

    let generated_text = decode_generated(&generated, model_cfg.special_token_ids.eos);
    println!("\nGenerated text: {}", generated_text);
    println!("Prompt + generated: {}{}", cfg.prompt, generated_text);

    Ok(())
}
