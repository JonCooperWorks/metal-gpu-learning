// =============================================================================
// LESSON 8: Basic LLM -- Tiny Autoregressive Generator (GPU Attention Core)
// =============================================================================
//
// WHAT THIS MINI LM IS
// -----------------------------------------------------------------------------
// This program is a teaching model that demonstrates the core next-token loop:
//   prompt -> tokenize -> attention -> logits -> next token -> repeat
//
// It is intentionally tiny, deterministic, and fully inspectable:
// - fixed small vocabulary
// - synthetic (non-trained) weights
// - single-head attention
// - one sequence at a time
//
// WHAT IS MISSING VS CHATGPT
// -----------------------------------------------------------------------------
// ChatGPT-like systems add many major capabilities that are NOT in this lesson:
// - huge trained models (many layers, many heads, massive vocabularies)
// - instruction tuning / RLHF for helpful conversational behavior
// - strong world knowledge learned from large corpora
// - long-context handling and optimized KV-cache decoding
// - safety systems, policy layers, and tool orchestration
// - multimodal capabilities and production reliability infrastructure
//
// So this lesson captures the core mechanics of generation, but not the scale,
// training, alignment, knowledge depth, or product features of ChatGPT.
//
// This lesson demonstrates the core inference loop of an LLM in a minimal form:
//   1) tokenize prompt
//   2) run attention for the current token
//   3) project context to logits
//   4) pick next token
//   5) append token and repeat
//
// Design goals:
// - Keep model tiny and deterministic.
// - Keep GPU work focused on the key primitive: scaled dot-product attention.
// - Keep code heavily annotated so each step is readable in sequence.
//
// Important simplifications compared with production LLMs:
// - single head (no multi-head)
// - one sequence at a time (no batch)
// - no KV cache (we recompute per step for clarity)
// - fixed tiny vocabulary
// - synthetic deterministic weights
// =============================================================================

use metal::*;
use objc::rc::autoreleasepool;
use std::collections::HashMap;
use std::env;
use std::f32;
use std::mem;

const SHADER_SOURCE: &str = include_str!("lesson8.metal");

const DEFAULT_PROMPT: &str = "i like";
const DEFAULT_GENERATE_TOKENS: u32 = 16;
const DEFAULT_MAX_SEQ: u32 = 32;
const DEFAULT_TOP_K: u32 = 5;
const DEFAULT_TEMPERATURE: f32 = 1.0;
const REPETITION_PENALTY_BASE: f32 = 0.08;
const REPETITION_LOOKBACK: usize = 12;
const MIN_GENERATED_BEFORE_EOS: usize = 6;
const EOS_BONUS_AFTER_MIN: f32 = 0.10;

// Kernel-side caps in lesson8.metal. Keep Rust side aligned.
const KERNEL_MAX_SEQ: u32 = 64;
const KERNEL_MAX_D_MODEL: u32 = 64;

const D_MODEL: usize = 32;

const BOS_ID: usize = 1;
const EOS_ID: usize = 2;
const UNK_ID: usize = 0;

const VOCAB: [&str; 128] = [
    "<unk>",
    "<bos>",
    "<eos>",
    "hello",
    "hi",
    "hey",
    "chat",
    "assistant",
    "user",
    "i",
    "you",
    "we",
    "they",
    "it",
    "am",
    "are",
    "is",
    "was",
    "be",
    "have",
    "do",
    "can",
    "will",
    "should",
    "please",
    "thanks",
    "thank",
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "help",
    "explain",
    "show",
    "tell",
    "build",
    "make",
    "create",
    "write",
    "read",
    "update",
    "fix",
    "test",
    "run",
    "code",
    "project",
    "lesson",
    "gpu",
    "metal",
    "rust",
    "llm",
    "model",
    "models",
    "attention",
    "token",
    "tokens",
    "prompt",
    "response",
    "generate",
    "generation",
    "predict",
    "next",
    "word",
    "text",
    "simple",
    "useful",
    "tiny",
    "basic",
    "real",
    "example",
    "demo",
    "kernel",
    "pipeline",
    "loop",
    "step",
    "math",
    "vector",
    "matrix",
    "softmax",
    "logits",
    "probability",
    "context",
    "embedding",
    "weights",
    "train",
    "training",
    "inference",
    "memory",
    "speed",
    "fast",
    "slow",
    "good",
    "better",
    "best",
    "great",
    "clear",
    "clean",
    "safe",
    "deterministic",
    "repeat",
    "chatbot",
    "conversation",
    "question",
    "answer",
    "about",
    "for",
    "with",
    "from",
    "to",
    "in",
    "on",
    "of",
    "and",
    "or",
    "not",
    "this",
    "that",
    "these",
    "those",
    "a",
    "an",
    "the",
    "yes",
    "no",
    "ok",
    "done",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ValidateMode {
    On,
    Off,
}

impl ValidateMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::On => "on",
            Self::Off => "off",
        }
    }
}

#[derive(Debug)]
struct Config {
    prompt: String,
    generate_tokens: u32,
    max_seq: u32,
    top_k: u32,
    temperature: f32,
    validate: ValidateMode,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct AttentionParams {
    seq_len: u32,
    d_model: u32,
    scale: f32,
    _pad: f32,
}

#[derive(Debug)]
struct ModelWeights {
    embeddings: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    w_vocab: Vec<f32>,
}

#[derive(Debug)]
struct AttentionOutputs {
    weights: Vec<f32>,
    context: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
struct TopKEntry {
    token_id: usize,
    prob: f32,
}

fn print_usage() {
    println!("Usage: basic-llm [--prompt TEXT] [--generate-tokens N] [--max-seq N] [--top-k N] [--temperature T] [--validate on|off]");
    println!();
    println!("Defaults:");
    println!("  --prompt \"{}\"", DEFAULT_PROMPT);
    println!("  --generate-tokens {}", DEFAULT_GENERATE_TOKENS);
    println!("  --max-seq {}", DEFAULT_MAX_SEQ);
    println!("  --top-k {}", DEFAULT_TOP_K);
    println!("  --temperature {:.1}", DEFAULT_TEMPERATURE);
    println!("  --validate on");
}

fn parse_u32(name: &str, value: &str) -> Result<u32, String> {
    value
        .parse::<u32>()
        .map_err(|_| format!("Invalid value for {name}: '{value}'"))
}

fn parse_f32(name: &str, value: &str) -> Result<f32, String> {
    value
        .parse::<f32>()
        .map_err(|_| format!("Invalid value for {name}: '{value}'"))
}

fn parse_args() -> Result<Config, String> {
    let mut prompt = DEFAULT_PROMPT.to_string();
    let mut generate_tokens = DEFAULT_GENERATE_TOKENS;
    let mut max_seq = DEFAULT_MAX_SEQ;
    let mut top_k = DEFAULT_TOP_K;
    let mut temperature = DEFAULT_TEMPERATURE;
    let mut validate = ValidateMode::On;

    let args: Vec<String> = env::args().collect();
    let mut i = 1usize;

    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            "--prompt" => {
                let v = args
                    .get(i + 1)
                    .ok_or_else(|| "Missing value for --prompt".to_string())?;
                prompt = v.clone();
                i += 2;
            }
            "--generate-tokens" => {
                let v = args
                    .get(i + 1)
                    .ok_or_else(|| "Missing value for --generate-tokens".to_string())?;
                generate_tokens = parse_u32("--generate-tokens", v)?;
                i += 2;
            }
            "--max-seq" => {
                let v = args
                    .get(i + 1)
                    .ok_or_else(|| "Missing value for --max-seq".to_string())?;
                max_seq = parse_u32("--max-seq", v)?;
                i += 2;
            }
            "--top-k" => {
                let v = args
                    .get(i + 1)
                    .ok_or_else(|| "Missing value for --top-k".to_string())?;
                top_k = parse_u32("--top-k", v)?;
                i += 2;
            }
            "--temperature" => {
                let v = args
                    .get(i + 1)
                    .ok_or_else(|| "Missing value for --temperature".to_string())?;
                temperature = parse_f32("--temperature", v)?;
                i += 2;
            }
            "--validate" => {
                let v = args
                    .get(i + 1)
                    .ok_or_else(|| "Missing value for --validate".to_string())?;
                validate = match v.as_str() {
                    "on" => ValidateMode::On,
                    "off" => ValidateMode::Off,
                    _ => return Err(format!("Invalid --validate value '{v}'")),
                };
                i += 2;
            }
            flag => return Err(format!("Unknown argument: {flag}")),
        }
    }

    if max_seq == 0 {
        return Err("--max-seq must be > 0".to_string());
    }
    if max_seq > KERNEL_MAX_SEQ {
        return Err(format!(
            "--max-seq must be <= {KERNEL_MAX_SEQ} (kernel cap)"
        ));
    }
    if generate_tokens > max_seq {
        return Err("--generate-tokens must be <= --max-seq".to_string());
    }
    if top_k == 0 {
        return Err("--top-k must be > 0".to_string());
    }
    if !(temperature > 0.0) {
        return Err("--temperature must be > 0".to_string());
    }

    Ok(Config {
        prompt,
        generate_tokens,
        max_seq,
        top_k,
        temperature,
        validate,
    })
}

// Deterministic tiny-weight generator.
// We avoid randomness and external files, so every run is reproducible.
fn det_weight(i: usize, j: usize, salt: usize) -> f32 {
    let x = (i as f32 * 0.173) + (j as f32 * 0.117) + (salt as f32 * 0.071);
    let y = (i as f32 * 0.037) - (j as f32 * 0.019) + (salt as f32 * 0.023);
    (x.sin() * 0.12) + (y.cos() * 0.08)
}

fn build_weights() -> ModelWeights {
    let vocab = VOCAB.len();
    let d = D_MODEL;

    let mut embeddings = vec![0.0f32; vocab * d];
    let mut wq = vec![0.0f32; d * d];
    let mut wk = vec![0.0f32; d * d];
    let mut wv = vec![0.0f32; d * d];
    let mut wo = vec![0.0f32; d * d];
    let mut w_vocab = vec![0.0f32; d * vocab];

    // Token embeddings.
    for tok in 0..vocab {
        for dim in 0..d {
            embeddings[tok * d + dim] = det_weight(tok, dim, 1);
        }
    }

    // Q/K/V/O projections.
    for i in 0..d {
        for j in 0..d {
            wq[i * d + j] = det_weight(i, j, 11);
            wk[i * d + j] = det_weight(i, j, 13);
            wv[i * d + j] = det_weight(i, j, 17);
            wo[i * d + j] = det_weight(i, j, 19);
        }
    }

    // Vocab projection: map hidden context to token logits.
    for dim in 0..d {
        for tok in 0..vocab {
            w_vocab[dim * vocab + tok] = det_weight(dim, tok, 23);
        }
    }

    ModelWeights {
        embeddings,
        wq,
        wk,
        wv,
        wo,
        w_vocab,
    }
}

fn vocab_map() -> HashMap<&'static str, usize> {
    let mut map = HashMap::with_capacity(VOCAB.len());
    for (i, token) in VOCAB.iter().enumerate() {
        map.insert(*token, i);
    }
    map
}

fn tokenize(prompt: &str, map: &HashMap<&'static str, usize>) -> Vec<usize> {
    let mut ids = Vec::new();
    for raw in prompt.split_whitespace() {
        // Trim common punctuation so "hello," and "hello" map to the same token.
        let token = raw
            .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '\'')
            .to_ascii_lowercase();
        if token.is_empty() {
            continue;
        }
        let id = map.get(token.as_str()).copied().unwrap_or(UNK_ID);
        ids.push(id);
    }
    ids
}

fn decode_tokens(ids: &[usize]) -> String {
    let mut out = Vec::new();
    for &id in ids {
        if id == BOS_ID {
            continue;
        }
        if id == EOS_ID {
            break;
        }
        out.push(VOCAB.get(id).copied().unwrap_or("<unk>"));
    }
    out.join(" ")
}

// Row-vector times matrix:
// input[cols_in] * matrix[cols_in x cols_out] -> output[cols_out]
fn matvec(input: &[f32], matrix: &[f32], cols_in: usize, cols_out: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; cols_out];
    for out_j in 0..cols_out {
        let mut acc = 0.0f32;
        for in_i in 0..cols_in {
            acc += input[in_i] * matrix[in_i * cols_out + out_j];
        }
        out[out_j] = acc;
    }
    out
}

fn stable_softmax(logits: &[f32]) -> Vec<f32> {
    let mut max_v = f32::NEG_INFINITY;
    for &v in logits {
        if v > max_v {
            max_v = v;
        }
    }

    let mut exps = vec![0.0f32; logits.len()];
    let mut sum = 0.0f32;
    for (i, &v) in logits.iter().enumerate() {
        let e = (v - max_v).exp();
        exps[i] = e;
        sum += e;
    }

    let inv = 1.0f32 / sum.max(1e-12);
    for v in &mut exps {
        *v *= inv;
    }
    exps
}

fn attention_cpu_reference(
    q: &[f32],
    k_all: &[f32],
    v_all: &[f32],
    seq_len: usize,
) -> AttentionOutputs {
    let d = D_MODEL;
    let scale = 1.0f32 / (d as f32).sqrt();

    // scores[i] = dot(q, k_i) * scale
    let mut scores = vec![0.0f32; seq_len];
    for i in 0..seq_len {
        let mut dot_qk = 0.0f32;
        for dim in 0..d {
            dot_qk += q[dim] * k_all[i * d + dim];
        }
        scores[i] = dot_qk * scale;
    }

    let weights = stable_softmax(&scores);

    // context[d] = sum_i weights[i] * v_i[d]
    let mut context = vec![0.0f32; d];
    for dim in 0..d {
        let mut acc = 0.0f32;
        for i in 0..seq_len {
            acc += weights[i] * v_all[i * d + dim];
        }
        context[dim] = acc;
    }

    AttentionOutputs { weights, context }
}

fn find_top_k(probs: &[f32], k: usize) -> Vec<TopKEntry> {
    let mut entries: Vec<TopKEntry> = probs
        .iter()
        .enumerate()
        .map(|(id, &prob)| TopKEntry { token_id: id, prob })
        .collect();

    entries.sort_by(|a, b| {
        b.prob
            .partial_cmp(&a.prob)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    entries.truncate(k.min(entries.len()));
    entries
}

fn argmax(values: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = values[0];
    for (i, &v) in values.iter().enumerate().skip(1) {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

// Apply deterministic decoding biases so greedy decoding does not collapse
// into repeating one token forever on this tiny synthetic model.
fn apply_decoding_biases(logits: &mut [f32], token_ids: &[usize], generated: &[usize]) {
    // Early in generation, suppress EOS so we produce at least a short reply.
    if generated.len() < MIN_GENERATED_BEFORE_EOS {
        logits[EOS_ID] -= 1.0;
    } else {
        // After minimum length, gently encourage EOS so generation can end.
        logits[EOS_ID] += EOS_BONUS_AFTER_MIN;
    }

    // Penalize recently used tokens, with stronger penalty for repeats.
    // This keeps generation deterministic but less repetitive.
    let start = token_ids.len().saturating_sub(REPETITION_LOOKBACK);
    let mut counts = vec![0u32; VOCAB.len()];
    for &tok in &token_ids[start..] {
        if tok != BOS_ID && tok != EOS_ID {
            counts[tok] += 1;
        }
    }
    for (tok, &count) in counts.iter().enumerate() {
        if count > 0 {
            let penalty = REPETITION_PENALTY_BASE * count as f32;
            logits[tok] -= penalty;
        }
    }
}

fn validate_context(cpu: &[f32], gpu: &[f32], tol: f32) -> Result<(), String> {
    let mut max_err = 0.0f32;
    let mut max_idx = 0usize;

    for i in 0..cpu.len() {
        let e = (cpu[i] - gpu[i]).abs();
        if e > max_err {
            max_err = e;
            max_idx = i;
        }
    }

    if max_err > tol {
        return Err(format!(
            "Validation failed: max_abs_err={max_err:.6e} at dim {max_idx} (cpu={:.6e}, gpu={:.6e})",
            cpu[max_idx], gpu[max_idx]
        ));
    }

    Ok(())
}

fn validate_vector(name: &str, cpu: &[f32], gpu: &[f32], tol: f32) -> Result<(), String> {
    let mut max_err = 0.0f32;
    let mut max_idx = 0usize;

    for i in 0..cpu.len() {
        let e = (cpu[i] - gpu[i]).abs();
        if e > max_err {
            max_err = e;
            max_idx = i;
        }
    }

    if max_err > tol {
        return Err(format!(
            "Validation failed ({name}): max_abs_err={max_err:.6e} at idx {max_idx} (cpu={:.6e}, gpu={:.6e})",
            cpu[max_idx], gpu[max_idx]
        ));
    }

    Ok(())
}

fn as_mut_slice<T>(buffer: &Buffer, len: usize) -> &mut [T] {
    unsafe { std::slice::from_raw_parts_mut(buffer.contents() as *mut T, len) }
}

fn as_slice<T>(buffer: &Buffer, len: usize) -> &[T] {
    unsafe { std::slice::from_raw_parts(buffer.contents() as *const T, len) }
}

fn main() {
    autoreleasepool(|| {
        if let Err(e) = run() {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    });
}

fn run() -> Result<(), String> {
    let config = parse_args()?;

    if D_MODEL as u32 > KERNEL_MAX_D_MODEL {
        return Err(format!(
            "D_MODEL={} exceeds kernel max {}",
            D_MODEL, KERNEL_MAX_D_MODEL
        ));
    }

    let vocab_index = vocab_map();
    let mut token_ids = vec![BOS_ID];
    token_ids.extend(tokenize(&config.prompt, &vocab_index));

    // This lesson is single-turn. We enforce max sequence upfront.
    let required = token_ids.len() as u32 + config.generate_tokens;
    if required > config.max_seq {
        return Err(format!(
            "Prompt tokens ({}) + generate-tokens ({}) exceeds --max-seq ({})",
            token_ids.len(),
            config.generate_tokens,
            config.max_seq
        ));
    }

    let device =
        Device::system_default().ok_or_else(|| "No Metal-capable GPU found".to_string())?;
    println!("Using GPU: {}", device.name());
    println!(
        "validate={}  d_model={}  vocab={}",
        config.validate.as_str(),
        D_MODEL,
        VOCAB.len()
    );

    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(SHADER_SOURCE, &options)
        .map_err(|e| format!("Failed to compile Metal shader: {e}"))?;
    let function = library
        .get_function("single_head_attention_step", None)
        .map_err(|e| format!("Failed to find kernel 'single_head_attention_step': {e}"))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| format!("Failed to create compute pipeline: {e}"))?;

    let queue = device.new_command_queue();

    let weights = build_weights();

    // Reusable staging buffers sized to max_seq.
    let q_buf = device.new_buffer(
        (D_MODEL * mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let k_buf = device.new_buffer(
        (config.max_seq as usize * D_MODEL * mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let v_buf = device.new_buffer(
        (config.max_seq as usize * D_MODEL * mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let attn_buf = device.new_buffer(
        (config.max_seq as usize * mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let context_buf = device.new_buffer(
        (D_MODEL * mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let params_buf = device.new_buffer(
        mem::size_of::<AttentionParams>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    println!("\nPrompt: \"{}\"", config.prompt);
    println!("Prompt token ids (with <bos>): {:?}", token_ids);

    // Keep generated tokens separate so final output is easy to inspect.
    let mut generated_only: Vec<usize> = Vec::new();

    for step in 0..config.generate_tokens {
        let seq_len = token_ids.len();
        if seq_len == 0 {
            return Err("Internal error: seq_len == 0".to_string());
        }

        // ---------------------------------------------------------------------
        // Build hidden states for all tokens currently in the prefix.
        // ---------------------------------------------------------------------
        // We keep this on CPU in this lesson so the GPU focus stays on
        // the attention primitive itself.
        let mut q_all = vec![0.0f32; seq_len * D_MODEL];
        let mut k_all = vec![0.0f32; seq_len * D_MODEL];
        let mut v_all = vec![0.0f32; seq_len * D_MODEL];

        for (pos, &tok) in token_ids.iter().enumerate() {
            let emb = &weights.embeddings[tok * D_MODEL..(tok + 1) * D_MODEL];
            let q = matvec(emb, &weights.wq, D_MODEL, D_MODEL);
            let k = matvec(emb, &weights.wk, D_MODEL, D_MODEL);
            let v = matvec(emb, &weights.wv, D_MODEL, D_MODEL);
            q_all[pos * D_MODEL..(pos + 1) * D_MODEL].copy_from_slice(&q);
            k_all[pos * D_MODEL..(pos + 1) * D_MODEL].copy_from_slice(&k);
            v_all[pos * D_MODEL..(pos + 1) * D_MODEL].copy_from_slice(&v);
        }

        let q_current = &q_all[(seq_len - 1) * D_MODEL..seq_len * D_MODEL];

        // CPU reference for correctness checks.
        let cpu_attention = attention_cpu_reference(q_current, &k_all, &v_all, seq_len);

        // ---------------------------------------------------------------------
        // Upload one-step attention inputs to GPU.
        // ---------------------------------------------------------------------
        as_mut_slice::<f32>(&q_buf, D_MODEL).copy_from_slice(q_current);
        as_mut_slice::<f32>(&k_buf, seq_len * D_MODEL).copy_from_slice(&k_all);
        as_mut_slice::<f32>(&v_buf, seq_len * D_MODEL).copy_from_slice(&v_all);

        let params = AttentionParams {
            seq_len: seq_len as u32,
            d_model: D_MODEL as u32,
            scale: 1.0f32 / (D_MODEL as f32).sqrt(),
            _pad: 0.0,
        };
        as_mut_slice::<AttentionParams>(&params_buf, 1)[0] = params;

        // ---------------------------------------------------------------------
        // Dispatch GPU attention kernel for this autoregressive step.
        // ---------------------------------------------------------------------
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&q_buf), 0);
        enc.set_buffer(1, Some(&k_buf), 0);
        enc.set_buffer(2, Some(&v_buf), 0);
        enc.set_buffer(3, Some(&attn_buf), 0);
        enc.set_buffer(4, Some(&context_buf), 0);
        enc.set_buffer(5, Some(&params_buf), 0);

        // One threadgroup is enough for this tiny lesson workload.
        let tg_size = MTLSize::new(64, 1, 1);
        let grid_size = MTLSize::new(64, 1, 1);
        enc.dispatch_threads(grid_size, tg_size);
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        let gpu_attn = as_slice::<f32>(&attn_buf, seq_len).to_vec();
        let gpu_context = as_slice::<f32>(&context_buf, D_MODEL).to_vec();

        if config.validate == ValidateMode::On {
            validate_context(&cpu_attention.context, &gpu_context, 1e-4)?;
            validate_vector("attention weights", &cpu_attention.weights, &gpu_attn, 1e-4)?;
        }

        // ---------------------------------------------------------------------
        // Finish language-model step on CPU: context -> logits -> next token.
        // ---------------------------------------------------------------------
        let context_post = matvec(&gpu_context, &weights.wo, D_MODEL, D_MODEL);
        let logits = matvec(&context_post, &weights.w_vocab, D_MODEL, VOCAB.len());
        let mut decode_logits = logits.clone();
        apply_decoding_biases(&mut decode_logits, &token_ids, &generated_only);

        // Decoding rule for this lesson: greedy on raw logits.
        let next_token = argmax(&decode_logits);

        // We still compute temperature-adjusted probabilities for explainability.
        let mut temp_logits = decode_logits;
        let inv_temp = 1.0f32 / config.temperature;
        for v in &mut temp_logits {
            *v *= inv_temp;
        }
        let probs = stable_softmax(&temp_logits);
        let top_k = find_top_k(&probs, config.top_k as usize);

        println!("\nStep {}", step + 1);
        println!(
            "  seq_len={}  chosen={} ({})",
            seq_len, next_token, VOCAB[next_token]
        );
        println!("  attention over prefix:");
        for (i, &w) in gpu_attn.iter().enumerate() {
            let tok = VOCAB[token_ids[i]];
            println!("    pos {:>2} {:>14}: {:.5}", i, tok, w);
        }
        println!("  top-{} probabilities:", top_k.len());
        for entry in top_k {
            println!(
                "    {:>14} (id {:>2}): {:.5}",
                VOCAB[entry.token_id], entry.token_id, entry.prob
            );
        }

        token_ids.push(next_token);
        generated_only.push(next_token);

        if next_token == EOS_ID {
            println!("  emitted <eos>; stopping generation early.");
            break;
        }
    }

    // Final user-facing text view.
    let full_text = decode_tokens(&token_ids);
    let generated_text = decode_tokens(&generated_only);

    println!("\n=== Final Output ===");
    let prompt_only_ids: Vec<usize> = {
        let mut ids = vec![BOS_ID];
        ids.extend(tokenize(&config.prompt, &vocab_index));
        ids
    };
    println!("Prompt only:       {}", decode_tokens(&prompt_only_ids));
    println!("Generated only:    {}", generated_text);
    println!("Prompt + generated: {}", full_text);

    if config.validate == ValidateMode::On {
        println!(
            "\nValidation: PASS (GPU attention context matches CPU reference within tolerance)"
        );
    }

    Ok(())
}
