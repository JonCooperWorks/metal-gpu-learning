# Lesson 09: Local LLM Training with Candle + Rust/Metal Inference

This lesson is written for readers who are new to GPU programming.
You will train a tiny transformer locally, export model weights to JSON, and run
inference where CPU and Metal GPU computations are compared step-by-step.

## How to Read This Code

Read in this order:

1. `trainer/src/tokenizer.rs` (text -> IDs)
2. `trainer/src/model.rs` (transformer forward math)
3. `trainer/src/bin/train.rs` (training loop)
4. `trainer/src/bin/export_json.rs` (checkpoint -> portable JSON)
5. `src/model_json.rs` (load + validate model artifact)
6. `src/cpu_ref.rs` (CPU reference forward pass)
7. `src/lesson9.metal` (GPU kernels)
8. `src/main.rs` (runtime orchestration + validation)

## CPU vs GPU Responsibilities

- CPU:
  - full reference forward pass
  - model artifact loading and shape checks
  - control flow for generation
- GPU:
  - attention for last token (`attention_last_token`)
  - logits projection (`logits_projection`)

This split keeps the lesson understandable while still showing real GPU kernels.

## End-to-End Commands

```bash
# 1) Prepare dataset tokens
cargo run -p lesson9-trainer --bin prepare_data -- \
  --input 09-local-llm-training/data/sample_corpus.txt \
  --out 09-local-llm-training/artifacts/data

# 2) Train model
cargo run -p lesson9-trainer --bin train -- \
  --data 09-local-llm-training/artifacts/data \
  --steps 1200 --seq-len 64 --batch-size 24 --d-model 128 --layers 2 \
  --out 09-local-llm-training/artifacts/checkpoint.safetensors

# 3) Export JSON artifact
cargo run -p lesson9-trainer --bin export_json -- \
  --checkpoint 09-local-llm-training/artifacts/checkpoint.safetensors \
  --out 09-local-llm-training/artifacts/model.json

# 4) Run Rust/Metal inference with validation
cargo run -p local-llm-training -- \
  --model-json 09-local-llm-training/artifacts/model.json \
  --prompt "hello gpu" --generate-tokens 32 --max-seq 64 --validate on
```

## Math Primer

## 1) Attention score
For a query vector `q` and key vector `k_t`:

`score_t = (q · k_t) / sqrt(d_model)`

`1/sqrt(d_model)` keeps score magnitudes stable as model width grows.

## 2) Stable softmax

`softmax(score_t) = exp(score_t - max_score) / sum_j exp(score_j - max_score)`

Subtracting `max_score` avoids overflow in `exp`.

## 3) Context vector

`context = Σ_t softmax(score)_t * v_t`

Each value vector `v_t` contributes in proportion to attention probability.

## 4) LayerNorm

`y = gamma * (x - mean) / sqrt(var + eps) + beta`

LayerNorm stabilizes activations and helps optimization.

## 5) Cross-entropy training loss
For target token `y` and logits `z`:

`loss = -log(softmax(z)[y])`

Lower loss means the model assigns higher probability to the correct next token.

## 6) GELU activation
GELU is a smooth nonlinearity used in transformer MLP blocks.
This lesson uses the common tanh approximation in CPU reference/trainer code.

## Equation -> Code Map

- Attention score + softmax + context:
  - `09-local-llm-training/src/lesson9.metal`
  - `09-local-llm-training/src/cpu_ref.rs`
- LayerNorm + residual + MLP:
  - `09-local-llm-training/trainer/src/model.rs`
  - `09-local-llm-training/src/cpu_ref.rs`
- Cross-entropy in training loop:
  - `09-local-llm-training/trainer/src/bin/train.rs`
- Greedy decoding (`argmax` logits):
  - `09-local-llm-training/src/main.rs`

## Glossary

- `tensor`: multi-dimensional numeric array
- `kernel`: GPU function launched over many threads
- `threadgroup`: cooperating group of GPU threads sharing scratch memory
- `dispatch`: launching a GPU kernel
- `logits`: raw class scores before softmax
- `causal mask`: blocks attention to future tokens
- `residual connection`: adds block input back to block output

## Common Mistakes

- Using model JSON with mismatched shape/checksum.
- Setting `--max-seq` above model max sequence length.
- Confusing logits with probabilities.
- Ignoring validation mismatches when editing kernels.
- Using too-small dataset so training cannot generalize.

## Troubleshooting

- `dataset too small`: add more text lines.
- `checksum mismatch`: re-export JSON with `export_json`.
- `Metal mismatch`: run with `--validate on` and inspect step/layer mismatch.
- slow training: reduce `--batch-size` or `--steps`.
