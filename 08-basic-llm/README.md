# Lesson 08: Basic LLM -- Tiny Autoregressive Generator

## Problem Statement

Build a minimal next-token generator that feels LLM-like while staying small
and understandable.

This lesson implements the core loop:

1. tokenize prompt
2. compute attention for current token
3. project to logits
4. choose next token
5. append and repeat

The lesson now uses a fixed 128-word vocabulary so common prompts like
`hello chat` are tokenized directly instead of falling back to `<unk>`.

## Pipeline Diagram

`tokens -> embeddings -> Q/K/V -> attention -> context -> logits -> next token -> repeat`

## What This Teaches

- how autoregressive generation works
- why attention is central to LLM inference
- how CPU and GPU can split responsibilities
- how to validate GPU math against a CPU reference

## What Real LLMs Add

- many layers and many heads
- trained checkpoints with billions of parameters
- KV cache for fast long-context decoding
- batching, quantization, and optimized kernels
- richer tokenization (BPE / sentencepiece / byte-level)

## CLI

- `--prompt "<text>"` (default: `"i like"`)
- `--generate-tokens <u32>` (default: `16`)
- `--max-seq <u32>` (default: `32`, max `64`)
- `--top-k <u32>` (default: `5`, display only)
- `--temperature <f32>` (default: `1.0`, display only)
- `--validate on|off` (default: `on`)

## Run

```bash
cargo run --release -p basic-llm
cargo run --release -p basic-llm -- --prompt "i like rust" --generate-tokens 8
cargo run --release -p basic-llm -- --validate on --top-k 3
```

## Output Interpretation

- `chosen=...` is the greedy next token from raw logits.
- `attention over prefix` shows which prior token positions influenced the step.
- `top-k probabilities` is temperature-adjusted only for inspection.
- If `<eos>` appears, generation stops early.
- With `--validate on`, CPU/GPU attention context must match within tolerance.

## Notes

This is an educational synthetic model. It is not trained and not suitable for
real NLP tasks.
