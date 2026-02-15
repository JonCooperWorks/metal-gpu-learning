#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

cargo run -p lesson9-trainer --bin prepare_data -- \
  --input 09-local-llm-training/data/sample_corpus.txt \
  --out 09-local-llm-training/artifacts/data

cargo run -p lesson9-trainer --bin train -- \
  --data 09-local-llm-training/artifacts/data \
  --profile auto --auto-hardware on \
  --out 09-local-llm-training/artifacts/checkpoint.safetensors

cargo run -p lesson9-trainer --bin export_json -- \
  --checkpoint 09-local-llm-training/artifacts/checkpoint.safetensors \
  --out 09-local-llm-training/artifacts/model.json

cargo run -p local-llm-training -- \
  --model-json 09-local-llm-training/artifacts/model.json \
  --prompt "hello gpu" --generate-tokens 32 --max-seq 64 --validate on
