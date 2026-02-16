#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/Users/jonathan/Development/gpuprogramming"
LESSON_DIR="$ROOT_DIR/10-medicaid-provider-spending"
INPUT_PATH="${1:-$LESSON_DIR/data/medicaid-provider-spending.parquet}"
OUT_DIR="${2:-$LESSON_DIR/data}"
TOP_K_PER_KERNEL="${TOP_K_PER_KERNEL:-500}"
FINAL_TOP_N="${FINAL_TOP_N:-100}"
MIN_VOTES="${MIN_VOTES:-2}"
KERNELS="${KERNELS:-2,3,4,5,6,7,8}"

RAW_REPORT="$OUT_DIR/fraud_raw_report.json"
AGG_JSON="$OUT_DIR/likely_fraud.json"
AGG_CSV="$OUT_DIR/likely_fraud.csv"

mkdir -p "$OUT_DIR"

echo "Running lesson 10 kernels: $KERNELS"
cargo run --release -p medicaid-provider-spending -- \
  --input "$INPUT_PATH" \
  --kernels "$KERNELS" \
  --top-k "$TOP_K_PER_KERNEL" \
  --compare-cpu off \
  --validate off \
  --output-json "$RAW_REPORT"

echo "Combining kernel signals into likely fraud ranking"
python3 "$LESSON_DIR/scripts/combine_kernel_signals.py" \
  --input "$RAW_REPORT" \
  --output-json "$AGG_JSON" \
  --output-csv "$AGG_CSV" \
  --top-n "$FINAL_TOP_N" \
  --min-votes "$MIN_VOTES"


echo "Done."
echo "  Raw kernel report:   $RAW_REPORT"
echo "  Fraud ranking JSON:  $AGG_JSON"
echo "  Fraud ranking CSV:   $AGG_CSV"
