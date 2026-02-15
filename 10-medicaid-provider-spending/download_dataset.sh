#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="/Users/jonathan/Development/gpuprogramming/10-medicaid-provider-spending/data"
OUT_FILE="$OUT_DIR/medicaid-provider-spending.parquet"
URL="https://stopendataprod.blob.core.windows.net/datasets/medicaid-provider-spending/2026-02-09/medicaid-provider-spending.parquet"

mkdir -p "$OUT_DIR"
curl -L --fail --progress-bar "$URL" -o "$OUT_FILE"

echo "Downloaded: $OUT_FILE"
ls -lh "$OUT_FILE"
shasum -a 256 "$OUT_FILE"
