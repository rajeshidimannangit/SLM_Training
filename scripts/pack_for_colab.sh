#!/usr/bin/env bash
# Helper: zip the project for Colab Option C (excludes heavy/local-only paths)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/../SLM-Finetuning-colab.zip}"
cd "$ROOT"
zip -r "$OUT" . \
  -x "./.venv/*" \
  -x "./.git/*" \
  -x "./__pycache__/*" \
  -x "./**/__pycache__/*" \
  -x "./.pytest_cache/*" \
  -x "./artifacts/mlruns/*" \
  -x "./models/finetuned/*/checkpoint-*" \
  -x "./models/base/**/*.safetensors" \
  -x "./models/base/**/*.bin"
echo "Created: $OUT"
