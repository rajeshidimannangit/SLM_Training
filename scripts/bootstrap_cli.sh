#!/usr/bin/env bash
# Quick local smoke: install editable package and show CLI help
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python -m pip install -e . -q
slm --help
slm metrics --help
