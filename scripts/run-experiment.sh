#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

cd "$PROJECT_ROOT"

: "${PYTHON:=python}"

"$PYTHON" -m src.main \
  --seed=1 \
  --dataset="news_headlines" \
  --pretrained-model-name="prajjwal1/bert-tiny" \
  --classification-head="bert_single_token_attention" \
  --dropout-p=0.1 \
  --train-test-split=0.7 \
  --batch-size=32 \
  --learning-rate=2e-5 \
  --max-grad-norm=1.0 \
  --num-train-epochs=5 \
  --experiment-name="test_experiment"
