#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

cd "$PROJECT_ROOT"

: "${PYTHON:=python}"


START_EXPERIMENT=1


# Fixed parameters (same as evaluation)
HEAD="bert_single_token_attention"
DROPOUT_P=0.1
TRAIN_TEST_SPLIT=0.7
MAX_GRAD_NORM=1.0
NUM_TRAIN_EPOCHS=5

# Get experiments from the JSON file with the best seeds from the evaluation
EXPERIMENT_LINES=()
while IFS= read -r experiment; do
  EXPERIMENT_LINES+=("$experiment")
done < <("$PYTHON" -c '
import json
d=json.load(open("results/best_seeds_evaluation.json"))
for _,cfg in d.items():
    print(cfg["dataset"], cfg["pretrained_model_name"], cfg["learning_rate"], cfg["batch_size"], cfg["seed"], sep="\t")
')

total_experiments=$((${#EXPERIMENT_LINES[@]}))
current_experiment=0

echo "Starting demo model training with ${total_experiments} total experiments..."
echo "Starting from experiment ${START_EXPERIMENT}..."

# Demo model training loop
for experiment in "${EXPERIMENT_LINES[@]}"; do
  IFS=$'\t' read -r dataset model lr batch_size seed <<<"$experiment"

  ((++current_experiment))

  if (( current_experiment < START_EXPERIMENT )); then
    continue
  fi

  experiment_name="demo_models_${dataset}"

  echo ""
  echo "Experiment ${current_experiment}/${total_experiments}"
  echo "Model: ${model}"
  echo "Dataset: ${dataset}"
  echo ""

  "$PYTHON" -m src.main \
    --seed="$seed" \
    --dataset="$dataset" \
    --pretrained-model-name="$model" \
    --classification-head="$HEAD" \
    --dropout-p="$DROPOUT_P" \
    --train-test-split="$TRAIN_TEST_SPLIT" \
    --batch-size="$batch_size" \
    --learning-rate="$lr" \
    --max-grad-norm="$MAX_GRAD_NORM" \
    --num-train-epochs="$NUM_TRAIN_EPOCHS" \
    --experiment-name="$experiment_name" \
    --save-model

    sleep 60

done
