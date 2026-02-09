#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

cd "$PROJECT_ROOT"

: "${PYTHON:=python}"


START_EXPERIMENT=1


# Fixed parameters (same as grid search)
HEAD="bert_single_token_attention"
DROPOUT_P=0.1
TRAIN_TEST_SPLIT=0.7
MAX_GRAD_NORM=1.0
NUM_TRAIN_EPOCHS=5

# Number of runs
NUM_RUNS=10

# Get experiments from the JSON file with the best hyperparameters from the grid search
EXPERIMENT_LINES=()
while IFS= read -r experiment; do
  EXPERIMENT_LINES+=("$experiment")
done < <("$PYTHON" -c '
import json
d=json.load(open("results/best_hparams_grid_search.json"))
for folder,cfg in d.items():
    if folder.startswith("grid_search"):
        print(cfg["dataset"], cfg["pretrained_model_name"], cfg["learning_rate"], cfg["batch_size"], sep="\t")
')

total_experiments=$((${#EXPERIMENT_LINES[@]} * NUM_RUNS))
current_experiment=0

echo "Starting evaluation with ${total_experiments} total experiments..."
echo "Starting from experiment ${START_EXPERIMENT}..."

# Evaluation loop
for experiment in "${EXPERIMENT_LINES[@]}"; do
  IFS=$'\t' read -r dataset model lr batch_size <<<"$experiment"

  for seed in $(seq 1 "$NUM_RUNS"); do
    ((++current_experiment))

    if (( current_experiment < START_EXPERIMENT )); then
      continue
    fi

    experiment_name="evaluation_${dataset}"

    echo ""
    echo "Experiment ${current_experiment}/${total_experiments}"
    echo "Model: ${model}"
    echo "Dataset: ${dataset}"
    echo "Seed: ${seed}"
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
      --experiment-name="$experiment_name"

    sleep 60

  done
done
