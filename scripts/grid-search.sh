#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

cd "$PROJECT_ROOT"

: "${PYTHON:=python}"


START_EXPERIMENT=1


# Models
MODELS=(
  "prajjwal1/bert-tiny"
  "prajjwal1/bert-mini"
  "prajjwal1/bert-small"
  "prajjwal1/bert-medium"
  "google-bert/bert-base-uncased"
)

# Datasets
DATASETS=(
  "news_headlines"
  "isarcasm"
  "figlang"
)

# Grid search parameters
LEARNING_RATES=(1e-5 2e-5 3e-5 5e-5 1e-4)
BATCH_SIZES=(8 16 32)

# Fixed parameters
SEED=1
HEAD="bert_single_token_attention"
DROPOUT_P=0.1
TRAIN_TEST_SPLIT=0.7
MAX_GRAD_NORM=1.0
NUM_TRAIN_EPOCHS=5


total_experiments=$((${#MODELS[@]} * ${#DATASETS[@]} * ${#LEARNING_RATES[@]} * ${#BATCH_SIZES[@]}))
current_experiment=0

echo "Starting grid search with ${total_experiments} total experiments..."
echo "Starting from experiment ${START_EXPERIMENT}..."

# Grid search loop
for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
      for batch_size in "${BATCH_SIZES[@]}"; do
        ((++current_experiment))

        if (( current_experiment < START_EXPERIMENT )); then
          continue
        fi

        # Experiment name
        model_short=$(echo "$model" | sed 's/.*\///' | sed 's/-/_/g')
        experiment_name="grid_search_${model_short}_${dataset}"

        echo ""
        echo "Experiment ${current_experiment}/${total_experiments}"
        echo "Model: ${model}"
        echo "Dataset: ${dataset}"
        echo "Learning Rate: ${lr}"
        echo "Batch Size: ${batch_size}"
        echo ""

        # Run experiment
        "$PYTHON" -m src.main \
          --seed="$SEED" \
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
  done
done
