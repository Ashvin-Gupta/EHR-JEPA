#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/Supervised/%x_%j.log
#SBATCH --job-name=sup-bert


# Usage:
#   sbatch submissions/supervised_bert_gpu.sh [TASK]
#   sbatch --job-name=sup-bert-icu submissions/supervised_bert_gpu.sh icu_mortality
#
# TASK (optional) overrides data.labels_task in the config, e.g.:
#   icu_mortality | hospital_mortality | hospital_readmission
# If omitted, the value from configs/bert_config.yaml is used.


set -e

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

export WANDB_API_KEY="3256683a0a9a004cf52e04107a3071099a53038e"


# --- Execute from Project Root ---
cd "${BASE_DIR}"
echo "Activating virtual environment..."
source .venv/bin/activate

# Disable Python output buffering so the log file updates in real-time
export PYTHONUNBUFFERED=1

TASK="${1:-}"

echo "Job ID:   ${SLURM_JOB_ID}"
echo "Log file: logs/Supervised/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"

PYTHON_ARGS=(
  --config configs/bert_config.yaml
  --checkpoint "${DATA_DIR}/bert_checkpoints/6_layer_256/best.pt"
)
if [[ -n "${TASK}" ]]; then
  PYTHON_ARGS+=(--task "${TASK}")
  echo "Task: ${TASK} (overrides data.labels_task)"
else
  echo "Task: from config (data.labels_task)"
fi

python main_supervised_downstream.py "${PYTHON_ARGS[@]}"

echo "Supervised finished."

deactivate