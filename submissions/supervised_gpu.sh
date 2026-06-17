#!/bin/bash
# Usage:
#   sbatch submissions/supervised_gpu.sh [TASK] [CHECKPOINT]
#   sbatch --job-name=sup-jepa-hosp submissions/supervised_gpu.sh hospital_mortality /path/to/your/checkpoint.pt
#
# TASK (optional) overrides data.labels_task in the config, e.g.:
#   icu_mortality | hospital_mortality | hospital_readmission
# If omitted, the value from configs/ehr_config.yaml is used.
#
# CHECKPOINT (optional) overrides the default checkpoint path.
# If omitted, the default last.pt is used.

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/Supervised/%x_%j.log
#SBATCH --job-name=sup-JEPA

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
CHECKPOINT="${2:-${DATA_DIR}/Fri22ndMaycausal_single/checkpoints/last.pt}"

echo "Job ID:   ${SLURM_JOB_ID}"
echo "Log file: logs/Supervised/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"

PYTHON_ARGS=(
  --config configs/ehr_config.yaml
  --checkpoint "${CHECKPOINT}"
)

if [[ -n "${TASK}" ]]; then
  PYTHON_ARGS+=(--task "${TASK}")
  echo "Task: ${TASK} (overrides data.labels_task)"
else
  echo "Task: from config (data.labels_task)"
fi

if [[ -n "${2}" ]]; then
  echo "Checkpoint: ${CHECKPOINT} (overrides default)"
else
  echo "Checkpoint: default (${CHECKPOINT})"
fi

python main_supervised_downstream.py "${PYTHON_ARGS[@]}"

echo "Supervised finished."

deactivate