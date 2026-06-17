#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/Probe/%x_%j.log
#SBATCH --job-name=EHR-JEPA-probe

set -e

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

export WANDB_API_KEY="3256683a0a9a004cf52e04107a3071099a53038e"
export PYTHONUNBUFFERED=1

cd "${BASE_DIR}"
echo "Activating virtual environment..."
source .venv/bin/activate

echo "Job ID:   ${SLURM_JOB_ID}"
echo "Log file: logs/Probe/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"

# Task is read from data.labels_task in the config.
# Override with --task <name> to run a different task without editing the config.
# e.g.  --task hospital_mortality
python evaluation/run_linear_probe.py \
    --checkpoint "${DATA_DIR}/checkpoints/best.pt" \
    --config     configs/ehr_config.yaml

echo "Linear probe finished."

deactivate
