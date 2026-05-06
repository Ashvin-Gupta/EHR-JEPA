#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/Pretrain/%x_%j.log
#SBATCH --job-name=EHR-JEPA-pretrain

set -e

# Set the base directory for your project
BASE_DIR="/home/ag619/EHR-JEPA"

export WANDB_API_KEY="3256683a0a9a004cf52e04107a3071099a53038e"


# --- Execute from Project Root ---
cd "${BASE_DIR}"
echo "Activating virtual environment..."
source .venv/bin/activate


export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"
# Disable Python output buffering so the log file updates in real-time
export PYTHONUNBUFFERED=1

echo "Job ID:   ${SLURM_JOB_ID}"
echo "Log file: logs/Pretrain/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
python main.py --config configs/ehr_config.yaml

echo "Pretrain finished."

deactivate