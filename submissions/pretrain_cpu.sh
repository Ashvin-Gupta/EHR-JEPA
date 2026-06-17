#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/Pretrain/%x_%j.log
#SBATCH --job-name=EHR-JEPA-pretrain

set -e

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

export WANDB_API_KEY="3256683a0a9a004cf52e04107a3071099a53038e"

cd "${BASE_DIR}"
echo "Activating virtual environment..."
source .venv/bin/activate

export PYTHONUNBUFFERED=1

echo "Job ID:   ${SLURM_JOB_ID}"
echo "Log file: logs/EHR-JEPA/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
echo "Starting pretrain"
python main.py --config configs/ehr_config.yaml

echo "Pretrain finished."

deactivate
