#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/ag619/EHR-JEPA/logs/Baseline/%x_%j.log
#SBATCH --job-name=EHR-AR-pretrain

set -e

BASE_DIR="/home/ag619/EHR-JEPA"

export WANDB_API_KEY="3256683a0a9a004cf52e04107a3071099a53038e"

cd "${BASE_DIR}"
echo "Activating virtual environment..."
source .venv/bin/activate

export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

echo "Job ID:   ${SLURM_JOB_ID}"
echo "Log file: logs/Baseline/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"

python main_ar.py --config configs/ar_config.yaml

echo "AR pretrain finished."

deactivate
