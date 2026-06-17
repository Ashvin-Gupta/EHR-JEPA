#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --output=/home/ag619/EHR-JEPA/logs/Pretrain/%x_%j.log
#SBATCH --job-name=EHR-JEPA-pretrain

set -e

BASE_DIR="/home/ag619/EHR-JEPA"

export WANDB_API_KEY="3256683a0a9a004cf52e04107a3071099a53038e"

cd "${BASE_DIR}"
echo "Activating virtual environment..."
source .venv/bin/activate

export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

RESUME_FROM="${1:-}"

echo "Job ID:   ${SLURM_JOB_ID}"
echo "Log file: logs/Pretrain/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
echo "GPUs:     ${SLURM_GPUS_ON_NODE:-4}"

if [[ -n "${RESUME_FROM}" ]]; then
    echo "Resume:   ${RESUME_FROM}"
else
    echo "Resume:   from config (training.resume_from / auto_resume_last)"
fi

# torchrun spawns one process per GPU on this node.
# MASTER_ADDR/PORT are set automatically by SLURM when --nodes=1.
PY_ARGS=(main.py --config configs/ehr_config.yaml)
if [[ -n "${RESUME_FROM}" ]]; then
    PY_ARGS+=(--resume-from "${RESUME_FROM}")
fi

torchrun \
    --standalone \
    --nproc_per_node=4 \
    "${PY_ARGS[@]}"

echo "Pretrain finished."

deactivate
