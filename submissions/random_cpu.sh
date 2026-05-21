#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/EHR-JEPA/%x_%j.log
#SBATCH --job-name=EHR-JEPA-random

set -e

# Set the base directory for your project
BASE_DIR="/home/ag619/EHR-JEPA"

export WANDB_API_KEY="3256683a0a9a004cf52e04107a3071099a53038e"


# --- Execute from Project Root ---
cd "${BASE_DIR}"
echo "Activating virtual environment..."
source .venv/bin/activate


export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

echo "Job ID:   ${SLURM_JOB_ID}"
echo "Log file: logs/EHR-JEPA/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
echo "Starting random plot"


python scripts/plot_inter_event_hours_histogram.py \
  --data_dir /home/ag619/clean_meds \
  --split train \
  --max-files 150 \
  --out logs/EHR-JEPA/inter_event_hours_histogram.png \
  --unit minutes


echo "random finished."
deactivate

