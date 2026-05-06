#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/EmbedCodes/%x_%j.log
#SBATCH --job-name=EHR-JEPA-embed-codes

set -e

# Set the base directory for your project
BASE_DIR="/home/ag619/EHR-JEPA"

export WANDB_API_KEY="3256683a0a9a004cf52e04107a3071099a53038e"


# --- Execute from Project Root ---
cd "${BASE_DIR}"
echo "Activating virtual environment..."
source .venv/bin/activate


export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

echo "Job ID:   ${SLURM_JOB_ID}"
echo "Log file: logs/EHR-JEPA/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
echo "Starting embed codes"

python scripts/encode_text_embeddings.py \
    --vocab_path       /home/ag619/EHR-JEPA-Data/vocab.json \
    --output_path      /home/ag619/EHR-JEPA-Data/code_embeddings.pt \
    --model_name       emilyalsentzer/Bio_ClinicalBERT \
    --batch_size       64 \
    --device           cpu \
    --labitems_file    /home/ag619/MIMIC_data/hosp/d_labitems.csv.gz \
    --diagnoses_file   /home/ag619/MIMIC_data/hosp/d_icd_diagnoses.csv.gz \
    --procedures_file  /home/ag619/MIMIC_data/hosp/d_icd_procedures.csv.gz

echo "Embed codes finished."

deactivate