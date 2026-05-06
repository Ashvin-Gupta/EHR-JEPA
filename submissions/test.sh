#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/EHR-JEPA/test.log
#SBATCH --job-name=EHR-JEPA-test

set -e

# Set the base directory for your project
BASE_DIR="/home/ag619/EHR-JEPA"

export WANDB_API_KEY="3256683a0a9a004cf52e04107a3071099a53038e"


# --- Execute from Project Root ---
cd "${BASE_DIR}"
echo "Activating virtual environment..."
source .venv/bin/activate


export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

echo "Starting code"
python tests/test_integration_real_data.py 
python tests/run_all_tests.py

# PYTHONPATH=. .venv/bin/python -m pytest tests/test_normalizer.py tests/test_event_embedding_mlp.py tests/test_transformer_encoder.py tests/test_span_masking.py tests/test_latent_pooling.py tests/test_predictor.py tests/test_losses.py tests/test_trainer_forward.py -v -s

echo "Test finished."

deactivate