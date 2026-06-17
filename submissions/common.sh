# Shared workspace paths — source from submission scripts:
#   source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${BASE_DIR}/.." && pwd)"

DATA_DIR="${EHR_JEPA_DATA:-${WORKSPACE_DIR}/EHR-JEPA-Data}"
MIMIC_DIR="${MIMIC_DATA:-${WORKSPACE_DIR}/MIMIC_data}"
CLEAN_MEDS_DIR="${CLEAN_MEDS:-${WORKSPACE_DIR}/clean_meds}"

export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"
