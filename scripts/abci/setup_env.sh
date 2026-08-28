#!/bin/bash
set -euo pipefail

PROJECT_DIR="${1:-$PWD}"
VENV_DIR="${PROJECT_DIR}/.venv-abci"

source /etc/profile.d/modules.sh
module purge
module load python/3.12/3.12.9 cuda/13.0/13.0.1

python -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${PROJECT_DIR}/requirements.txt"

python -m pip check
python - <<'PY'
import torch
import torchaudio

print(f"torch={torch.__version__}")
print(f"torchaudio={torchaudio.__version__}")
print(f"cuda_build={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
PY
