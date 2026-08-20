#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-activemap}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
elif [[ -x "$HOME/yes/bin/conda" ]]; then
  CONDA_BIN="$HOME/yes/bin/conda"
else
  echo "Conda was not found. Activate an existing Python 3.10/3.11 environment manually."
  exit 1
fi

eval "$("$CONDA_BIN" shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi
conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$ROOT_DIR/requirements.txt"
python -m pip install -r "$ROOT_DIR/requirements-dev.txt"

if ! python -c 'import torch' >/dev/null 2>&1; then
  if [[ -z "${TORCH_INDEX_URL:-}" ]]; then
    echo "PyTorch is absent. Set TORCH_INDEX_URL to the wheel index matching server CUDA, then rerun."
    echo "Example only: TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124"
    exit 2
  fi
  python -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
fi

python -m pip install -e "$ROOT_DIR" --no-deps
python -m activemap --help >/dev/null
python - <<'PY'
import torch
print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu_count": torch.cuda.device_count()})
PY
