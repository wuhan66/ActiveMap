#!/usr/bin/env bash
set -euo pipefail

REPO=${ACTIVEMAP_PROJECT_ROOT:-/home/wh/projects/activemap-v1-joint-debug}
CONDA=${CONDA_EXE:-/home/wh/yes/bin/conda}
ENV_NAME=${SAM_ROAD_ENV_NAME:-activemap-sam-road}
ENV_ROOT=${SAM_ROAD_ENV:-/home/wh/yes/envs/$ENV_NAME}
PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}

if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
  PIP_INDEX_URL="$PIP_INDEX_URL" "$CONDA" env create \
    -f "$REPO/environments/sam-road.yml"
fi

"$ENV_ROOT/bin/python" -m pip install --index-url "$PIP_INDEX_URL" \
  setuptools==79.0.1 lightning==2.2.5 pytorch-lightning==2.2.5 torchmetrics==1.3.2 \
  wandb==0.16.6 addict==2.4.0 'matplotlib>=3.8,<4'

PYTHONPATH="$REPO/src" "$ENV_ROOT/bin/python" - <<'PY'
import addict
import lightning
import numpy
import rasterio
import torch
import torchmetrics
import torchvision
import wandb

assert tuple(int(part) for part in numpy.__version__.split(".")[:1]) < (2,)
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
print(f"torchvision={torchvision.__version__} numpy={numpy.__version__}")
print(f"lightning={lightning.__version__} torchmetrics={torchmetrics.__version__}")
print(f"wandb={wandb.__version__} rasterio={rasterio.__version__} addict={addict.__version__}")
PY
