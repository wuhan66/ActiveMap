#!/usr/bin/env bash
set -euo pipefail

BASE_PYTHON=${BASE_PYTHON:-/home/wh/yes/envs/cbelief/bin/python}
ENV_ROOT=${SAM_ROAD_SMOKE_ENV:-/mnt/mydisk/wh/ActiveMap/envs/sam-road-smoke}
PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}

if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$ENV_ROOT"
fi
"$ENV_ROOT/bin/python" -m pip install --index-url "$PIP_INDEX_URL" \
  numpy==1.26.4 rasterio==1.4.3 lightning==2.2.5 torchmetrics==1.3.2 \
  wandb==0.16.6 addict==2.4.0
"$ENV_ROOT/bin/python" - <<'PY'
import addict
import lightning
import torch
import torchmetrics
import torchvision
import wandb

print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
print(f"torchvision={torchvision.__version__}")
print(f"lightning={lightning.__version__} torchmetrics={torchmetrics.__version__}")
print(f"wandb={wandb.__version__} addict={addict.__version__}")
PY
