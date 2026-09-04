#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${1:-/home/wh/ActiveMap/envs/argotweak-legacy}"
CONDA_BIN="${CONDA_BIN:-/home/wh/miniconda3/bin/conda}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/configs/external/argotweak_legacy_environment.yaml"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  "${CONDA_BIN}" env create -p "${ENV_PREFIX}" -f "${ENV_FILE}"
fi

PYTHON="${ENV_PREFIX}/bin/python"

# mmdet3d imports torch while generating package metadata, so the legacy
# OpenMMLab stack must be installed in dependency order rather than one batch.
"${PYTHON}" -m pip install \
  torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
"${PYTHON}" -m pip install \
  mmcv-full==1.5.2 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
"${PYTHON}" -m pip install \
  mmdet==2.26.0 mmsegmentation==0.29.1
"${PYTHON}" -m pip install \
  mmdet3d==1.0.0rc6 --no-build-isolation
"${PYTHON}" -m pip install \
  similaritymeasures==0.6.0 numpy==1.22.4 scipy==1.8.0 \
  setuptools==59.5.0 openlanev2==2.1.0 ortools==9.3.10497 \
  huggingface_hub av2==0.2.0 shapely==2.0.0 filelock pyarrow tabulate iso3166 \
  --no-deps

"${PYTHON}" - <<'PY'
import torch
import mmcv
import mmdet
import mmseg
import mmdet3d

print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "mmcv": mmcv.__version__,
    "mmdet": mmdet.__version__,
    "mmseg": mmseg.__version__,
    "mmdet3d": mmdet3d.__version__,
})
PY
