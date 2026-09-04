#!/usr/bin/env bash
set -euo pipefail

STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
CONDA="${CONDA:-/home/wh/miniconda3/bin/conda}"
ENV_PREFIX="${STORAGE_ROOT}/envs/activemap-opencd"
OPENCD_ROOT="${STORAGE_ROOT}/external/open_cd"
LOG_ROOT="${STORAGE_ROOT}/logs/opencd_setup_20260726"
EXPECTED_COMMIT="790a1972b538baaaffd4c3022a4d9afaa3f861e5"
EXPECTED_LICENSE_SHA256="652623ad25486a679b4e6da22a0c221eddb4c75d09f2733df8b8853969d55020"

mkdir -p "${LOG_ROOT}"
test -x "${CONDA}"
test "$(git -C "${OPENCD_ROOT}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
echo "${EXPECTED_LICENSE_SHA256}  ${OPENCD_ROOT}/LICENSE" | sha256sum --check

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  "${CONDA}" create -y -p "${ENV_PREFIX}" python=3.10 pip=24.0
fi
PYTHON="${ENV_PREFIX}/bin/python"
PIP="${ENV_PREFIX}/bin/pip"

"${PIP}" install \
  torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu118
"${PIP}" install openmim==0.3.9
"${ENV_PREFIX}/bin/mim" install \
  mmengine==0.10.7 \
  mmcv==2.0.1 \
  mmpretrain==1.2.0
"${PIP}" install \
  mmsegmentation==1.2.2 \
  mmdet==3.3.0 \
  albumentations==1.3.1 \
  numpy==1.24.4 \
  scipy==1.10.1 \
  scikit-image==0.21.0 \
  opencv-python==4.8.1.78 \
  opencv-python-headless==4.8.1.78 \
  ftfy==6.2.3 \
  regex \
  prettytable==3.9.0 \
  packaging
"${PIP}" install --force-reinstall --no-deps \
  opencv-python==4.8.1.78 \
  opencv-python-headless==4.8.1.78
"${PIP}" install --no-deps -e "${OPENCD_ROOT}"

"${PYTHON}" - <<'PY' > "${LOG_ROOT}/runtime_audit.json"
import json
import mmcv
import mmdet
import mmengine
import mmseg
import opencd
import torch

print(json.dumps({
    "schema_version": "activemap-opencd-runtime-audit-v1",
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "mmcv": mmcv.__version__,
    "mmengine": mmengine.__version__,
    "mmseg": mmseg.__version__,
    "mmdet": mmdet.__version__,
    "opencd_imported": opencd is not None,
}, indent=2))
PY
"${PIP}" freeze > "${LOG_ROOT}/pip_freeze.txt"
"${PYTHON}" -m pip check > "${LOG_ROOT}/pip_check.txt"
cat "${LOG_ROOT}/runtime_audit.json"
