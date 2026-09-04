#!/usr/bin/env bash
set -euo pipefail

CONDA="${CONDA:-/home/wh/miniconda3/bin/conda}"
ENV_PREFIX="${ENV_PREFIX:-/home/wh/ActiveMap/envs/activemap-change-mamba}"
SOURCE="${SOURCE:-/home/wh/ActiveMap/external/change_mamba}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-9ce9cec13f9ea14bc0ad91f071577ec9b3a97983}"
MAX_JOBS="${MAX_JOBS:-4}"

test -x "${CONDA}"
test -d "${SOURCE}/.git"
test -x "${CUDA_HOME}/bin/nvcc"

actual_commit="$(git -C "${SOURCE}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
  echo "ChangeMamba commit mismatch: ${actual_commit}" >&2
  exit 1
fi

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  "${CONDA}" create -y -p "${ENV_PREFIX}" python=3.10 pip
fi

PYTHON="${ENV_PREFIX}/bin/python"
"${PYTHON}" -m pip install --upgrade pip "setuptools<81" wheel
"${PYTHON}" -m pip install \
  torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
"${PYTHON}" -m pip install \
  numpy==1.26.4 packaging triton==2.1.0 timm==0.4.12 pytest chardet yacs termcolor \
  submitit tensorboardX fvcore seaborn ninja einops tqdm

(
  cd "${SOURCE}/kernels/selective_scan"
  env CUDA_HOME="${CUDA_HOME}" MAX_JOBS="${MAX_JOBS}" \
    "${PYTHON}" -m pip install --no-build-isolation .
)

env PYTHONPATH="${SOURCE}" "${PYTHON}" -c \
  "import torch, selective_scan_cuda_core, selective_scan_cuda_oflex; print(torch.__version__)"
