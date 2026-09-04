#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
source "${PROJECT_ROOT}/scripts/server_hdpi_env.sh"
CONDA="${CONDA:-/home/wh/miniconda3/bin/conda}"
OFFICIAL="${MUNO21_OFFICIAL_ROOT:-${PROJECT_ROOT}/external/muno21-official}"

if [[ ! -x "${ACTIVEMAP_GRAPH_ENV}/bin/python" ]]; then
  "${CONDA}" create -y -p "${ACTIVEMAP_GRAPH_ENV}" -c conda-forge \
    python=3.11 go=1.23 pip
fi
"${ACTIVEMAP_GRAPH_ENV}/bin/python" -m pip install --upgrade pip setuptools wheel
cd "${PROJECT_ROOT}"
"${ACTIVEMAP_GRAPH_ENV}/bin/python" -m pip install -e '.[graph-eval]'
"${ACTIVEMAP_GRAPH_ENV}/bin/python" -c \
  "import numpy, scipy, networkx, skimage; print('graph Python dependencies ready')"
env -C "${OFFICIAL}/go" \
  GOPROXY="${GOPROXY:-https://goproxy.cn,https://proxy.golang.org,direct}" \
  "${ACTIVEMAP_GO_BIN}" mod download
"${ACTIVEMAP_GO_BIN}" version
