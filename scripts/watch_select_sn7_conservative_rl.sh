#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"
POLL_SECONDS="${POLL_SECONDS:-60}"
TABLE="${RUN}/rl_selection_diagnostic_n512_seed20260718/controller_table.json"
OUTPUT="${RUN}/conservative_rl_selection_n512_seed20260718"

until [[ -s "${TABLE}" ]]; do
  sleep "${POLL_SECONDS}"
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
"${PYTHON}" scripts/select_sn7_conservative_rl_candidate.py \
  "${TABLE}" "${OUTPUT}" \
  --candidate rl_lr2p5_kl005_bal50 \
  --candidate rl_lr2p5_kl010_bal50 \
  --candidate rl_lr5_kl010_bal50 \
  --candidate rl_lr5_kl005_bal25 \
  --candidate rl_lr2p5_kl010_bal25 \
  --min-utility-ci-low 0.0 \
  --min-quality-ci-low 0.0 \
  --max-false-edit-ci-high 0.0 \
  --min-accuracy-observed 0.0
