#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
RUN="${STORE}/runs/sn7_active_catalog"
OUTPUT="${STORE}/artifacts/paper_evidence/sn7_fullval_controller_matrix_20260801"
POLL_SECONDS="${POLL_SECONDS:-120}"

required=(
  "${RUN}/hybrid_residual_fullval_matrix_20260801/MATRIX_COMPLETE"
  "${RUN}/hybrid_residual_8k_fullval_matrix_20260801/MATRIX_COMPLETE"
  "${RUN}/direct_vlm_fullval_seed20260718/process_result.json"
  "${RUN}/react_style_fullval_seed20260718/process_result.json"
)

while true; do
  ready=1
  for path in "${required[@]}"; do
    [[ -s "${path}" ]] || ready=0
  done
  [[ "${ready}" -eq 1 ]] && break
  sleep "${POLL_SECONDS}"
done

cd "${PROJECT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
[[ ! -e "${OUTPUT}" ]] || exit 0

"${PYTHON}" scripts/audit_sn7_fullval_controller_matrix.py "${OUTPUT}" \
  --run "old_vla:reference=${RUN}/hybrid_residual_fullval_matrix_20260801/old_vla/merged/traces.jsonl" \
  --run "hybrid_8k:20260821=${RUN}/hybrid_residual_8k_fullval_matrix_20260801/seed20260821" \
  --run "hybrid_8k:20260822=${RUN}/hybrid_residual_8k_fullval_matrix_20260801/seed20260822" \
  --run "hybrid_8k:20260823=${RUN}/hybrid_residual_8k_fullval_matrix_20260801/seed20260823" \
  --run "direct_vlm:20260718=${RUN}/direct_vlm_fullval_seed20260718" \
  --run "react:20260718=${RUN}/react_style_fullval_seed20260718" \
  --reference-method old_vla --expected-count 6369 \
  --repetitions 10000 --bootstrap-seed 20260801
