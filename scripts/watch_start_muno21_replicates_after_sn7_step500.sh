#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
DIAGNOSTIC_REPORT="${DIAGNOSTIC_REPORT:-${STORAGE_ROOT}/runs/sn7_active_catalog/seed1_sampling_ablation_step500.json}"
POLL_SECONDS="${POLL_SECONDS:-60}"
PYTHON="${ACTIVEMAP_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"

cd "${PROJECT_ROOT}"
# shellcheck source=/dev/null
source scripts/server_hdpi_env.sh

while [[ ! -s "${DIAGNOSTIC_REPORT}" ]]; do
  echo "$(date -Is) waiting for SN7 step-500 diagnostic: ${DIAGNOSTIC_REPORT}"
  sleep "${POLL_SECONDS}"
done

"${PYTHON}" - "${DIAGNOSTIC_REPORT}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("diagnostic_only") is not True:
    raise SystemExit("step-500 report must be diagnostic_only=true")
if report.get("promotion_eligible") is not False:
    raise SystemExit("step-500 report must be promotion_eligible=false")
PY

export MUNO21_REPLICATE_SEEDS="${MUNO21_REPLICATE_SEEDS:-20260822 20260823}"
export MUNO21_REPLICATE_GPUS="${MUNO21_REPLICATE_GPUS:-4 5}"
echo "$(date -Is) SN7 diagnostic complete; starting MUNO21 seeds ${MUNO21_REPLICATE_SEEDS} on GPUs ${MUNO21_REPLICATE_GPUS}"
exec bash scripts/run_sparse_tool_sft_replicates_parallel.sh
