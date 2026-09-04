#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
RUN_ROOT="${RUN_ROOT:-/home/wh/ActiveMap/runs/sn7_active_catalog}"
TOKEN_REPORT="${TOKEN_REPORT:-${RUN_ROOT}/token_audit/full_qwen3vl4b.json}"
POLL_SECONDS="${POLL_SECONDS:-60}"
AUTO_START_SEED1="${AUTO_START_SEED1:-0}"
SEEDS="${SEEDS:-20260717,20260718,20260719}"

cd "${PROJECT_ROOT}"
first_seed="${SEEDS%%,*}"

while [[ ! -s "${TOKEN_REPORT}" ]]; do
  echo "$(date -Is) waiting for token audit: ${TOKEN_REPORT}"
  sleep "${POLL_SECONDS}"
done

bash scripts/run_sn7_active_catalog_qwen.sh verify_token_audit
bash scripts/run_sn7_active_catalog_qwen.sh preflight

smoke_root="${RUN_ROOT}/smoke_full_protocol"
if [[ ! -e "${smoke_root}" ]]; then
  bash scripts/run_sn7_active_catalog_qwen.sh smoke
fi
bash scripts/run_sn7_active_catalog_qwen.sh verify_smoke

if [[ "${AUTO_START_SEED1}" == "1" ]]; then
  seed_root="${RUN_ROOT}/qwen3vl4b_seed1/seed${first_seed}"
  if [[ -e "${seed_root}" ]]; then
    echo "refusing to overwrite existing seed1 output: ${seed_root}" >&2
    exit 7
  fi
  bash scripts/run_sn7_active_catalog_qwen.sh seed1
else
  echo "all pretraining gates passed; seed1 remains intentionally gated"
fi
