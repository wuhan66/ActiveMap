#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
RUN_ROOT="${RUN_ROOT:-/home/wh/ActiveMap/runs/sn7_active_catalog}"
DATA_ROOT="${DATA_ROOT:-/home/wh/ActiveMap/processed/sn7_v1/agent/sequential_selector_v1}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
SEEDS="${SEEDS:-20260717,20260718,20260719}"
EVAL_GPU="${EVAL_GPU:-1}"
POLL_SECONDS="${POLL_SECONDS:-60}"
SFT_RUN_NAME="${SFT_RUN_NAME:-qwen3vl4b_seed1_eval500}"
RUN_FAMILY="${RUN_FAMILY:-${SFT_RUN_NAME}}"
EVALUATION_STAGE="${EVALUATION_STAGE:-evaluate_seed1}"

case "${EVALUATION_STAGE}" in
  evaluate_seed1|evaluate_unweighted_seed1) ;;
  *) echo "unsupported validation stage: ${EVALUATION_STAGE}" >&2; exit 3 ;;
esac

cd "${PROJECT_ROOT}"
source scripts/server_hdpi_env.sh
first_seed="${SEEDS%%,*}"
seed_root="${RUN_ROOT}/${RUN_FAMILY}/seed${first_seed}"
state="${seed_root}/run_state.json"
result="${seed_root}/process_result.json"
adapter="${seed_root}/final/adapter_config.json"
evaluation="${seed_root}/active_catalog_val"

while true; do
  if [[ -s "${state}" ]]; then
    status="$(${ACTIVEMAP_AGENT_ENV}/bin/python -c \
      'import json,sys; print(json.load(open(sys.argv[1])).get("status", "unknown"))' \
      "${state}")"
    case "${status}" in
      completed) break ;;
      failed)
        echo "seed1 training failed: ${state}" >&2
        exit 7
        ;;
    esac
  fi
  echo "$(date -Is) waiting for seed1 completion"
  sleep "${POLL_SECONDS}"
done

${ACTIVEMAP_AGENT_ENV}/bin/python - "${state}" "${result}" "${adapter}" <<'PY'
import json
import pathlib
import sys

state_path, result_path, adapter_path = map(pathlib.Path, sys.argv[1:])
state = json.loads(state_path.read_text())
result = json.loads(result_path.read_text())
if state.get("status") != "completed" or state.get("returncode") != 0:
    raise SystemExit(f"invalid completed state: {state}")
if result.get("returncode") != 0:
    raise SystemExit(f"training process failed: {result}")
if not adapter_path.is_file():
    raise SystemExit(f"missing final adapter: {adapter_path}")
PY

[[ ! -e "${evaluation}" ]] || {
  echo "refusing to overwrite seed1 evaluation: ${evaluation}" >&2
  exit 8
}

MODEL="${MODEL}" DATA_ROOT="${DATA_ROOT}" RUN_ROOT="${RUN_ROOT}" \
GPU_SECOND="${EVAL_GPU}" SEEDS="${SEEDS}" SFT_RUN_NAME="${SFT_RUN_NAME}" \
  bash scripts/run_sn7_active_catalog_qwen.sh "${EVALUATION_STAGE}"
