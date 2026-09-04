#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
if [[ -n "${ACTIVEMAP_SERVER_ENV:-}" ]]; then
  # shellcheck source=/dev/null
  source "${ACTIVEMAP_SERVER_ENV}"
else
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/scripts/server_hdpi_env.sh"
fi
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
GPU="${MUNO21_AGENT_GPU:-4}"
SEED="${MUNO21_AGENT_SEED:-20260821}"
SELECTOR_SEED="${MUNO21_SELECTOR_SEED:-20260811}"
RUN_FAMILY="${MUNO21_AGENT_RUN_FAMILY:-muno21_qwen3_4b_balanced_tool_sft}"
RUN_DIR="${MUNO21_AGENT_RUN_DIR:-${STORAGE_ROOT}/runs/agent/${RUN_FAMILY}_seed${SEED}}"
DATA_ROOT="${MUNO21_AGENT_DATA_ROOT:-${STORAGE_ROOT}/processed/muno21_v2/agent/agent_data_v10_balanced_sparse_tools}"
ROLLOUT_ROOT="${MUNO21_AGENT_ROLLOUT_ROOT:-${STORAGE_ROOT}/artifacts/paper_rollouts/agent_v10_balanced_tool_val}"
STATUS="${RUN_DIR}/evaluation/v10_watcher.exit_code"

if [[ "${1:-}" == "--daemon" ]]; then
  mkdir -p "${RUN_DIR}/evaluation"
  watcher_log="${RUN_DIR}/evaluation/v10_watcher.log"
  watcher_pid="${RUN_DIR}/evaluation/v10_watcher.pid"
  if [[ -s "${watcher_pid}" ]] && kill -0 "$(cat "${watcher_pid}")" 2>/dev/null; then
    echo "MUNO21 v10 watcher is already running: PID $(cat "${watcher_pid}")"
    exit 0
  fi
  nohup bash "$0" >"${watcher_log}" 2>&1 </dev/null &
  pid=$!
  echo "${pid}" >"${watcher_pid}"
  sleep 2
  kill -0 "${pid}" 2>/dev/null || {
    echo "MUNO21 v10 watcher failed to start; inspect ${watcher_log}" >&2
    exit 1
  }
  echo "MUNO21 v10 watcher started: PID ${pid}, log ${watcher_log}"
  exit 0
fi

mkdir -p "$(dirname "${STATUS}")"
rm -f "${STATUS}"
trap 'status=$?; printf "%s\n" "$status" > "${STATUS}"' EXIT

export MUNO21_AGENT_GPU="${GPU}"
export MUNO21_AGENT_SEED="${SEED}"
export MUNO21_AGENT_RUN_DIR="${RUN_DIR}"
export MUNO21_AGENT_DATA_ROOT="${DATA_ROOT}"

if [[ -s "${RUN_DIR}/train.pid" ]]; then
  train_pid="$(cat "${RUN_DIR}/train.pid")"
  while kill -0 "${train_pid}" 2>/dev/null; do
    echo "[$(date --iso-8601=seconds)] waiting for MUNO21 v10 SFT PID ${train_pid}"
    sleep 60
  done
fi

bash "${PROJECT_ROOT}/scripts/evaluate_promote_sparse_tool_sft.sh"

export MUNO21_AGENT_SEEDS="${SEED}"
export MUNO21_SELECTOR_SEEDS="${SELECTOR_SEED}"
export MUNO21_AGENT_RUN_FAMILY="${RUN_FAMILY}"
export MUNO21_TOOL_DATA="${DATA_ROOT}"
export MUNO21_AGENT_ROLLOUT_ROOT="${ROLLOUT_ROOT}"
export MUNO21_MAX_TOOL_CALLS="${MUNO21_MAX_TOOL_CALLS:-2}"
bash "${PROJECT_ROOT}/scripts/evaluate_agent_three_seeds.sh"

static_decision="${RUN_DIR}/evaluation/selection/static_checkpoint_decision.json"
rollout_summary="${ROLLOUT_ROOT}/seed${SEED}/summary.json"
reachability="${ROLLOUT_ROOT}/tool_call_reachability_audit.json"
diagnostic="${ROLLOUT_ROOT}/diagnostic_promotion.json"
"${ACTIVEMAP_AGENT_ENV}/bin/python" "${PROJECT_ROOT}/scripts/audit_tool_call_reachability_gap.py" \
  "${reachability}" --static "${SEED}=${static_decision}" \
  --recurrent "${SEED}=${rollout_summary}" --allow-single-seed-diagnostic
"${ACTIVEMAP_AGENT_ENV}/bin/python" "${PROJECT_ROOT}/scripts/assess_muno21_v10_diagnostic.py" \
  "${static_decision}" "${rollout_summary}" "${reachability}" "${diagnostic}"

diagnostic_passed="$("${ACTIVEMAP_AGENT_ENV}/bin/python" -c '
import json, sys
print("1" if json.load(open(sys.argv[1])).get("diagnostic_passed") is True else "0")
' "${diagnostic}")"
if [[ "${diagnostic_passed}" == "1" ]]; then
  replicate_gpus="${MUNO21_V10_REPLICATE_GPUS:-4 5}"
  MUNO21_REPLICATE_SEEDS="20260822 20260823" \
  MUNO21_REPLICATE_GPUS="${replicate_gpus}" \
  MUNO21_AGENT_RUN_FAMILY="${RUN_FAMILY}" \
  MUNO21_AGENT_START_SCRIPT="start_muno21_agent_v10_balanced_tool_sft.sh" \
  MUNO21_AGENT_DATA_ROOT="${DATA_ROOT}" \
    bash "${PROJECT_ROOT}/scripts/run_sparse_tool_sft_replicates_parallel.sh"

  three_seed_root="${STORAGE_ROOT}/artifacts/paper_rollouts/agent_v10_balanced_tool_three_seed_val"
  MUNO21_AGENT_SEEDS="20260821 20260822 20260823" \
  MUNO21_SELECTOR_SEEDS="20260811 20260812 20260813" \
  MUNO21_AGENT_RUN_FAMILY="${RUN_FAMILY}" \
  MUNO21_TOOL_DATA="${DATA_ROOT}" \
  MUNO21_AGENT_ROLLOUT_ROOT="${three_seed_root}" \
  MUNO21_AGENT_GPU="${GPU}" MUNO21_MAX_TOOL_CALLS=2 \
    bash "${PROJECT_ROOT}/scripts/evaluate_agent_three_seeds.sh"
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${ACTIVEMAP_AGENT_ENV}/bin/python" \
    "${PROJECT_ROOT}/scripts/aggregate_agent_three_seeds.py" \
    "${three_seed_root}" "${three_seed_root}/three_seed_bootstrap.json"
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${ACTIVEMAP_AGENT_ENV}/bin/python" \
    "${PROJECT_ROOT}/scripts/assess_agent_three_seed_promotion.py" \
    "${three_seed_root}/three_seed_bootstrap.json" \
    "${three_seed_root}/three_seed_promotion.json"
else
  echo "[$(date --iso-8601=seconds)] MUNO21 v10 rejected; replicate training not started"
fi

echo "[$(date --iso-8601=seconds)] MUNO21 v10 diagnostic evaluation complete"
