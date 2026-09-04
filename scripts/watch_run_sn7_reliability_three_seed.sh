#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
RUN_ROOT="${RUN_ROOT:-/home/wh/ActiveMap/runs/sn7_active_catalog}"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/ActiveMap/envs/activemap-agent/bin/python}"
GPU="${SN7_RELIABILITY_GPU:-6}"
PRIMARY_SEED="${ACTIVE_CATALOG_PRIMARY_SEED:-20260717}"
REPLICATE_SEEDS="${REPLICATE_SEEDS:-20260718,20260719}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MANIFEST="${RUN_ROOT}/qwen3vl4b_promoted_three_seed/promotion_manifest.json"
ASSET_ROOT="${SN7_ASSET_ROOT:-/home/wh/ActiveMap/datasets/sn7}"
ASSET_ROOT_MAP="${SN7_ASSET_ROOT_MAP:-/mnt/mydisk/wh/ActiveMap/datasets/sn7=${ASSET_ROOT}}"
CONTROL_ROOT="${RUN_ROOT}/reliability_gate_three_seed"

cd "${PROJECT_ROOT}"
# shellcheck source=/dev/null
source scripts/server_hdpi_env.sh
# shellcheck source=/dev/null
source scripts/assert_allowed_gpu.sh
activemap_assert_allowed_gpu "${GPU}"

if [[ "${1:-}" == "--daemon" ]]; then
  mkdir -p "${CONTROL_ROOT}"
  log="${CONTROL_ROOT}/watcher.log"
  pid_file="${CONTROL_ROOT}/watcher.pid"
  if [[ -s "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
    echo "SN7 reliability watcher is already running: PID $(cat "${pid_file}")"
    exit 0
  fi
  nohup bash "$0" >"${log}" 2>&1 </dev/null &
  pid=$!
  echo "${pid}" >"${pid_file}"
  sleep 2
  kill -0 "${pid}" 2>/dev/null || {
    echo "SN7 reliability watcher failed to start; inspect ${log}" >&2
    exit 1
  }
  echo "SN7 reliability watcher started: PID ${pid}, log ${log}"
  exit 0
fi

status="${CONTROL_ROOT}/watcher.exit_code"
mkdir -p "${CONTROL_ROOT}"
rm -f "${status}"
trap 'code=$?; printf "%s\n" "$code" > "${status}"' EXIT

while [[ ! -s "${MANIFEST}" ]]; do
  echo "[$(date --iso-8601=seconds)] waiting for promoted SN7 selector manifest"
  sleep "${POLL_SECONDS}"
done

decision="$(${PYTHON} -c '
import json, sys
payload = json.load(open(sys.argv[1]))
assert payload.get("test_assets_read") is False
decision = payload.get("sampling_decision")
assert decision in {"weighted", "unweighted"}
print(decision)
' "${MANIFEST}")"
if [[ "${decision}" == "weighted" ]]; then
  primary_family="qwen3vl4b_seed1_eval500"
else
  primary_family="qwen3vl4b_unweighted_seed1"
fi
replicate_family="qwen3vl4b_promoted_${decision}_replicates"

IFS=',' read -r -a replicate_values <<< "${REPLICATE_SEEDS}"
(( ${#replicate_values[@]} == 2 )) || {
  echo "exactly two SN7 replicate seeds are required" >&2
  exit 2
}
seed_values=("${PRIMARY_SEED}" "${replicate_values[@]}")

wait_for_gpu() {
  while [[ -n "$(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
    echo "[$(date --iso-8601=seconds)] waiting for GPU ${GPU}"
    sleep "${POLL_SECONDS}"
  done
}

for seed in "${seed_values[@]}"; do
  family="${replicate_family}"
  [[ "${seed}" != "${PRIMARY_SEED}" ]] || family="${primary_family}"
  adapter="${RUN_ROOT}/${family}/seed${seed}/final"
  [[ -s "${adapter}/adapter_config.json" ]] || {
    echo "missing promoted adapter for seed ${seed}: ${adapter}" >&2
    exit 3
  }
  wait_for_gpu
  echo "[$(date --iso-8601=seconds)] running SN7 reliability seed ${seed}"
  SEED="${seed}" CLOSED_LOOP_ADAPTER="${adapter}" GPU_SECOND="${GPU}" \
    RUN_ROOT="${RUN_ROOT}" WRITEBACK_ASSET_ROOT_MAP="${ASSET_ROOT_MAP}" \
    TOOL_ASSET_ROOT_MAP="${ASSET_ROOT_MAP}" \
    bash scripts/run_sn7_active_catalog_tool_seed.sh
done

all_seeds="${PRIMARY_SEED},${REPLICATE_SEEDS}"
run_aggregate_if_missing() {
  local artifact="$1"
  local stage="$2"
  if [[ -s "${artifact}" ]]; then
    echo "skip=${stage} existing=${artifact}"
  else
    SEEDS="${all_seeds}" RUN_ROOT="${RUN_ROOT}" \
      bash scripts/run_sn7_active_catalog_qwen.sh "${stage}"
  fi
}

run_aggregate_if_missing "${CONTROL_ROOT}/component_ablation.json" compare_tool_belief_reliability_gate_three_seed
run_aggregate_if_missing "${RUN_ROOT}/tool_branch_three_seed/paired_tool_branches.json" aggregate_tool_branches_three_seed
run_aggregate_if_missing "${RUN_ROOT}/tool_branch_three_seed/promotion.json" assess_tool_branches_three_seed
run_aggregate_if_missing "${RUN_ROOT}/tool_writeback_three_seed/paired_no_tool_aoi.json" aggregate_tool_writebacks_three_seed
run_aggregate_if_missing "${RUN_ROOT}/tool_writeback_three_seed/promotion.json" assess_tool_writebacks_three_seed
run_aggregate_if_missing "${CONTROL_ROOT}/promotion.json" aggregate_reliability_gate_three_seed

echo "[$(date --iso-8601=seconds)] SN7 three-seed reliability pipeline complete"
