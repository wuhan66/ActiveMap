#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
RUN_BASE="${RUN_BASE:-${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_corruption_dense_v3b_20260802}"
CORRUPTION_SEED=20260802

run_realization() {
  local severity="$1"
  local gpu="$2"
  local status="${RUN_BASE}/status/seed${CORRUPTION_SEED}_severity${severity}_realization_retry.txt"
  printf 'running\n' >"${status}"
  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" RUN_BASE="${RUN_BASE}" \
    CORRUPTION_SEED="${CORRUPTION_SEED}" SEVERITY="${severity}" \
    bash "${PROJECT_ROOT}/scripts/watch_sn7_controller_prior_corruption_realization_v2.sh"
  code="$?"
  set -e
  printf '%s\n' "${code}" >"${status}"
  exit "${code}"
}

run_full_condition() {
  local severity="$1"
  local gpu="$2"
  local run_root="${RUN_BASE}/seed${CORRUPTION_SEED}"
  local status="${RUN_BASE}/status/seed${CORRUPTION_SEED}_severity${severity}.txt"
  printf 'running\n' >"${status}"
  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" RUN_ROOT="${run_root}" \
    CORRUPTION_SEED="${CORRUPTION_SEED}" SEVERITY="${severity}" \
    bash "${PROJECT_ROOT}/scripts/run_sn7_controller_prior_corruption_states.sh"
  code="$?"
  if [[ "${code}" -eq 0 ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" RUN_BASE="${RUN_BASE}" \
      CORRUPTION_SEED="${CORRUPTION_SEED}" SEVERITY="${severity}" \
      bash "${PROJECT_ROOT}/scripts/watch_sn7_controller_prior_corruption_realization_v2.sh"
    code="$?"
  fi
  set -e
  printf '%s\n' "${code}" >"${status}"
  exit "${code}"
}

wait_for_gpu() {
  local gpu="$1"
  while true; do
    pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
    [[ -n "${pids//[[:space:]]/}" ]] || return 0
    sleep 60
  done
}

if [[ "${1:-}" == realization ]]; then
  run_realization "${2:?missing severity}" "${3:?missing GPU}"
elif [[ "${1:-}" == full ]]; then
  run_full_condition "${2:?missing severity}" "${3:?missing GPU}"
elif [[ "${1:-}" == queued-realization ]]; then
  wait_for_gpu "${3:?missing GPU}"
  run_realization "${2:?missing severity}" "${3:?missing GPU}"
fi

mkdir -p "${RUN_BASE}/logs" "${RUN_BASE}/pids" "${RUN_BASE}/status"
for gpu in 2 5 7; do
  pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${pids//[[:space:]]/}" ]] || {
    echo "GPU ${gpu} is occupied: ${pids}" >&2
    exit 3
  }
done
for severity in 1 2; do
  [[ -s "${RUN_BASE}/seed${CORRUPTION_SEED}/severity${severity}/COMPLETE.json" ]] || {
    echo "Missing completed states for severity ${severity}" >&2
    exit 4
  }
done
[[ ! -e "${RUN_BASE}/seed${CORRUPTION_SEED}/severity4" ]] || {
  echo "Refusing existing severity4 output" >&2
  exit 5
}

nohup bash "$0" realization 1 2 \
  >"${RUN_BASE}/logs/seed${CORRUPTION_SEED}_severity1_realization_retry.log" 2>&1 < /dev/null &
echo "$!" >"${RUN_BASE}/pids/seed${CORRUPTION_SEED}_severity1_realization_retry.pid"
nohup bash "$0" realization 2 5 \
  >"${RUN_BASE}/logs/seed${CORRUPTION_SEED}_severity2_realization_retry.log" 2>&1 < /dev/null &
echo "$!" >"${RUN_BASE}/pids/seed${CORRUPTION_SEED}_severity2_realization_retry.pid"
nohup bash "$0" full 4 7 \
  >"${RUN_BASE}/logs/seed${CORRUPTION_SEED}_severity4.log" 2>&1 < /dev/null &
echo "$!" >"${RUN_BASE}/pids/seed${CORRUPTION_SEED}_severity4.pid"

echo "severity1 realization: GPU2"
echo "severity2 realization: GPU5"
echo "severity4 full condition: GPU7"
