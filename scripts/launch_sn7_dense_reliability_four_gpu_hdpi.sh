#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
RUN_BASE="${RUN_BASE:-${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_corruption_dense_v3b_20260802}"
GPU_IDS=(1 2 3 4)
CONDITIONS=(20260730:1 20260731:1 20260730:2 20260731:2)

require_free_gpu() {
  local gpu="$1"
  local pids
  pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${pids//[[:space:]]/}" ]] || {
    echo "GPU ${gpu} has active compute processes: ${pids}" >&2
    exit 4
  }
}

run_lane() {
  local corruption_seed="$1"
  local severity="$2"
  local gpu="$3"
  local run_root="${RUN_BASE}/seed${corruption_seed}"
  local status_file="${RUN_BASE}/status/seed${corruption_seed}_severity${severity}.txt"
  printf 'running\n' >"${status_file}"
  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" RUN_ROOT="${run_root}" \
    CORRUPTION_SEED="${corruption_seed}" SEVERITY="${severity}" \
    bash "${PROJECT_ROOT}/scripts/run_sn7_controller_prior_corruption_states.sh"
  local code="$?"
  if [[ "${code}" -eq 0 ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" RUN_BASE="${RUN_BASE}" \
      CORRUPTION_SEED="${corruption_seed}" SEVERITY="${severity}" \
      bash "${PROJECT_ROOT}/scripts/watch_sn7_controller_prior_corruption_realization_v2.sh"
    code="$?"
  fi
  set -e
  printf '%s\n' "${code}" >"${status_file}"
  return "${code}"
}

if [[ "${1:-}" == worker ]]; then
  run_lane "${2:?missing corruption seed}" "${3:?missing severity}" "${4:?missing GPU}"
  exit $?
fi

[[ ! -e "${RUN_BASE}" ]] || { echo "Refusing existing run root: ${RUN_BASE}" >&2; exit 5; }
for gpu in "${GPU_IDS[@]}"; do
  require_free_gpu "${gpu}"
done
mkdir -p "${RUN_BASE}/logs" "${RUN_BASE}/pids" "${RUN_BASE}/status"
cat >"${RUN_BASE}/protocol.txt" <<EOF
split=val
test_assets_read=false
severities=1,2
corruption_seeds=20260730,20260731
policy_seeds=20260730,20260731,20260801
variants=notool,forced,benefit
gpus=1,2,3,4
EOF

for index in "${!CONDITIONS[@]}"; do
  IFS=: read -r corruption_seed severity <<<"${CONDITIONS[$index]}"
  gpu="${GPU_IDS[$index]}"
  name="seed${corruption_seed}_severity${severity}"
  nohup bash "$0" worker "${corruption_seed}" "${severity}" "${gpu}" \
    >"${RUN_BASE}/logs/${name}.log" 2>&1 < /dev/null &
  echo "$!" >"${RUN_BASE}/pids/${name}.pid"
  echo "${name}: gpu=${gpu} pid=$!"
done
