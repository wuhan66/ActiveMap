#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
RUN_BASE="${RUN_BASE:-${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_corruption_dense_v3b_20260802}"
CORRUPTION_SEED="${CORRUPTION_SEED:-20260802}"
GPU_IDS=(5 7)
SEVERITIES=(1 2)

require_free_gpu() {
  local gpu="$1"
  local pids
  pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${pids//[[:space:]]/}" ]] || {
    echo "GPU ${gpu} has active compute processes: ${pids}" >&2
    exit 4
  }
}

if [[ "${1:-}" == worker ]]; then
  severity="${2:?missing severity}"
  gpu="${3:?missing GPU}"
  run_root="${RUN_BASE}/seed${CORRUPTION_SEED}"
  status_file="${RUN_BASE}/status/seed${CORRUPTION_SEED}_severity${severity}.txt"
  printf 'running\n' >"${status_file}"
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
  printf '%s\n' "${code}" >"${status_file}"
  exit "${code}"
fi

mkdir -p "${RUN_BASE}/logs" "${RUN_BASE}/pids" "${RUN_BASE}/status"
for gpu in "${GPU_IDS[@]}"; do
  require_free_gpu "${gpu}"
done
cat >"${RUN_BASE}/protocol_extension_seed${CORRUPTION_SEED}.txt" <<EOF
split=val
test_assets_read=false
corruption_seed=${CORRUPTION_SEED}
severities=1,2
policy_seeds=20260730,20260731,20260801
variants=notool,forced,benefit
gpus=5,7
EOF

for index in "${!SEVERITIES[@]}"; do
  severity="${SEVERITIES[$index]}"
  gpu="${GPU_IDS[$index]}"
  name="seed${CORRUPTION_SEED}_severity${severity}"
  output_root="${RUN_BASE}/seed${CORRUPTION_SEED}/severity${severity}"
  [[ ! -e "${output_root}" ]] || {
    echo "Refusing existing output: ${output_root}" >&2
    exit 5
  }
  nohup bash "$0" worker "${severity}" "${gpu}" \
    >"${RUN_BASE}/logs/${name}.log" 2>&1 < /dev/null &
  echo "$!" >"${RUN_BASE}/pids/${name}.pid"
  echo "${name}: gpu=${gpu} pid=$!"
done
