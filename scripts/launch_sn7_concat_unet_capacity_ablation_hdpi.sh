#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
RUNNER="${PROJECT_ROOT}/scripts/run_updater_training_job.sh"
LOG_ROOT="${STORAGE_ROOT}/logs"

launch_one() {
  local gpu="$1"
  local width="$2"
  local config="${PROJECT_ROOT}/configs/updater/sn7_v4_hierarchical_vector_change_scratch_width${width}_seed20260716_server.yaml"
  local run="${STORAGE_ROOT}/runs/updater/v4_hierarchical_vector_change_scratch_width${width}_seed20260716"
  local archive="${PROJECT_ROOT}/outputs/updater/v4_hierarchical_vector_change_scratch_width${width}_seed20260716"
  local log="${LOG_ROOT}/train_sn7_concat_unet_width${width}_seed20260716.log"
  local pid_file="${STORAGE_ROOT}/run_control/sn7_concat_unet_width${width}_seed20260716.pid"

  if [[ -e "${run}" ]]; then
    echo "Refusing to overwrite existing run: ${run}" >&2
    return 1
  fi
  mkdir -p "${run}" "${archive}" "${LOG_ROOT}" "${STORAGE_ROOT}/run_control"
  printf '%s\n' \
    "CUDA_VISIBLE_DEVICES=${gpu} UPDATER_PHYSICAL_GPU=${gpu} ${RUNNER}" \
    > "${run}/launch_command.txt"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export UPDATER_PHYSICAL_GPU="${gpu}"
    export UPDATER_RUN_DIR="${run}"
    export UPDATER_CONFIG="${config}"
    export ACTIVEMAP_PYTHON="${PYTHON}"
    export UPDATER_ARCHIVE_DIR="${archive}"
    exec bash "${RUNNER}"
  ) > "${log}" 2>&1 &
  local pid="$!"
  printf '%s\n' "${pid}" > "${pid_file}"
  echo "width=${width} gpu=${gpu} pid=${pid} run=${run}"
}

cd "${PROJECT_ROOT}"
launch_one "${WIDTH48_GPU:-2}" 48
launch_one "${WIDTH64_GPU:-3}" 64
