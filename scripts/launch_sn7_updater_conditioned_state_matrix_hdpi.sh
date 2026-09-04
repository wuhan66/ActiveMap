#!/usr/bin/env bash
# Launch the four pre-registered f0/f1 state-generation shards on HDPI.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/paper_evidence/sn7_updater_conditioned_matrix_v1}"
GPUS="${GPUS:-1,3,4,5}"
IFS=',' read -r -a gpu_list <<< "${GPUS}"
[[ ${#gpu_list[@]} -eq 4 ]] || { echo 'set exactly four physical GPU IDs' >&2; exit 2; }

for gpu in "${gpu_list[@]}"; do
  [[ "${gpu}" != "0" && "${gpu}" != "2" ]] || { echo "GPU ${gpu} is reserved" >&2; exit 2; }
  active="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${active//[[:space:]]/}" ]] || { echo "GPU ${gpu} is occupied: ${active}" >&2; exit 3; }
done

test ! -e "${RUN_ROOT}" || { echo "immutable run root exists: ${RUN_ROOT}" >&2; exit 3; }
mkdir -p "${RUN_ROOT}"
printf '{"schema_version":"sn7-updater-conditioned-state-matrix-v1","states":["f0","f1"],"splits":["train","val"],"gpus":[%s],"test_assets_read":false}\n' \
  "$(IFS=,; echo "${gpu_list[*]}")" > "${RUN_ROOT}/launch_manifest.json"

jobs=("f0 train" "f0 val" "f1 train" "f1 val")
for index in "${!jobs[@]}"; do
  read -r state split <<< "${jobs[$index]}"
  gpu="${gpu_list[$index]}"
  job_root="${RUN_ROOT}/states/${state}/${split}"
  mkdir -p "${job_root}"
  nohup env STATE="${state}" SPLIT="${split}" GPU="${gpu}" RUN_ROOT="${RUN_ROOT}" \
    bash "${PROJECT_ROOT}/scripts/run_sn7_updater_conditioned_state_job.sh" \
    > "${job_root}/launcher.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "${job_root}/pid"
done

echo "launched ${RUN_ROOT}; monitor states/*/*/{run.log,COMPLETE.json,FAILED.json}"
