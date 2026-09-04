#!/usr/bin/env bash
set -euo pipefail

# Resume only the interrupted visual-backend runs after every NTS GPU is idle.
# This queue never terminates or competes with other users' compute processes.
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1-joint-debug}"
STORE="${STORE:-/mnt/mydisk/wh/ActiveMap}"
PYTHON="${PYTHON:-/home/wh/venvs/activemap/bin/python}"
RUN_ROOT="${STORE}/runs/updater/nts_visual_matrix_20260808"
QUEUE_ROOT="${RUN_ROOT}/resume_queue_v1"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
POLL_SECONDS="${POLL_SECONDS:-120}"

labels=(inria_seed20260821 sn7_v4_seed20260821 muno_v7_prior_roi_seed20260833)
gpus=(0 1 2)

mkdir -p "${QUEUE_ROOT}/configs" "${QUEUE_ROOT}/logs" "${QUEUE_ROOT}/status"
exec 9>"${QUEUE_ROOT}/.queue.lock"
flock -n 9 || exit 0
[[ ! -e "${QUEUE_ROOT}/COMPLETE.json" ]] || exit 0

IFS=',' read -r -a all_gpus <<<"${GPU_IDS}"
[[ "${#all_gpus[@]}" -eq 4 ]] || { echo "GPU_IDS must contain 0,1,2,3" >&2; exit 2; }

for label in "${labels[@]}"; do
  [[ -s "${RUN_ROOT}/${label}/last.pt" ]] || { echo "missing resumable checkpoint: ${label}" >&2; exit 3; }
  [[ -s "${RUN_ROOT}/configs/${label}.yaml" ]] || { echo "missing source config: ${label}" >&2; exit 3; }
done

printf '%s waiting for all NTS GPUs to become idle: %s\n' "$(date -Is)" "${GPU_IDS}" | tee -a "${QUEUE_ROOT}/queue.log"
while true; do
  busy=false
  for gpu in "${all_gpus[@]}"; do
    pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
    [[ -z "${pids//[[:space:]]/}" ]] || busy=true
  done
  "${busy}" || break
  printf '%s external compute is active; retrying in %ss\n' "$(date -Is)" "${POLL_SECONDS}" >>"${QUEUE_ROOT}/queue.log"
  sleep "${POLL_SECONDS}"
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

run_one() {
  local label="$1" gpu="$2"
  local source="${RUN_ROOT}/configs/${label}.yaml"
  local config="${QUEUE_ROOT}/configs/${label}_resume.yaml"
  local run="${RUN_ROOT}/${label}"
  local log="${QUEUE_ROOT}/logs/${label}.log"

  sed 's|resume: false|resume: true|' "${source}" >"${config}"
  grep -q 'resume: true' "${config}" || { echo "resume flag not set: ${label}" >&2; return 4; }
  {
    echo "$(date -Is) RESUME ${label} physical_gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m activemap.cli train-updater "${config}"
    test -s "${run}/state.json"
    date -Is >"${QUEUE_ROOT}/status/${label}.done"
    echo "$(date -Is) DONE ${label}"
  } >"${log}" 2>&1
}

pids=()
for index in 0 1 2; do
  run_one "${labels[$index]}" "${gpus[$index]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done

if ((status == 0)); then
  printf '{"status":"complete","schema_version":"nts-visual-updater-resume-v1","split":"train-validation-only","test_assets_read":false}\n' >"${QUEUE_ROOT}/COMPLETE.json"
else
  printf '{"status":"failed","schema_version":"nts-visual-updater-resume-v1","split":"train-validation-only","test_assets_read":false}\n' >"${QUEUE_ROOT}/FAILED.json"
  exit 1
fi
