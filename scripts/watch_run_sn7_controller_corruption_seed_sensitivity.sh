#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
BASE="${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_corruption_v1"
PRIMARY="${BASE}/full_seed20260729"
SELECTOR_SEED=20260730
CORRUPTION_SEEDS=(20260730 20260731)
GPU_IDS=(4 5)

for corruption_seed in "${CORRUPTION_SEEDS[@]}"; do
  while [[ ! -f "${BASE}/corruption_seed${corruption_seed}/severity16/COMPLETE.json" ]]; do
    sleep 30
  done
done

run_one() {
  local corruption_seed="$1"
  local variant="$2"
  local gpu="$3"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    RUN_ROOT="${BASE}/corruption_seed${corruption_seed}" \
    CORRUPTION_SEED="${corruption_seed}" \
    SEVERITY=16 SEED="${SELECTOR_SEED}" VARIANT="${variant}" \
    bash "${PROJECT_ROOT}/scripts/run_sn7_controller_prior_corruption_job.sh"
}

jobs=(
  20260730:notool
  20260730:benefit
  20260731:notool
  20260731:benefit
)
index=0
while (( index < ${#jobs[@]} )); do
  pids=()
  labels=()
  for gpu in "${GPU_IDS[@]}"; do
    (( index < ${#jobs[@]} )) || break
    IFS=: read -r corruption_seed variant <<<"${jobs[$index]}"
    run_one "${corruption_seed}" "${variant}" "${gpu}" &
    pids+=("$!")
    labels+=("${corruption_seed}:${variant}:gpu${gpu}")
    ((index += 1))
  done
  status=0
  for job_index in "${!pids[@]}"; do
    if wait "${pids[$job_index]}"; then
      echo "completed ${labels[$job_index]}"
    else
      echo "failed ${labels[$job_index]}" >&2
      status=1
    fi
  done
  (( status == 0 )) || exit "${status}"
done

while [[ ! -f "${PRIMARY}/severity16/seed${SELECTOR_SEED}/notool/COMPLETE.json" ||
         ! -f "${PRIMARY}/severity16/seed${SELECTOR_SEED}/benefit/COMPLETE.json" ]]; do
  sleep 30
done

OUTPUT="${BASE}/severe_seed_sensitivity_v1"
mkdir -p "${OUTPUT}"
PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
  "${PYTHON}" scripts/aggregate_active_catalog_tool_writebacks.py \
    "${OUTPUT}/benefit_vs_notool.json" \
    --baseline "20260729=${PRIMARY}/severity16/seed${SELECTOR_SEED}/notool/writeback/writeback.jsonl" \
    --candidate "20260729=${PRIMARY}/severity16/seed${SELECTOR_SEED}/benefit/writeback/writeback.jsonl" \
    --baseline "20260730=${BASE}/corruption_seed20260730/severity16/seed${SELECTOR_SEED}/notool/writeback/writeback.jsonl" \
    --candidate "20260730=${BASE}/corruption_seed20260730/severity16/seed${SELECTOR_SEED}/benefit/writeback/writeback.jsonl" \
    --baseline "20260731=${BASE}/corruption_seed20260731/severity16/seed${SELECTOR_SEED}/notool/writeback/writeback.jsonl" \
    --candidate "20260731=${BASE}/corruption_seed20260731/severity16/seed${SELECTOR_SEED}/benefit/writeback/writeback.jsonl" \
    --repetitions 5000 --seed 20260729 --split val

printf '{"status":"complete","corruption_seeds":[20260729,20260730,20260731],"selector_seed":20260730,"test_assets_read":false}\n' \
  >"${OUTPUT}/COMPLETE.json"
