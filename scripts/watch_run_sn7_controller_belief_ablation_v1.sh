#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
REALIZATION_ROOT="${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_corruption_realization_v2"
SOURCE_ROOT="${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_corruption_v1/full_seed20260729/severity4"
RUN_ROOT="${STORAGE_ROOT}/runs/sn7_active_catalog/controller_belief_ablation_v1/severity4"
SEEDS=(20260730 20260731 20260801)
mkdir -p "${RUN_ROOT}"
FAILURE_MARKER="${RUN_ROOT}/MATRIX_FAILED.json"
rm -f "${FAILURE_MARKER}"

record_failure() {
  local rc=$?
  trap - EXIT
  if ((rc != 0)); then
    local temporary="${FAILURE_MARKER}.tmp.$$"
    printf \
      '{"status":"failed","stage":"belief_ablation_matrix","exit_code":%d,"test_assets_read":false}\n' \
      "${rc}" >"${temporary}"
    mv -f "${temporary}" "${FAILURE_MARKER}"
  fi
  exit "${rc}"
}
trap record_failure EXIT

for corruption_seed in 20260730 20260731; do
  for severity in 4 8; do
    marker="${REALIZATION_ROOT}/seed${corruption_seed}/severity${severity}/REALIZATION_MATRIX_COMPLETE.json"
    failure="${REALIZATION_ROOT}/seed${corruption_seed}/severity${severity}/MATRIX_FAILED.json"
    while [[ ! -s "${marker}" ]]; do
      [[ ! -s "${failure}" ]] || { cat "${failure}" >&2; exit 5; }
      sleep 60
    done
  done
done

wait_for_gpu() {
  local gpu="$1"
  while true; do
    local used
    used="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits)"
    ((used < 1024)) && return
    sleep 30
  done
}

cd "${PROJECT_ROOT}"
for index in 0 1 2; do
  seed="${SEEDS[$index]}"
  gpu="$((index + 1))"
  marker="${RUN_ROOT}/ungated/seed${seed}/COMPLETE.json"
  [[ -s "${marker}" ]] && continue
  wait_for_gpu "${gpu}"
  session="sn7_belief_ungated_s${seed}"
  log="${STORAGE_ROOT}/logs/${session}.log"
  tmux has-session -t "${session}" 2>/dev/null ||
    tmux new-session -d -s "${session}" \
      "cd ${PROJECT_ROOT} && CUDA_VISIBLE_DEVICES=${gpu} SEED=${seed} ABLATION=ungated bash scripts/run_sn7_controller_belief_ablation_job.sh > ${log} 2>&1"
done

wait_for_gpu 4
tmux has-session -t sn7_belief_frozen_prior 2>/dev/null ||
  tmux new-session -d -s sn7_belief_frozen_prior \
    "cd ${PROJECT_ROOT} && for seed in ${SEEDS[*]}; do CUDA_VISIBLE_DEVICES=4 SEED=\${seed} ABLATION=frozen_prior bash scripts/run_sn7_controller_belief_ablation_job.sh || exit; done > ${STORAGE_ROOT}/logs/sn7_belief_frozen_prior.log 2>&1"

for ablation in ungated frozen_prior; do
  for seed in "${SEEDS[@]}"; do
    marker="${RUN_ROOT}/${ablation}/seed${seed}/COMPLETE.json"
    failure="${RUN_ROOT}/${ablation}/seed${seed}/FAILED.json"
    while [[ ! -s "${marker}" ]]; do
      [[ ! -s "${failure}" ]] || { cat "${failure}" >&2; exit 6; }
      sleep 60
    done
  done
done

mkdir -p "${RUN_ROOT}/comparisons"
gated_writebacks=()
ungated_writebacks=()
frozen_writebacks=()
for seed in "${SEEDS[@]}"; do
  gated="${SOURCE_ROOT}/seed${seed}/benefit"
  ungated="${RUN_ROOT}/ungated/seed${seed}"
  frozen="${RUN_ROOT}/frozen_prior/seed${seed}"
  reliability_report="${RUN_ROOT}/comparisons/gated_vs_ungated_seed${seed}.json"
  if [[ ! -s "${reliability_report}" ]]; then
    PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
      "${PYTHON}" scripts/compare_closed_loop_belief_ablation_baselines.py \
        "${reliability_report}" \
        --ablation ungated \
        --reference-summary "${ungated}/closed_loop/summary.json" \
        --gated-summary "${gated}/closed_loop/summary.json" \
        --reference-traces "${ungated}/closed_loop/edit_utility.jsonl" \
        --gated-traces "${gated}/closed_loop/edit_utility.jsonl" \
        --repetitions 5000 --seed "${seed}"
  fi
  frozen_report="${RUN_ROOT}/comparisons/gated_vs_frozen_prior_seed${seed}.json"
  if [[ ! -s "${frozen_report}" ]]; then
    PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
      "${PYTHON}" scripts/compare_closed_loop_belief_ablation_baselines.py \
        "${frozen_report}" \
        --ablation frozen_prior \
        --reference-summary "${frozen}/closed_loop/summary.json" \
        --gated-summary "${gated}/closed_loop/summary.json" \
        --reference-traces "${frozen}/closed_loop/edit_utility.jsonl" \
        --gated-traces "${gated}/closed_loop/edit_utility.jsonl" \
        --repetitions 5000 --seed "${seed}"
  fi
  gated_writebacks+=(--candidate "${seed}=${gated}/writeback/writeback.jsonl")
  ungated_writebacks+=(--baseline "${seed}=${ungated}/writeback/writeback.jsonl")
  frozen_writebacks+=(--baseline "${seed}=${frozen}/writeback/writeback.jsonl")
done

if [[ ! -s "${RUN_ROOT}/comparisons/gated_vs_ungated_writeback.json" ]]; then
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" scripts/aggregate_active_catalog_tool_writebacks.py \
      "${RUN_ROOT}/comparisons/gated_vs_ungated_writeback.json" \
      "${ungated_writebacks[@]}" "${gated_writebacks[@]}" \
      --repetitions 5000 --seed 20260730 --split val
fi
if [[ ! -s "${RUN_ROOT}/comparisons/gated_vs_frozen_prior_writeback.json" ]]; then
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" scripts/aggregate_active_catalog_tool_writebacks.py \
      "${RUN_ROOT}/comparisons/gated_vs_frozen_prior_writeback.json" \
      "${frozen_writebacks[@]}" "${gated_writebacks[@]}" \
      --repetitions 5000 --seed 20260730 --split val
fi

(
  cd "${RUN_ROOT}"
  find ungated frozen_prior comparisons -type f \
    \( -name '*.json' -o -name '*.jsonl' \) -print0 |
    sort -z | xargs -0 sha256sum
) >"${RUN_ROOT}/MANIFEST.sha256"
temporary="${RUN_ROOT}/SUMMARY_COMPLETE.json.tmp.$$"
printf '{"status":"complete","ablations":["ungated","frozen_prior"],"policy_seeds":3,"test_assets_read":false}\n' \
  >"${temporary}"
mv -f "${temporary}" "${RUN_ROOT}/SUMMARY_COMPLETE.json"
trap - EXIT
