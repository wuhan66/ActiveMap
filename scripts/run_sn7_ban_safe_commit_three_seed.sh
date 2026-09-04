#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-opencd/bin/python"
OPENCD_ROOT="${STORAGE_ROOT}/external/open_cd"
CONFIG="${OPENCD_ROOT}/configs/ban/ban_vit-b16-clip_mit-b0_512x512_40k_levircd.py"
MANIFEST="${STORAGE_ROOT}/processed/sn7_v1/updater_v4_cap20/train_val_complete_20260726.jsonl"
RUN_ROOT="${STORAGE_ROOT}/runs/external_baselines/sn7/open_cd_ban"
LOG_ROOT="${STORAGE_ROOT}/logs/opencd_ban_safe_commit_20260727"

mkdir -p "${LOG_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

run_train_audit() {
  local gpu="$1"
  local seed="$2"
  local run="${RUN_ROOT}/full_weight5_seed${seed}_v1"
  local output="${run}/train_audit"
  if [[ -s "${output}/summary.json" ]]; then
    return
  fi
  if [[ -e "${output}" ]]; then
    echo "Refusing partial train audit: ${output}" >&2
    return 1
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/evaluate_sn7_opencd_ban.py \
    "${run}/best.pt" "${MANIFEST}" "${OPENCD_ROOT}" "${CONFIG}" "${output}" \
    --split train --device cuda:0 --batch-size 8 --workers 8 --no-save-masks \
    >"${LOG_ROOT}/seed${seed}_train_audit.log" 2>&1
}

run_train_audit 1 20260725 &
pid1="$!"
run_train_audit 4 20260726 &
pid2="$!"
run_train_audit 6 20260727 &
pid3="$!"

status=0
for pid in "${pid1}" "${pid2}" "${pid3}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

safe_runs=()
for seed in 20260725 20260726 20260727; do
  run="${RUN_ROOT}/full_weight5_seed${seed}_v1"
  output="${RUN_ROOT}/safe_commit_seed${seed}_v1"
  if [[ ! -s "${output}/summary.json" ]]; then
    if [[ -e "${output}" ]]; then
      echo "Refusing partial safe-commit output: ${output}" >&2
      exit 1
    fi
    "${PYTHON}" scripts/calibrate_sn7_changemamba_safe_commit.py \
      "${run}/train_audit/per_sample.jsonl" \
      "${run}/val_audit/per_sample.jsonl" \
      "${output}" --folds 5 --l2 0.01 \
      >"${LOG_ROOT}/seed${seed}_calibrate.log" 2>&1
  fi
  safe_runs+=("${output}")
done

"${PYTHON}" scripts/aggregate_sn7_changemamba_safe_commit.py \
  "${RUN_ROOT}/ban_safe_commit_three_seed_20260727.json" \
  "${safe_runs[@]}" \
  --output-markdown "${RUN_ROOT}/ban_safe_commit_three_seed_20260727.md"

"${PYTHON}" scripts/bootstrap_sn7_changemamba_safe_commit.py \
  "${RUN_ROOT}/ban_safe_commit_three_seed_aoi_bootstrap_20260727.json" \
  "${safe_runs[@]}" \
  --draws 5000 --seed 20260727 \
  --output-markdown \
    "${RUN_ROOT}/ban_safe_commit_three_seed_aoi_bootstrap_20260727.md"
