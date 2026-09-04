#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
SAMPLES="${STORAGE_ROOT}/processed/sn7_v1/updater_v4_cap20/updater_samples.jsonl"
OUTPUT="${STORAGE_ROOT}/runs/updater/concat_unet_capacity_ablation_20260729"
LOG="${STORAGE_ROOT}/logs/concat_unet_capacity_ablation_20260729.log"
WIDTH32_HISTORY="${WIDTH32_HISTORY:-${STORAGE_ROOT}/runs/updater/v4_hierarchical_vector_change_scratch_seed20260716/history.jsonl}"

if [[ ! -s "${WIDTH32_HISTORY}" ]]; then
  WIDTH32_HISTORY="${OUTPUT}/reference_width32/history.jsonl"
fi

declare -a RUNS=(
  "2|48|6586000|v4_hierarchical_vector_change_scratch_width48_seed20260716"
  "3|64|11701776|v4_hierarchical_vector_change_scratch_width64_seed20260716"
)

mkdir -p "${OUTPUT}" "$(dirname "${LOG}")"
exec >>"${LOG}" 2>&1
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
echo "[$(date --iso-8601=seconds)] waiting for concat U-Net capacity runs"

for spec in "${RUNS[@]}"; do
  IFS='|' read -r _ width _ run_name <<<"${spec}"
  state="${STORAGE_ROOT}/runs/updater/${run_name}/state.json"
  while true; do
    if grep -q '"status": "completed"' "${state}" 2>/dev/null; then
      break
    fi
    if grep -Eq '"status": "(failed|stopped)"' "${state}" 2>/dev/null; then
      echo "capacity run width=${width} did not complete" >&2
      exit 5
    fi
    sleep 120
  done
done

declare -a PIDS=()
for spec in "${RUNS[@]}"; do
  IFS='|' read -r gpu width _ run_name <<<"${spec}"
  run="${STORAGE_ROOT}/runs/updater/${run_name}"
  eval_dir="${OUTPUT}/width${width}"
  if [[ -s "${eval_dir}/summary.json" ]]; then
    continue
  fi
  mkdir -p "${eval_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m activemap.cli evaluate-updater \
    "${run}/best_quality.pt" "${SAMPLES}" "${eval_dir}" \
    --split val --device auto --batch-size 64 --num-workers 4 \
    --bootstrap 5000 --seed 20260716 --edit-decoding auto \
    >"${eval_dir}/eval.log" 2>&1 &
  PIDS+=("$!")
  echo "started width=${width} validation on gpu=${gpu}, pid=$!"
done
for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

"${PYTHON}" scripts/summarize_sn7_concat_unet_capacity.py \
  "${OUTPUT}/paper_artifacts" \
  --run "32,2930448=${STORAGE_ROOT}/runs/updater/concat_unet_three_seed_val_20260729/seed20260716/summary.json" \
  --run "48,6586000=${OUTPUT}/width48/summary.json" \
  --run "64,11701776=${OUTPUT}/width64/summary.json"

"${PYTHON}" scripts/plot_sn7_concat_unet_training.py \
  "${OUTPUT}/training_curves_v2" \
  --history "width32=${WIDTH32_HISTORY}" \
  --history "width48=${STORAGE_ROOT}/runs/updater/v4_hierarchical_vector_change_scratch_width48_seed20260716/history.jsonl" \
  --history "width64=${STORAGE_ROOT}/runs/updater/v4_hierarchical_vector_change_scratch_width64_seed20260716/history.jsonl"

CUDA_VISIBLE_DEVICES=2 "${PYTHON}" scripts/export_concat_unet_capacity_masks.py \
  "${SAMPLES}" "${OUTPUT}/qualitative_masks" \
  --split val --device auto --per-edit 4 --output-size 512 \
  --checkpoint "width32=${STORAGE_ROOT}/runs/updater/v4_hierarchical_vector_change_scratch_seed20260716/best_quality.pt" \
  --checkpoint "width48=${STORAGE_ROOT}/runs/updater/v4_hierarchical_vector_change_scratch_width48_seed20260716/best_quality.pt" \
  --checkpoint "width64=${STORAGE_ROOT}/runs/updater/v4_hierarchical_vector_change_scratch_width64_seed20260716/best_quality.pt" \
  --predictions "width32=${STORAGE_ROOT}/runs/updater/concat_unet_three_seed_val_20260729/seed20260716/predictions.jsonl" \
  --predictions "width48=${OUTPUT}/width48/predictions.jsonl" \
  --predictions "width64=${OUTPUT}/width64/predictions.jsonl"

echo "[$(date --iso-8601=seconds)] concat U-Net capacity ablation complete"
