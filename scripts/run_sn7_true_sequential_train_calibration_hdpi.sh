#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
MATRIX="${1:-gate_delta}"
TRAIN_EPISODES="${TRAIN_EPISODES:-${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_train_bundle_v1/episodes_train.jsonl}"
ONLINE_REGISTRY="${ONLINE_REGISTRY:-${STORAGE_ROOT}/runs/selector/sn7_mask_features_v2_online_observable_seed20260902_strict_v1/online_controller_registry.yaml}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_true_sequential_train_calibration_${MATRIX}_20260902_v1}"
GPU_IDS=(${GPU_IDS:-1 2 3 4})
ASSET_MAP="${ASSET_MAP:-/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}}"

case "${MATRIX}" in
  gate_delta)
    POINTS=(
      "g015_d010|0.15|0.10|0.50|0|0.65"
      "g025_d010|0.25|0.10|0.50|0|0.65"
      "g015_d015|0.15|0.15|0.50|0|0.65"
      "g025_d015|0.25|0.15|0.50|0|0.65"
    )
    ;;
  writeback)
    POINTS=(
      "g015_d000|0.15|0.00|0.50|0|0.65"
      "g015_d0025|0.15|0.025|0.50|0|0.65"
      "g015_d005|0.15|0.05|0.50|0|0.65"
      "g015_d0075|0.15|0.075|0.50|0|0.65"
    )
    ;;
  mask_threshold)
    POINTS=(
      "g015_t035|0.15|0.00|0.35|0|0.65"
      "g015_t040|0.15|0.00|0.40|0|0.65"
      "g015_t045|0.15|0.00|0.45|0|0.65"
      "g015_t050|0.15|0.00|0.50|0|0.65"
    )
    ;;
  component_filter)
    POINTS=(
      "g015_t040_c004|0.15|0.00|0.40|4|0.65"
      "g015_t040_c016|0.15|0.00|0.40|16|0.65"
      "g015_t040_c032|0.15|0.00|0.40|32|0.65"
      "g015_t040_c064|0.15|0.00|0.40|64|0.65"
    )
    ;;
  safe_confidence)
    POINTS=(
      "g015_t040_c004_s065|0.15|0.00|0.40|4|0.65"
      "g015_t040_c004_s068|0.15|0.00|0.40|4|0.68"
      "g015_t040_c004_s070|0.15|0.00|0.40|4|0.70"
      "g015_t040_c004_s072|0.15|0.00|0.40|4|0.72"
    )
    ;;
  *)
    echo "unknown calibration matrix: ${MATRIX}" >&2
    exit 2
    ;;
esac

[[ ${#GPU_IDS[@]} -ge ${#POINTS[@]} ]] || { echo "four GPU ids required" >&2; exit 2; }
[[ -x "${PYTHON}" && -f "${TRAIN_EPISODES}" && -f "${ONLINE_REGISTRY}" ]] || {
  echo "missing Python, train episodes, or registry" >&2
  exit 2
}
[[ ! -e "${RUN_ROOT}" ]] || { echo "refusing to overwrite ${RUN_ROOT}" >&2; exit 2; }

mkdir -p "${RUN_ROOT}/logs"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

pids=()
for index in "${!POINTS[@]}"; do
  IFS='|' read -r name gate_threshold delta_margin mask_threshold min_component_pixels safe_confidence <<<"${POINTS[$index]}"
  output="${RUN_ROOT}/${name}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$index]}" "${PYTHON}" \
    scripts/evaluate_online_full_controller.py \
    "${TRAIN_EPISODES}" "${ONLINE_REGISTRY}" 20260902 "${output}" \
    --storage-root "${STORAGE_ROOT}" --project-root "${PROJECT_ROOT}" \
    --device cuda:0 --split train --max-chains 20 --image-size 512 \
    --budget 3.0 --max-candidates 16 --max-acquisitions 2 --max-tool-calls 4 \
    --minimum-chain-length 2 --threshold "${mask_threshold}" --delta-margin "${delta_margin}" \
    --min-delta-component-pixels "${min_component_pixels}" \
    --safe-confidence-threshold "${safe_confidence}" --safe-replay-iou-threshold 0.99 \
    --diagnostic-tool-gate-threshold-override "${gate_threshold}" \
    --asset-root-map "${ASSET_MAP}" \
    --policy direct_current_hypothesis_safe --policy active_selective_safe \
    >"${RUN_ROOT}/logs/${name}.log" 2>&1 &
  pids+=("$!")
  printf '%s\n' "$!" >"${RUN_ROOT}/logs/${name}.pid"
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
(( status == 0 )) || exit "${status}"

echo "completed train calibration under ${RUN_ROOT}"
