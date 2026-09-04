#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-opencd/bin/python"
GIS_PYTHON="${STORE}/envs/activemap-gis/bin/python"
OPENCD="${STORE}/external/open_cd"
TRAIN="${STORE}/runs/spacenet8_germany/opencd_modern_baselines_stratified_v2_512"
OUTPUT="${STORE}/runs/spacenet8_germany/active_multi_post_modern_cross_region_20260802"
LOGS="${STORE}/logs/spacenet8_modern_cross_region_20260802"
CHANGER_CONFIG="${OPENCD}/configs/changer/changer_ex_r18_512x512_40k_levircd.py"
TINYCD_CONFIG="${OPENCD}/configs/tinycd_v2/tinycd_v2_s_256x256_40k_levircd.py"

mkdir -p "${OUTPUT}" "${LOGS}"
cd "${PROJECT}"

run_backend() {
  local gpu="$1" backend="$2" config label_prefix
  if [[ "${backend}" == "changer" ]]; then
    config="${CHANGER_CONFIG}"
    label_prefix="changer_r18_pre_post_weight10"
  else
    config="${TINYCD_CONFIG}"
    label_prefix="tinycd_v2_s_pre_post_weight10"
  fi
  for rank in 2 100; do
    local data="${STORE}/processed/spacenet8_germany/multi_post_spatial_active_rank${rank}_block4_buffer1_512"
    for seed in 20260802 20260803 20260804; do
      local checkpoint="${TRAIN}/${label_prefix}_seed${seed}/best.pt"
      local out="${OUTPUT}/rank${rank}_${backend}_seed${seed}"
      [[ -s "${data}/summary.json" && -s "${checkpoint}" ]]
      if [[ -s "${out}/summary.json" ]]; then continue; fi
      [[ ! -e "${out}" ]] || { echo "refusing incomplete output: ${out}" >&2; return 3; }
      CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src:. "${PYTHON}" \
        scripts/infer_spacenet8_multi_post_candidates.py \
        "${data}/manifest.jsonl" "${OPENCD}" "${config}" \
        "${checkpoint}" "${out}" --device cuda:0 --save-masks \
        >"${LOGS}/rank${rank}_${backend}_seed${seed}.log" 2>&1
    done
  done
}

run_backend 5 changer >"${LOGS}/changer_series.log" 2>&1 & p1=$!
run_backend 7 tinycd >"${LOGS}/tinycd_series.log" 2>&1 & p2=$!
wait "${p1}"
wait "${p2}"

for rank in 2 100; do
  for backend in changer tinycd; do
    inputs=()
    for seed in 20260802 20260803 20260804; do
      inputs+=("${OUTPUT}/rank${rank}_${backend}_seed${seed}/per_candidate.jsonl")
    done
    result="${OUTPUT}/rank${rank}_${backend}_selector_safe_commit"
    PYTHONPATH=src:. "${GIS_PYTHON}" scripts/evaluate_spacenet8_active_multi_post.py \
      "${result}" "${inputs[@]}" >"${LOGS}/rank${rank}_${backend}_evaluate.log" 2>&1
    bootstrap="${OUTPUT}/rank${rank}_${backend}_bootstrap5000"
    PYTHONPATH=src:. "${GIS_PYTHON}" scripts/bootstrap_spacenet8_active_multi_post.py \
      "${bootstrap}" "${inputs[@]}" --draws 5000 \
      >"${LOGS}/rank${rank}_${backend}_bootstrap.log" 2>&1
  done
done
date -Is >"${OUTPUT}/MATRIX_COMPLETED"
