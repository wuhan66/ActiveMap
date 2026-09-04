#!/usr/bin/env bash
set -euo pipefail

# Germany is the only fitting source. Louisiana-East Training Public is external val.
PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
GIS_PYTHON="${STORE}/envs/activemap-gis/bin/python"
PYTHON="${STORE}/envs/activemap-opencd/bin/python"
OPENCD="${STORE}/external/open_cd"
RAW="${STORE}/datasets/spacenet8/raw/Louisiana-East_Training_Public.tar.gz"
EXTRACT_PARENT="${STORE}/datasets/spacenet8/extracted"
EPISODES="${STORE}/episodes/spacenet8_louisiana_east_external_val_full_20260904"
ALIGNED="${STORE}/processed/spacenet8_louisiana_east/external_val_full_512_20260904"
MULTI="${STORE}/processed/spacenet8_louisiana_east/multi_post_external_val_full_512_20260904"
ROOT="${STORE}/runs/spacenet8_louisiana_east/frozen_germany_v2_external_20260904"
GERMANY_ROOT="${STORE}/runs/spacenet8_germany/active_multi_post_spatial_v2_512"
CHECKPOINT_ROOT="${STORE}/runs/spacenet8_germany/opencd_spatial_active_block4_buffer1_v2_512"
CONFIG="${OPENCD}/configs/changeformer/changeformer_mit-b0_256x256_40k_levircd.py"
LOGS="${STORE}/logs/spacenet8_louisiana_east_external_20260904"

[[ -s "${RAW}" ]] || { echo "missing archive: ${RAW}" >&2; exit 2; }
mkdir -p "${EXTRACT_PARENT}" "${ROOT}" "${LOGS}"

if [[ ! -d "${EXTRACT_PARENT}/Louisiana-East_Training_Public" ]]; then
  tar -xzf "${RAW}" -C "${EXTRACT_PARENT}"
fi
DATASET="${EXTRACT_PARENT}/Louisiana-East_Training_Public"
[[ -d "${DATASET}/annotations" && -d "${DATASET}/PRE-event" && -d "${DATASET}/POST-event" ]] || {
  echo "unexpected Louisiana-East archive layout" >&2
  exit 3
}

cd "${PROJECT}"
if [[ ! -s "${EPISODES}/episodes.jsonl" ]]; then
  PYTHONPATH=src:. "${GIS_PYTHON}" scripts/build_spacenet8_disaster_smoke.py \
    "${DATASET}" "${EPISODES}" --limit 0 --selection sorted \
    --region louisiana-east --aoi-id louisiana-east --fixed-split val
fi
if [[ ! -s "${ALIGNED}/manifest.jsonl" ]]; then
  PYTHONPATH=src:. "${GIS_PYTHON}" scripts/prepare_spacenet8_flood_smoke.py \
    "${EPISODES}/episodes.jsonl" "${ALIGNED}" --size 512
fi
if [[ ! -s "${MULTI}/manifest.jsonl" ]]; then
  PYTHONPATH=src:. "${GIS_PYTHON}" scripts/prepare_spacenet8_multi_post_candidates.py \
    "${EPISODES}/episodes.jsonl" "${ALIGNED}/manifest.jsonl" "${MULTI}" --size 512
fi

run_seed() {
  local gpu="$1" seed="$2"
  local output="${ROOT}/seed${seed}"
  [[ -s "${output}/per_candidate.jsonl" ]] && return
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src:. "${PYTHON}" \
    scripts/infer_spacenet8_multi_post_candidates.py "${MULTI}/manifest.jsonl" \
    "${OPENCD}" "${CONFIG}" "${CHECKPOINT_ROOT}/changeformer_weight5_seed${seed}/best.pt" \
    "${output}" --device cuda:0 --save-masks >"${LOGS}/seed${seed}.log" 2>&1
}

run_seed 4 20260802 & p1=$!
run_seed 5 20260803 & p2=$!
run_seed 7 20260804 & p3=$!
wait "${p1}"
wait "${p2}"
wait "${p3}"

COMBINED="${ROOT}/germany_train_louisiana_val"
if [[ ! -s "${COMBINED}/summary.json" ]]; then
  PYTHONPATH=src:. "${GIS_PYTHON}" scripts/combine_spacenet8_external_validation.py \
    "${GERMANY_ROOT}" "${ROOT}" "${COMBINED}"
fi
EVALUATION="${ROOT}/selector_safe_commit"
if [[ ! -s "${EVALUATION}/summary.json" ]]; then
  PYTHONPATH=src:. "${GIS_PYTHON}" scripts/evaluate_spacenet8_active_multi_post.py \
    "${EVALUATION}" "${COMBINED}"/seed*_per_candidate.jsonl
fi
PYTHONPATH=src:. "${GIS_PYTHON}" scripts/bootstrap_spacenet8_active_multi_post.py \
  "${ROOT}/bootstrap5000" "${COMBINED}"/seed*_per_candidate.jsonl --draws 5000
date -Is >"${ROOT}/COMPLETED"
