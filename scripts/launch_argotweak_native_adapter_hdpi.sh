#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
LEGACY_PYTHON="${STORE}/envs/argotweak-legacy/bin/python"
ACTIVE_PYTHON="${STORE}/envs/activemap-gis/bin/python"
DATA="${STORE}/datasets/argotweak/tbv_balanced_24_8_v1"
SOURCE="${STORE}/outputs/argotweak/balanced_24_8_v1"
ROOT="${ROOT:-${STORE}/runs/argotweak/native_adapter_frozen_baseline_v1}"
MATCH_DISTANCE="${MATCH_DISTANCE:-1.5}"
COMMIT_CONFIDENCE="${COMMIT_CONFIDENCE:-0.5}"

mkdir -p "${ROOT}/shards" "${ROOT}/proposals" "${ROOT}/episodes"
exec 9>"${ROOT}/.launcher.lock"
flock -n 9 || exit 0
[[ ! -e "${ROOT}/MATRIX_COMPLETED" ]] || exit 0
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

specs=(
  "train_00_03:train_argotweak_00_03.pkl:train"
  "train_04_07:train_argotweak_04_07.pkl:train"
  "train_08_11:train_argotweak_08_11.pkl:train"
  "train_12_15:train_argotweak_12_15.pkl:train"
  "train_16_18:train_argotweak_16_18.pkl:train"
  "train_19:train_argotweak_19.pkl:train"
  "train_20_23:train_argotweak_20_23.pkl:train"
  "val_00_03:val_argotweak_00_03.pkl:val"
  "val_04_07:val_argotweak_04_07.pkl:val"
)

train_shards=()
val_shards=()
for spec in "${specs[@]}"; do
  IFS=: read -r name annotation split <<<"${spec}"
  output="${ROOT}/shards/${name}.jsonl"
  if [[ ! -s "${output}" ]]; then
    "${LEGACY_PYTHON}" scripts/export_argotweak_official_proposals.py \
      --results "${SOURCE}/${name}/results.pkl" \
      --annotations "${DATA}/official_shards/${annotation}" \
      --output "${output}" --object-threshold 0.3 \
      --object-match-distance "${MATCH_DISTANCE}"
  fi
  if [[ "${split}" == "train" ]]; then
    train_shards+=("${output}")
  else
    val_shards+=("${output}")
  fi
done

for split in train val; do
  merged="${ROOT}/proposals/${split}.jsonl"
  if [[ ! -s "${merged}" ]]; then
    if [[ "${split}" == "train" ]]; then shards=("${train_shards[@]}"); else shards=("${val_shards[@]}"); fi
    "${ACTIVE_PYTHON}" scripts/merge_argotweak_proposal_shards.py \
      --inputs "${shards[@]}" --output "${merged}"
  fi
  native="${ROOT}/episodes/${split}.jsonl"
  if [[ ! -s "${native}" ]]; then
    "${ACTIVE_PYTHON}" scripts/build_argotweak_native_episodes.py \
      --proposals "${merged}" \
      --scenes "${DATA}/activemap_episodes_official_v2/${split}_scenes.jsonl" \
      --output "${native}" --commit-confidence "${COMMIT_CONFIDENCE}"
  fi
done

touch "${ROOT}/MATRIX_COMPLETED"
