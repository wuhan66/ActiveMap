#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-opencd/bin/python"
DATA="${DATA:-${STORE}/processed/spacenet8_germany/aligned_smoke80_v1}"
OPENCD="${STORE}/external/open_cd"
CONFIG="${OPENCD}/configs/changeformer/changeformer_mit-b0_256x256_40k_levircd.py"
PRETRAINED="${STORE}/models/opencd_ban/mit_b0_20220624-7e0fe6dd.pth"
ROOT="${ROOT:-${STORE}/runs/spacenet8_germany/opencd_smoke80_v1}"
LOGS="${LOGS:-${STORE}/logs/spacenet8_opencd_smoke80_followup_v1}"

mkdir -p "${ROOT}" "${LOGS}"
cd "${PROJECT}"

run_one() {
  local gpu="$1" label="$2" seed="$3" positive_weight="$4"
  local mode="${5:-pre_post}"
  local output="${ROOT}/${label}"
  [[ ! -e "${output}" ]] || { echo "Refusing existing output: ${output}" >&2; return 2; }
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src:. nohup "${PYTHON}" \
    scripts/train_spacenet8_opencd_smoke.py \
    "${DATA}/manifest.jsonl" "${OPENCD}" "${CONFIG}" "${output}" \
    --pretrained-checkpoint "${PRETRAINED}" --mode "${mode}" --device cuda:0 \
    --epochs 40 --min-epochs 12 --patience 8 --batch-size 2 --workers 2 \
    --learning-rate 0.00006 --positive-weight "${positive_weight}" --seed "${seed}" \
    >"${LOGS}/${label}.log" 2>&1 < /dev/null &
  echo "$!" >"${LOGS}/${label}.pid"
  echo "${label}: GPU ${gpu}, PID $!"
}

if [[ "${MATRIX:-followup}" == "controls" ]]; then
  run_one 1 post_only_seed20260802_corrected_v1 20260802 20 post_only
  run_one 3 pre_only_seed20260802_corrected_v1 20260802 20 pre_only
elif [[ "${MATRIX:-followup}" == "full" ]]; then
  run_one 1 pre_post_seed20260802 20260802 20
  run_one 3 pre_post_seed20260803 20260803 20
  run_one 4 pre_post_seed20260804 20260804 20
  run_one 2 post_only_seed20260802_corrected_v1 20260802 20 post_only
  run_one 5 pre_only_seed20260802_corrected_v1 20260802 20 pre_only
else
  run_one 1 pre_post_seed20260803 20260803 20
  run_one 3 pre_post_seed20260804 20260804 20
  run_one 4 pre_post_seed20260805 20260805 20
  run_one 2 pre_post_weight10_seed20260802 20260802 10
  run_one 5 pre_post_weight30_seed20260802 20260802 30
fi
