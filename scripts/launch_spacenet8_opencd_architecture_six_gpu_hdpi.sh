#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-opencd/bin/python"
DATA="${STORE}/processed/spacenet8_germany/aligned_full202_stratified_v2"
OPENCD="${STORE}/external/open_cd"
ROOT="${STORE}/runs/spacenet8_germany/opencd_architecture_stratified_v2"
LOGS="${STORE}/logs/spacenet8_opencd_architecture_stratified_v2"

mkdir -p "${ROOT}" "${LOGS}"
cd "${PROJECT}"

run_one() {
  local gpu="$1" model="$2" seed="$3" config="$4" batch="$5"
  local label="${model}_pre_post_weight10_seed${seed}"
  local output="${ROOT}/${label}"
  [[ ! -e "${output}" ]] || { echo "Refusing existing output: ${output}" >&2; return 2; }
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src:. nohup "${PYTHON}" \
    scripts/train_spacenet8_opencd_smoke.py \
    "${DATA}/manifest.jsonl" "${OPENCD}" "${config}" "${output}" \
    --model-name "${model}" --mode pre_post --device cuda:0 \
    --epochs 40 --min-epochs 12 --patience 8 --batch-size "${batch}" --workers 4 \
    --learning-rate 0.0001 --positive-weight 10 --seed "${seed}" \
    >"${LOGS}/${label}.log" 2>&1 < /dev/null &
  echo "$!" >"${LOGS}/${label}.pid"
  echo "${label}: GPU ${gpu}, PID $!"
}

BIT="${OPENCD}/configs/bit/bit_r18_256x256_40k_levircd.py"
FCSIAM="${OPENCD}/configs/fcsn/fc_siam_diff_256x256_40k_levircd.py"

run_one 1 bit_r18 20260802 "${BIT}" 8
run_one 2 bit_r18 20260803 "${BIT}" 8
run_one 3 bit_r18 20260804 "${BIT}" 8
run_one 4 fc_siam_diff 20260802 "${FCSIAM}" 16
run_one 5 fc_siam_diff 20260803 "${FCSIAM}" 16
run_one 7 fc_siam_diff 20260804 "${FCSIAM}" 16
