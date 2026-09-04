#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-opencd/bin/python"
DATA="${STORE}/processed/spacenet8_germany/aligned_full202_stratified_v2_512"
OPENCD="${STORE}/external/open_cd"
CHANGEFORMER="${OPENCD}/configs/changeformer/changeformer_mit-b0_256x256_40k_levircd.py"
BIT="${OPENCD}/configs/bit/bit_r18_256x256_40k_levircd.py"
PRETRAINED="${STORE}/models/opencd_ban/mit_b0_20220624-7e0fe6dd.pth"
ROOT="${STORE}/runs/spacenet8_germany/opencd_full202_stratified_v2_512"
LOGS="${STORE}/logs/spacenet8_opencd_full202_stratified_v2_512_controls"

mkdir -p "${ROOT}" "${LOGS}"
cd "${PROJECT}"

run_one() {
  local gpu="$1" label="$2" model="$3" mode="$4" config="$5" batch="$6"
  local output="${ROOT}/${label}"
  [[ ! -e "${output}" ]] || { echo "Refusing existing output: ${output}" >&2; return 2; }
  local pretrained_args=()
  [[ "${model}" != bit_r18_512 ]] || pretrained_args=()
  [[ "${model}" == bit_r18_512 ]] || pretrained_args=(--pretrained-checkpoint "${PRETRAINED}")
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src:. nohup "${PYTHON}" \
    scripts/train_spacenet8_opencd_smoke.py \
    "${DATA}/manifest.jsonl" "${OPENCD}" "${config}" "${output}" \
    --model-name "${model}" "${pretrained_args[@]}" --mode "${mode}" --device cuda:0 \
    --epochs 40 --min-epochs 12 --patience 8 --batch-size "${batch}" --workers 4 \
    --learning-rate 0.00006 --positive-weight 10 --seed 20260802 \
    >"${LOGS}/${label}.log" 2>&1 < /dev/null &
  echo "$!" >"${LOGS}/${label}.pid"
  echo "${label}: GPU ${gpu}, PID $!"
}

run_one 1 post_only_weight10_seed20260802 ChangeFormer-MiT-B0-512 post_only "${CHANGEFORMER}" 2
run_one 2 pre_only_weight10_seed20260802 ChangeFormer-MiT-B0-512 pre_only "${CHANGEFORMER}" 2
run_one 3 bit_r18_pre_post_weight10_seed20260802 bit_r18_512 pre_post "${BIT}" 4
