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
LOGS="${STORE}/logs/spacenet8_opencd_full202_stratified_v2_512_completion"

mkdir -p "$ROOT" "$LOGS"
cd "$PROJECT"

run_one() {
  local gpu="$1" label="$2" model="$3" config="$4" seed="$5" weight="$6" batch="$7"
  local output="${ROOT}/${label}"
  local pretrained_args=()
  [[ "$model" == bit_r18_512 ]] || pretrained_args=(--pretrained-checkpoint "$PRETRAINED")
  if [[ -s "${output}/summary.json" ]]; then
    echo "skip_completed=${label}"
    return
  fi
  [[ ! -e "$output" ]] || { echo "Refusing incomplete existing output: $output" >&2; return 2; }
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src:. nohup "$PYTHON" \
    scripts/train_spacenet8_opencd_smoke.py \
    "${DATA}/manifest.jsonl" "$OPENCD" "$config" "$output" \
    --model-name "$model" "${pretrained_args[@]}" --mode pre_post --device cuda:0 \
    --epochs 40 --min-epochs 12 --patience 8 --batch-size "$batch" --workers 4 \
    --learning-rate 0.00006 --positive-weight "$weight" --seed "$seed" \
    >"${LOGS}/${label}.log" 2>&1 < /dev/null &
  echo "$!" >"${LOGS}/${label}.pid"
  echo "started=${label} gpu=${gpu} pid=$!"
}

run_one 1 bit_r18_pre_post_weight10_seed20260803 bit_r18_512 "$BIT" 20260803 10 4
run_one 2 bit_r18_pre_post_weight10_seed20260804 bit_r18_512 "$BIT" 20260804 10 4
run_one 3 pre_post_weight5_seed20260803 ChangeFormer-MiT-B0-512 "$CHANGEFORMER" 20260803 5 2
run_one 4 pre_post_weight5_seed20260804 ChangeFormer-MiT-B0-512 "$CHANGEFORMER" 20260804 5 2
run_one 5 pre_post_weight15_seed20260803 ChangeFormer-MiT-B0-512 "$CHANGEFORMER" 20260803 15 2
run_one 7 pre_post_weight15_seed20260804 ChangeFormer-MiT-B0-512 "$CHANGEFORMER" 20260804 15 2
