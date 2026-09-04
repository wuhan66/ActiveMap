#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-opencd/bin/python"
DATA="${STORE}/processed/spacenet8_germany/aligned_full202_stratified_v2_512"
OPENCD="${STORE}/external/open_cd"
SNUNET="${OPENCD}/configs/snunet/snunet_c32_256x256_40k_levircd.py"
LIGHTCD="${OPENCD}/configs/lightcdnet/lightcdnet_b_256x256_40k_levircd.py"
ROOT="${STORE}/runs/spacenet8_germany/opencd_light_baselines_stratified_v2_512"
LOGS="${STORE}/logs/spacenet8_opencd_light_baselines_stratified_v2_512"

mkdir -p "$ROOT" "$LOGS"
cd "$PROJECT"

run_one() {
  local gpu="$1" label="$2" model="$3" config="$4" seed="$5" batch="$6"
  local output="${ROOT}/${label}"
  if [[ -s "${output}/summary.json" ]]; then
    echo "skip_completed=${label}"
    return
  fi
  [[ ! -e "$output" ]] || { echo "Refusing incomplete existing output: $output" >&2; return 2; }
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src:. nohup "$PYTHON" \
    scripts/train_spacenet8_opencd_smoke.py \
    "${DATA}/manifest.jsonl" "$OPENCD" "$config" "$output" \
    --model-name "$model" --mode pre_post --device cuda:0 \
    --epochs 40 --min-epochs 12 --patience 8 --batch-size "$batch" --workers 3 \
    --learning-rate 0.00006 --positive-weight 10 --seed "$seed" \
    >"${LOGS}/${label}.log" 2>&1 < /dev/null &
  echo "$!" >"${LOGS}/${label}.pid"
  echo "started=${label} gpu=${gpu} pid=$!"
}

run_one 1 snunet_c32_pre_post_weight10_seed20260802 snunet_c32_512 "$SNUNET" 20260802 4
run_one 2 snunet_c32_pre_post_weight10_seed20260803 snunet_c32_512 "$SNUNET" 20260803 4
run_one 3 snunet_c32_pre_post_weight10_seed20260804 snunet_c32_512 "$SNUNET" 20260804 4
run_one 4 lightcdnet_b_pre_post_weight10_seed20260802 lightcdnet_b_512 "$LIGHTCD" 20260802 8
run_one 5 lightcdnet_b_pre_post_weight10_seed20260803 lightcdnet_b_512 "$LIGHTCD" 20260803 8
run_one 7 lightcdnet_b_pre_post_weight10_seed20260804 lightcdnet_b_512 "$LIGHTCD" 20260804 8
