#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-opencd/bin/python"
DATA="${STORE}/processed/spacenet8_germany/aligned_full202_spatial_block4_buffer1_v1_512"
OPENCD="${STORE}/external/open_cd"
CHANGEFORMER="${OPENCD}/configs/changeformer/changeformer_mit-b0_256x256_40k_levircd.py"
CHANGER="${OPENCD}/configs/changer/changer_ex_r18_512x512_40k_levircd.py"
PRETRAINED="${STORE}/models/opencd_ban/mit_b0_20220624-7e0fe6dd.pth"
ROOT="${STORE}/runs/spacenet8_germany/opencd_spatial_block4_buffer1_v1_512"
LOGS="${STORE}/logs/spacenet8_opencd_spatial_block4_buffer1_v1_512"

mkdir -p "$ROOT" "$LOGS"
cd "$PROJECT"

run_one() {
  local gpu="$1" label="$2" model="$3" config="$4" seed="$5" batch="$6"
  local output="${ROOT}/${label}"
  local pretrained_args=()
  [[ "$model" == changer_r18_512 ]] || pretrained_args=(--pretrained-checkpoint "$PRETRAINED")
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
    --learning-rate 0.00006 --positive-weight 10 --seed "$seed" \
    >"${LOGS}/${label}.log" 2>&1 < /dev/null &
  echo "$!" >"${LOGS}/${label}.pid"
  echo "started=${label} gpu=${gpu} pid=$!"
}

run_one 1 changeformer_weight10_seed20260802 ChangeFormer-MiT-B0-512 "$CHANGEFORMER" 20260802 3
run_one 2 changeformer_weight10_seed20260803 ChangeFormer-MiT-B0-512 "$CHANGEFORMER" 20260803 3
run_one 3 changeformer_weight10_seed20260804 ChangeFormer-MiT-B0-512 "$CHANGEFORMER" 20260804 3
run_one 4 changer_r18_weight10_seed20260802 changer_r18_512 "$CHANGER" 20260802 4
run_one 5 changer_r18_weight10_seed20260803 changer_r18_512 "$CHANGER" 20260803 4
run_one 7 changer_r18_weight10_seed20260804 changer_r18_512 "$CHANGER" 20260804 4
