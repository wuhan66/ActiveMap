#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-opencd/bin/python"
DATA="${STORE}/processed/spacenet8_germany/aligned_full202_spatial_block4_buffer1_v1_512"
OPENCD="${STORE}/external/open_cd"
CLIP="${STORE}/models/opencd_ban/clip_vit-base-patch16-224_3rdparty-d08f8887.pth"
BAN_MIT="${PROJECT}/configs/opencd/ban_vit_b16_clip_mit_b0_local_hdpi.py"
BAN_BIT="${OPENCD}/configs/ban/ban_vit-b16-clip_bit_512x512_40k_levircd.py"
ROOT="${STORE}/runs/spacenet8_germany/opencd_spatial_ban_block4_buffer1_v1_512"
LOGS="${STORE}/logs/spacenet8_opencd_spatial_ban_block4_buffer1_v1_512"

for path in "$PYTHON" "${DATA}/manifest.jsonl" "$CLIP" "$BAN_MIT" "$BAN_BIT"; do
  [[ -e "$path" ]] || { echo "Required input is missing: $path" >&2; exit 2; }
done
mkdir -p "$ROOT" "$LOGS"
cd "$PROJECT"

run_one() {
  local gpu="$1" label="$2" config="$3" seed="$4"
  local output="${ROOT}/${label}"
  if [[ -s "${output}/summary.json" ]]; then
    echo "skip_completed=${label}"
    return
  fi
  [[ ! -e "$output" ]] || { echo "Refusing incomplete existing output: $output" >&2; return 2; }
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src:. nohup "$PYTHON" \
    scripts/train_spacenet8_opencd_smoke.py \
    "${DATA}/manifest.jsonl" "$OPENCD" "$config" "$output" \
    --model-name "$label" --pretrained-checkpoint "$CLIP" \
    --mode pre_post --device cuda:0 --epochs 40 --min-epochs 12 --patience 8 \
    --batch-size 1 --workers 3 --learning-rate 0.00006 \
    --positive-weight 5 --seed "$seed" \
    >"${LOGS}/${label}.log" 2>&1 < /dev/null &
  echo "$!" >"${LOGS}/${label}.pid"
  echo "started=${label} gpu=${gpu} pid=$!"
}

run_one 1 ban_mit_b0_local_weight5_seed20260802 "$BAN_MIT" 20260802
run_one 2 ban_mit_b0_local_weight5_seed20260803 "$BAN_MIT" 20260803
run_one 3 ban_mit_b0_local_weight5_seed20260804 "$BAN_MIT" 20260804
run_one 4 ban_bit_weight5_seed20260802 "$BAN_BIT" 20260802
run_one 5 ban_bit_weight5_seed20260803 "$BAN_BIT" 20260803
run_one 7 ban_bit_weight5_seed20260804 "$BAN_BIT" 20260804
