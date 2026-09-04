#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/wh/projects/activemap-v1
STORE=/home/wh/ActiveMap
PYTHON="$STORE/envs/activemap-opencd/bin/python"
MANIFEST="$STORE/processed/sn7_v1/updater_v4_cap20/train_val_complete_20260726.jsonl"
OPENCD="$STORE/external/open_cd"
ROOT="$STORE/runs/external_baselines/sn7/opencd_night_20260731"
LOG_ROOT="$STORE/logs/opencd_night_20260731"
MIT_B0="$STORE/models/opencd_ban/mit_b0_20220624-7e0fe6dd.pth"

mkdir -p "$ROOT" "$LOG_ROOT"
cd "$PROJECT"

run_one() {
  local gpu="$1"
  local name="$2"
  local seed="$3"
  local config="$4"
  local lr="$5"
  local batch="$6"
  local pretrained="${7:-}"
  local run="$ROOT/${name}_seed${seed}"
  local log="$LOG_ROOT/${name}_seed${seed}.log"

  if [[ -e "$run" ]]; then
    echo "Refusing to overwrite $run" >&2
    return 2
  fi
  local pretrained_args=()
  if [[ -n "$pretrained" ]]; then
    pretrained_args=(--pretrained-checkpoint "$pretrained")
  fi
  {
    echo "START $(date -Is) physical_gpu=$gpu model=$name seed=$seed"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/train_sn7_opencd_generic.py \
      "$MANIFEST" "$OPENCD" "$config" "$run" \
      --model-name "$name" \
      --device cuda:0 --image-size 128 --batch-size "$batch" --workers 8 \
      --epochs 40 --min-epochs 10 --patience 8 \
      --learning-rate "$lr" --weight-decay 0.0001 \
      --positive-class-weight 5.0 --lovasz-weight 0.75 \
      --gradient-clip 1.0 --seed "$seed" "${pretrained_args[@]}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/evaluate_sn7_opencd_generic.py \
      "$run/best.pt" "$MANIFEST" "$OPENCD" "$config" "$run/val_audit" \
      --split val --device cuda:0 --image-size 128 --batch-size "$batch" \
      --workers 8
    echo "COMPLETE $(date -Is) physical_gpu=$gpu model=$name seed=$seed"
  } >"$log" 2>&1
}

CHANGEFORMER="$OPENCD/configs/changeformer/changeformer_mit-b0_256x256_40k_levircd.py"
BIT="$OPENCD/configs/bit/bit_r18_256x256_40k_levircd.py"
FCSIAM="$OPENCD/configs/fcsn/fc_siam_diff_256x256_40k_levircd.py"

run_one 1 changeformer 20260731 "$CHANGEFORMER" 0.00006 8 "$MIT_B0" &
run_one 2 changeformer 20260801 "$CHANGEFORMER" 0.00006 8 "$MIT_B0" &
run_one 3 bit_r18 20260731 "$BIT" 0.001 8 &
run_one 4 fc_siam_diff 20260731 "$FCSIAM" 0.001 16 &
wait
