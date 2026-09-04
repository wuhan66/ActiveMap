#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/wh/projects/activemap-v1
STORE=/home/wh/ActiveMap
PYTHON="$STORE/envs/activemap-opencd/bin/python"
MANIFEST="$STORE/processed/sn7_v1/updater_v4_cap20/train_val_complete_20260726.jsonl"
OPENCD="$STORE/external/open_cd"
ROOT="$STORE/runs/external_baselines/sn7/opencd_followup_20260731"
LOG_ROOT="$STORE/logs/opencd_followup_20260731"
MIT_B0="$STORE/models/opencd_ban/mit_b0_20220624-7e0fe6dd.pth"

mkdir -p "$ROOT" "$LOG_ROOT"
cd "$PROJECT"

run_one() {
  local gpu="$1" name="$2" seed="$3" config="$4" lr="$5" batch="$6"
  local input_mode="$7" class_weight="$8" pretrained="${9:-}"
  local run="$ROOT/${name}_${input_mode}_seed${seed}"
  local log="$LOG_ROOT/${name}_${input_mode}_seed${seed}.log"
  [[ ! -e "$run" ]] || { echo "Refusing to overwrite $run" >&2; return 2; }
  local pretrained_args=()
  [[ -z "$pretrained" ]] || pretrained_args=(--pretrained-checkpoint "$pretrained")
  {
    echo "START $(date -Is) physical_gpu=$gpu model=$name input=$input_mode seed=$seed"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/train_sn7_opencd_generic.py \
      "$MANIFEST" "$OPENCD" "$config" "$run" \
      --model-name "$name" --input-mode "$input_mode" \
      --device cuda:0 --image-size 128 --batch-size "$batch" --workers 8 \
      --epochs 40 --min-epochs 10 --patience 8 \
      --learning-rate "$lr" --weight-decay 0.0001 \
      --positive-class-weight "$class_weight" --lovasz-weight 0.75 \
      --gradient-clip 1.0 --seed "$seed" "${pretrained_args[@]}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/evaluate_sn7_opencd_generic.py \
      "$run/best.pt" "$MANIFEST" "$OPENCD" "$config" "$run/val_audit" \
      --split val --input-mode "$input_mode" --device cuda:0 \
      --image-size 128 --batch-size "$batch" --workers 8
    echo "COMPLETE $(date -Is) physical_gpu=$gpu model=$name input=$input_mode seed=$seed"
  } >"$log" 2>&1
}

CHANGEFORMER="$OPENCD/configs/changeformer/changeformer_mit-b0_256x256_40k_levircd.py"
BIT="$OPENCD/configs/bit/bit_r18_256x256_40k_levircd.py"
FCSIAM="$OPENCD/configs/fcsn/fc_siam_diff_256x256_40k_levircd.py"

run_one 1 changeformer 20260802 "$CHANGEFORMER" 0.00006 8 image_prior 5 "$MIT_B0" &
run_one 2 changeformer 20260731 "$CHANGEFORMER" 0.00006 8 image_only 5 "$MIT_B0" &
run_one 3 changeformer 20260731 "$CHANGEFORMER" 0.00006 8 prior_only 5 "$MIT_B0" &
run_one 4 bit_r18 20260802 "$BIT" 0.0001 8 image_prior 5 &
run_one 5 fc_siam_diff 20260802 "$FCSIAM" 0.0001 16 image_prior 5 &
wait
