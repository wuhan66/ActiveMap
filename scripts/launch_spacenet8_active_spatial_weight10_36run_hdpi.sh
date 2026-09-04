#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-opencd/bin/python"
DATA_ROOT="${STORE}/processed/spacenet8_germany"
OPENCD="${STORE}/external/open_cd"
CHANGE_CONFIG="${OPENCD}/configs/changeformer/changeformer_mit-b0_256x256_40k_levircd.py"
BAN_CONFIG="${PROJECT}/configs/opencd/ban_vit_b16_clip_mit_b0_local_hdpi.py"
MIT="${STORE}/models/opencd_ban/mit_b0_20220624-7e0fe6dd.pth"
CLIP="${STORE}/models/opencd_ban/clip_vit-base-patch16-224_3rdparty-d08f8887.pth"
ROOT="${STORE}/runs/spacenet8_germany/opencd_spatial_active_weight10_512"
LOGS="${STORE}/logs/spacenet8_active_spatial_weight10"

mkdir -p "$ROOT" "$LOGS"
cd "$PROJECT"

run_series() {
  local gpu="$1" backend="$2" seed="$3"
  local config pretrained batch
  if [[ "$backend" == changeformer ]]; then
    config="$CHANGE_CONFIG"; pretrained="$MIT"; batch=3
  else
    config="$BAN_CONFIG"; pretrained="$CLIP"; batch=1
  fi
  for rank in 2 3 10 25 50 100; do
    local data="${DATA_ROOT}/aligned_full202_spatial_active_rank${rank}_block4_buffer1_512"
    local label="rank${rank}_${backend}_weight10_seed${seed}"
    local output="${ROOT}/${label}"
    [[ -s "${output}/summary.json" ]] && continue
    [[ ! -e "$output" ]] || { echo "Refusing incomplete output: $output" >&2; return 3; }
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src:. "$PYTHON" \
      scripts/train_spacenet8_opencd_smoke.py \
      "${data}/manifest.jsonl" "$OPENCD" "$config" "$output" \
      --model-name "$label" --pretrained-checkpoint "$pretrained" \
      --mode pre_post --device cuda:0 --epochs 40 --min-epochs 12 --patience 8 \
      --batch-size "$batch" --workers 3 --learning-rate 0.00006 \
      --positive-weight 10 --seed "$seed" >"${LOGS}/${label}.log" 2>&1
  done
}

pids=()
run_series 1 changeformer 20260802 & pids+=("$!")
run_series 2 changeformer 20260803 & pids+=("$!")
run_series 3 changeformer 20260804 & pids+=("$!")
run_series 4 ban_mit 20260802 & pids+=("$!")
run_series 5 ban_mit 20260803 & pids+=("$!")
run_series 7 ban_mit 20260804 & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done
