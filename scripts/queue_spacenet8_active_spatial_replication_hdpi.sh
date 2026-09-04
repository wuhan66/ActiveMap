#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-opencd/bin/python"
OPENCD="${STORE}/external/open_cd"
CONFIG="${OPENCD}/configs/changeformer/changeformer_mit-b0_256x256_40k_levircd.py"
PRETRAINED="${STORE}/models/opencd_ban/mit_b0_20220624-7e0fe6dd.pth"
CURRENT="${STORE}/runs/spacenet8_germany/opencd_spatial_active_block4_buffer1_v2_512"
ROOT="${STORE}/runs/spacenet8_germany/opencd_spatial_active_replication_512"
LOGS="${STORE}/logs/spacenet8_active_spatial_replication"

mkdir -p "$ROOT" "$LOGS"
for seed in 20260802 20260803 20260804; do
  for label in \
    "ban_mit_weight10_seed${seed}" "ban_mit_weight15_seed${seed}" \
    "ban_mit_post_only_weight5_seed${seed}" "ban_mit_pre_only_weight5_seed${seed}"; do
    while [[ ! -s "${CURRENT}/${label}/summary.json" ]]; do sleep 30; done
  done
done
cd "$PROJECT"

run_one() {
  local gpu="$1" rank="$2" seed="$3"
  local data="${STORE}/processed/spacenet8_germany/aligned_full202_spatial_active_rank${rank}_block4_buffer1_512"
  local label="rank${rank}_changeformer_weight5_seed${seed}"
  local output="${ROOT}/${label}"
  [[ ! -s "${output}/summary.json" ]] || return
  if [[ -e "$output" ]]; then
    while [[ ! -s "${output}/summary.json" ]]; do sleep 30; done
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src:. "$PYTHON" \
    scripts/train_spacenet8_opencd_smoke.py \
    "${data}/manifest.jsonl" "$OPENCD" "$CONFIG" "$output" \
    --model-name "$label" --pretrained-checkpoint "$PRETRAINED" \
    --mode pre_post --device cuda:0 --epochs 40 --min-epochs 12 --patience 8 \
    --batch-size 3 --workers 4 --learning-rate 0.00006 \
    --positive-weight 5 --seed "$seed" >"${LOGS}/${label}.log" 2>&1
}

pids=()
run_one 1 10 20260802 & pids+=("$!")
run_one 2 10 20260803 & pids+=("$!")
run_one 3 10 20260804 & pids+=("$!")
run_one 4 50 20260802 & pids+=("$!")
run_one 5 50 20260803 & pids+=("$!")
run_one 7 50 20260804 & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done
