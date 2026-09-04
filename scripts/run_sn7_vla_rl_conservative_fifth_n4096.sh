#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"
MODEL="/home/wh/hf_models/Qwen3-VL-4B-Instruct"
ADAPTER="${RUN}/qwen3vl4b_weighted_replication_seed2/seed20260718/final"
RL_DATA="${RUN}/active_catalog_rl_states"
DATA="${STORAGE_ROOT}/processed/sn7_v1/agent/sequential_selector_v1/full"
SEED=20260718
GPU=4
NAME="contextual_rl_conservative_lr2p5e6_kl010_bal25_n4096"
OUTPUT="${RUN}/${NAME}"
ROLLOUT="${RUN}/closed_loop_rl_conservative_lr2p5e6_kl010_bal25_n4096_seed20260718_n512"
SFT_ROLLOUT="${RUN}/closed_loop_sft_seed20260718_n512"

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"

[[ ! -e "${OUTPUT}" ]] || {
  echo "refusing existing RL output: ${OUTPUT}" >&2
  exit 1
}
[[ ! -e "${ROLLOUT}" ]] || {
  echo "refusing existing rollout: ${ROLLOUT}" >&2
  exit 1
}

"${PYTHON}" scripts/launch_active_catalog_vlm_rl.py \
  "${ADAPTER}" "${RL_DATA}/train.jsonl" "${RL_DATA}/val.jsonl" "${OUTPUT}" \
  --gpu "${GPU}" --seed "${SEED}" --epochs 1 \
  --learning-rate 2.5e-6 --gradient-accumulation 16 \
  --max-length 2048 --action-limit 4 --kl-beta 0.10 \
  --entropy-weight 0.01 --temperature 1.0 --utility-scale 30 \
  --acquire-fraction 0.25 \
  --pairwise-weight 0.1 --pairwise-margin 0.2 \
  --length-normalize-action-score \
  --logging-steps 10 --eval-steps 64 --save-steps 64 \
  --max-train-samples 4096 --max-eval-samples 512 --monitor-interval 5

"${PYTHON}" scripts/launch_active_catalog_closed_loop.py \
  "${MODEL}" "${OUTPUT}/seed${SEED}/final" \
  "${DATA}/closed_loop_v1/states_val_step0.jsonl" \
  "${DATA}/closed_loop_v1/episodes_val.jsonl" \
  "${DATA}/active_catalog_sft_v4/val.jsonl" \
  "${DATA}/active_catalog_sft_v4/val_evaluation_index.jsonl" \
  "${ROLLOUT}" --gpu "${GPU}" --seed "${SEED}" \
  --max-candidates 16 --max-acquisitions 2 --max-new-tokens 64 \
  --bootstrap-repetitions 500 --limit 512 --monitor-interval 5

[[ -s "${SFT_ROLLOUT}/evaluation/traces.jsonl" ]] || {
  echo "missing matched SFT rollout: ${SFT_ROLLOUT}" >&2
  exit 1
}

"${PYTHON}" scripts/compare_active_catalog_closed_loop.py \
  "${RUN}/closed_loop_rl_conservative_lr2p5e6_kl010_bal25_n4096_seed20260718_n512_paired.json" \
  --reference sft --repetitions 2000 --seed "${SEED}" \
  --records "sft=${SFT_ROLLOUT}/evaluation/traces.jsonl" \
  --records "lr2p5_kl010_bal25=${ROLLOUT}/evaluation/traces.jsonl"
