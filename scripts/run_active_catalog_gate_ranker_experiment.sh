#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 RUN_NAME SEED ACQUIRE_TARGET GPU" >&2
  exit 2
fi

RUN_NAME="$1"
SEED="$2"
ACQUIRE_TARGET="$3"
GPU="$4"

ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
REPO="${ACTIVEMAP_REPOSITORY:-/home/wh/projects/activemap-v1}"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-$ROOT/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
DATA="${GATE_DATA:-$ROOT/processed/sn7_v1/agent/sequential_selector_v1/full/active_catalog_gate_sft_v1}"
INDEX="${EVALUATION_INDEX:-$ROOT/processed/sn7_v1/agent/sequential_selector_v1/full/active_catalog_sft_v4/val_evaluation_index.jsonl}"
RANKER="${RANKER_CHECKPOINT:-$ROOT/runs/sn7_active_catalog/candidate_ranker_v4/seed20260720/best.pt}"
INIT_ADAPTER="${INIT_ADAPTER:-$ROOT/runs/sn7_active_catalog/qwen3vl4b_weighted_replication_seed2/seed20260718/final}"
RUN="$ROOT/runs/sn7_active_catalog/$RUN_NAME/seed$SEED"
OUTPUT="$RUN/active_catalog_gate_ranker_val"
BATCH_SIZE="${GATE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION="${GATE_GRADIENT_ACCUMULATION:-16}"
DATALOADER_WORKERS="${GATE_DATALOADER_WORKERS:-0}"
DATALOADER_PREFETCH="${GATE_DATALOADER_PREFETCH:-2}"
ATTN_IMPLEMENTATION="${GATE_ATTN_IMPLEMENTATION:-sdpa}"

for path in \
  "$PYTHON" "$DATA/train.jsonl" "$DATA/val.jsonl" "$INDEX" \
  "$RANKER" "$INIT_ADAPTER/adapter_config.json"; do
  [[ -s "$path" ]] || { echo "missing required file: $path" >&2; exit 3; }
done
[[ ! -e "$RUN" ]] || { echo "refusing existing run: $RUN" >&2; exit 4; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be an integer" >&2; exit 2; }
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "GPU must be an integer" >&2; exit 2; }
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "invalid batch size" >&2; exit 2; }
[[ "$GRADIENT_ACCUMULATION" =~ ^[1-9][0-9]*$ ]] || {
  echo "invalid gradient accumulation" >&2
  exit 2
}
[[ "$DATALOADER_WORKERS" =~ ^[0-9]+$ ]] || { echo "invalid workers" >&2; exit 2; }
(( BATCH_SIZE * GRADIENT_ACCUMULATION == 16 )) || {
  echo "effective batch must remain frozen at 16" >&2
  exit 2
}

active_pids="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "${active_pids//[[:space:]]/}" ]] || {
  echo "GPU $GPU is occupied by: $active_pids" >&2
  exit 5
}

cd "$REPO"
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false

dataloader_args=(
  --dataloader-num-workers "$DATALOADER_WORKERS"
  --dataloader-prefetch-factor "$DATALOADER_PREFETCH"
  --attn-implementation "$ATTN_IMPLEMENTATION"
)
if [[ "$DATALOADER_WORKERS" -gt 0 ]]; then
  dataloader_args+=(--dataloader-persistent-workers)
fi

"$PYTHON" scripts/train_semantic_vlm_sft.py \
  "$MODEL" "$DATA/train.jsonl" "$RUN" \
  --eval-jsonl "$DATA/val.jsonl" \
  --epochs 1.0 \
  --learning-rate 0.00005 \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation "$GRADIENT_ACCUMULATION" \
  --max-length 2048 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --seed "$SEED" \
  --logging-steps 5 \
  --eval-steps 10000 \
  --save-steps 500 \
  --save-total-limit 2 \
  --early-stopping-patience 0 \
  "${dataloader_args[@]}" \
  --acquire-sampling-target "$ACQUIRE_TARGET" \
  --init-adapter "$INIT_ADAPTER"

[[ -s "$RUN/final/adapter_config.json" ]] || {
  echo "training ended without a complete final adapter" >&2
  exit 6
}

"$PYTHON" scripts/evaluate_active_catalog_gate_vlm.py \
  "$MODEL" "$RUN/final" "$RANKER" "$DATA/val.jsonl" "$INDEX" "$OUTPUT" \
  --device cuda --seed "$SEED" --bootstrap-repetitions 2000
