#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 TRAIN_PID" >&2
  exit 2
fi

TRAIN_PID="$1"
ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
REPO="${ACTIVEMAP_REPOSITORY:-/home/wh/projects/activemap-v1}"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-$ROOT/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
RUN="${GATE_RUN:-$ROOT/runs/sn7_active_catalog/qwen3vl4b_gate_only_t10_seed1/seed20260720}"
RANKER="${RANKER_CHECKPOINT:-$ROOT/runs/sn7_active_catalog/candidate_ranker_v4/seed20260720/best.pt}"
GATE_DATA="${GATE_DATA:-$ROOT/processed/sn7_v1/agent/sequential_selector_v1/full/active_catalog_gate_sft_v1}"
INDEX="${EVALUATION_INDEX:-$ROOT/processed/sn7_v1/agent/sequential_selector_v1/full/active_catalog_sft_v4/val_evaluation_index.jsonl}"
OUTPUT="${GATE_EVALUATION_OUTPUT:-$RUN/active_catalog_gate_ranker_val}"
GPU="${GATE_EVALUATION_GPU:-2}"

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 60
done

if [[ ! -s "$RUN/final/adapter_config.json" ]] || \
   [[ ! -s "$RUN/final/adapter_model.safetensors" ]] || \
   [[ ! -s "$RUN/train_metrics.json" ]]; then
  echo "gate training ended without a complete final adapter" >&2
  exit 1
fi
if [[ -e "$OUTPUT" ]]; then
  echo "refusing to overwrite $OUTPUT" >&2
  exit 1
fi

cd "$REPO"
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES="$GPU"
exec "$PYTHON" scripts/evaluate_active_catalog_gate_vlm.py \
  "$MODEL" "$RUN/final" "$RANKER" "$GATE_DATA/val.jsonl" "$INDEX" "$OUTPUT" \
  --device cuda --seed 20260720 --bootstrap-repetitions 2000
