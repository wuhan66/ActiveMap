#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
RUN="${STORE}/runs/sn7_active_catalog"
DATA="${STORE}/processed/sn7_v1/agent/sequential_selector_v1/full"
OUT="${RUN}/direct_vlm_no_tools_fullval_seed20260718"

cd "${PROJECT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"

if [[ -f "${OUT}/process_result.json" ]] && \
   grep -q '"status": "completed"' "${OUT}/process_result.json"; then
  echo "Direct VLM without tools full validation already complete"
  exit 0
fi
if [[ -e "${OUT}" ]]; then
  echo "refusing incomplete existing output: ${OUT}" >&2
  exit 20
fi

"${PYTHON}" scripts/launch_active_catalog_closed_loop.py \
  /home/wh/hf_models/Qwen3-VL-4B-Instruct \
  "${RUN}/qwen3vl4b_weighted_replication_seed2/seed20260718/final" \
  "${DATA}/closed_loop_v1/states_val_step0.jsonl" \
  "${DATA}/episodes_train_val.jsonl" \
  "${DATA}/active_catalog_sft_v4/val.jsonl" \
  "${DATA}/active_catalog_sft_v4/val_evaluation_index.jsonl" \
  "${OUT}" \
  --gpu 3 --seed 20260718 --split val --policy-mode full_action \
  --tool-mode none --belief-mode recurrent \
  --max-candidates 16 --max-acquisitions 2 --max-new-tokens 64 \
  --bootstrap-repetitions 5000 --monitor-interval 5
