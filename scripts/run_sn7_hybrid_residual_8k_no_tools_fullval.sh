#!/usr/bin/env bash
set -euo pipefail

SEED="${1:?usage: $0 SEED GPU}"
GPU="${2:?usage: $0 SEED GPU}"
PROJECT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
RUN="${STORE}/runs/sn7_active_catalog"
DATA="${STORE}/processed/sn7_v1/agent/sequential_selector_v1/full"
ROOT="${RUN}/hybrid_residual_8k_no_tools_fullval_matrix_20260801"
OUT="${ROOT}/seed${SEED}"

cd "${PROJECT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
mkdir -p "${ROOT}"

if [[ -f "${OUT}/process_result.json" ]] && \
   grep -q '"status": "completed"' "${OUT}/process_result.json"; then
  echo "Hybrid 8k without tools seed${SEED} already complete"
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
  --gpu "${GPU}" --seed "${SEED}" --split val \
  --policy-mode hybrid_residual_ranker \
  --ranker-checkpoint "${RUN}/online_hybrid_residual_8k_seed${SEED}/best.pt" \
  --utility-head "${RUN}/vla_utility_head_shared_last_full/gate.joblib" \
  --utility-head-summary "${RUN}/vla_utility_head_shared_last_full/summary.json" \
  --tool-mode none --belief-mode recurrent \
  --max-candidates 16 --max-acquisitions 2 \
  --bootstrap-repetitions 0 --monitor-interval 5
