#!/usr/bin/env bash
set -euo pipefail

SEED="${1:?usage: $0 SEED GPU frozen_prior|identity|forced_tools}"
GPU="${2:?usage: $0 SEED GPU frozen_prior|identity|forced_tools}"
VARIANT="${3:?usage: $0 SEED GPU frozen_prior|identity|forced_tools}"
PROJECT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
RUN="${STORE}/runs/sn7_active_catalog"
DATA="${STORE}/processed/sn7_v1/agent/sequential_selector_v1/full"
ROOT="${RUN}/hybrid_residual_8k_tool_belief_ablation_20260801/${VARIANT}"
OUT="${ROOT}/seed${SEED}"

case "${VARIANT}" in
  frozen_prior)
    TOOL_MODE=selective
    BELIEF_MODE=frozen_prior
    ;;
  identity)
    TOOL_MODE=selective
    BELIEF_MODE=identity
    ;;
  forced_tools)
    TOOL_MODE=forced
    BELIEF_MODE=recurrent
    ;;
  *)
    echo "unknown variant: ${VARIANT}" >&2
    exit 2
    ;;
esac

cd "${PROJECT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
mkdir -p "${ROOT}"
if [[ -f "${OUT}/process_result.json" ]] && \
   grep -q '"status": "completed"' "${OUT}/process_result.json"; then
  echo "${VARIANT} seed${SEED} already complete"
  exit 0
fi
if [[ -e "${OUT}" ]]; then
  echo "refusing incomplete existing output: ${OUT}" >&2
  exit 20
fi

args=(
  /home/wh/hf_models/Qwen3-VL-4B-Instruct
  "${RUN}/qwen3vl4b_weighted_replication_seed2/seed20260718/final"
  "${DATA}/closed_loop_v1/states_val_step0.jsonl"
  "${DATA}/episodes_train_val.jsonl"
  "${DATA}/active_catalog_sft_v4/val.jsonl"
  "${DATA}/active_catalog_sft_v4/val_evaluation_index.jsonl"
  "${OUT}"
  --gpu "${GPU}" --seed "${SEED}" --split val
  --policy-mode hybrid_residual_ranker
  --ranker-checkpoint "${RUN}/online_hybrid_residual_8k_seed${SEED}/best.pt"
  --utility-head "${RUN}/vla_utility_head_shared_last_full/gate.joblib"
  --utility-head-summary "${RUN}/vla_utility_head_shared_last_full/summary.json"
  --tool-mode "${TOOL_MODE}" --belief-mode "${BELIEF_MODE}"
  --tool-belief-checkpoint "${STORE}/runs/agent/sn7_step0_tool_belief_gated_seed20260730/best_promoted.pt"
  --tool-artifact-root "${ROOT}/seed${SEED}_tools"
  --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}"
  --max-candidates 16 --max-acquisitions 2
  --bootstrap-repetitions 0 --monitor-interval 5
)
if [[ "${TOOL_MODE}" == selective ]]; then
  args+=(
    --tool-gate "${RUN}/step0_tool_need_gate_seed20260730_benefit_gate_v1/gate.joblib"
    --tool-gate-summary "${RUN}/step0_tool_need_gate_seed20260730_benefit_gate_v1/summary.json"
  )
fi
"${PYTHON}" scripts/launch_active_catalog_closed_loop.py "${args[@]}"
