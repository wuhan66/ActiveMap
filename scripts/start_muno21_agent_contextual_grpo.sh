#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
if [[ -f "${PROJECT_ROOT}/scripts/server_hdpi_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/scripts/server_hdpi_env.sh"
fi
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/acquire_training_slot.sh"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-${ACTIVEMAP_AGENT_ENV:-${STORAGE_ROOT}/envs/activemap-agent}/bin/python}"
AGENT_SITE_PACKAGES="${ACTIVEMAP_AGENT_SITE_PACKAGES:-}"
GPU="${MUNO21_AGENT_GPU:-${ACTIVEMAP_GPU_IDS%%,*}}"
SFT_RUN="${MUNO21_SFT_RUN:-${STORAGE_ROOT}/runs/agent/muno21_qwen3_4b_sparse_tool_sft_seed20260821}"
PROMOTION="${MUNO21_SFT_PROMOTION:-${SFT_RUN}/evaluation/selection/promoted_adapter.json}"
CHECKPOINT="${MUNO21_SFT_CHECKPOINT:-}"
DATA="${MUNO21_RL_DATA:-${STORAGE_ROOT}/processed/muno21_v2/agent/agent_rl_states_v2_grounded_tools}"
RUN_DIR="${MUNO21_RL_RUN:-${STORAGE_ROOT}/runs/agent/muno21_qwen3_4b_contextual_grpo_seed20260821}"
READINESS="${MUNO21_RL_READINESS:-${SFT_RUN}/evaluation/selection/rl_readiness_decision.json}"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/assert_allowed_gpu.sh"
activemap_assert_allowed_gpu "${GPU}"
cd "$PROJECT_ROOT"
PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" scripts/assert_training_ready.py \
  configs/experiments/paper_registry.yaml "$STORAGE_ROOT"

[[ -s "$PROMOTION" ]] || {
  echo "Missing validation-selected SFT promotion: ${PROMOTION}" >&2
  exit 2
}
PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "$PYTHON" scripts/verify_sparse_tool_sft_adapter.py "$PROMOTION"
promotion_ready="$($PYTHON -c 'import json,sys; d=json.load(open(sys.argv[1])); print("1" if d.get("approved") and d.get("test_assets_read") is False else "0")' "$PROMOTION")"
[[ "$promotion_ready" == "1" ]] || {
  echo "SFT promotion is not approved or test-isolated" >&2
  exit 2
}
if [[ -z "$CHECKPOINT" ]]; then
  CHECKPOINT="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["adapter_path"])' "$PROMOTION")"
fi
[[ -s "${CHECKPOINT}/adapter_config.json" ]] || {
  echo "Missing promoted SFT adapter: ${CHECKPOINT}" >&2
  exit 2
}
for path in "${DATA}/train.jsonl" "${DATA}/val.jsonl" "$READINESS"; do
  [[ -s "$path" ]] || { echo "Missing required RL input: $path" >&2; exit 2; }
done
ready="$($PYTHON -c 'import json,sys; d=json.load(open(sys.argv[1])); print("1" if d.get("ready_for_single_seed_rl") else "0")' "$READINESS")"
[[ "$ready" == "1" ]] || {
  echo "Frozen SFT closed-loop safety gates do not permit RL" >&2
  exit 3
}
selected="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1])).get("selected_checkpoint") or "")' "$READINESS")"
promoted="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1])).get("selected_checkpoint") or "")' "$PROMOTION")"
[[ -n "$selected" && "$selected" == "$promoted" && "$(basename "$CHECKPOINT")" == "$selected" ]] || {
  echo "Readiness decision selected ${selected:-none}, but checkpoint is ${CHECKPOINT}" >&2
  exit 3
}
if [[ -n "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; then
  echo "Physical GPU ${GPU} is occupied; refusing to start a second experiment" >&2
  exit 4
fi
[[ ! -e "${RUN_DIR}/train.pid" ]] || {
  echo "Run already has a PID record: ${RUN_DIR}" >&2
  exit 5
}

mkdir -p "$RUN_DIR"
cp "$READINESS" "${RUN_DIR}/rl_readiness_decision.json"
cp "$PROMOTION" "${RUN_DIR}/sft_promoted_adapter.json"
cp "${DATA}/train.jsonl.summary.json" "${RUN_DIR}/train_data_summary.json"
cp "${DATA}/val.jsonl.summary.json" "${RUN_DIR}/val_data_summary.json"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="${PROJECT_ROOT}/src${AGENT_SITE_PACKAGES:+:${AGENT_SITE_PACKAGES}}${PYTHONPATH:+:${PYTHONPATH}}"
nohup "$PYTHON" scripts/train_agent_grpo.py \
  "$CHECKPOINT" "${DATA}/train.jsonl" "$RUN_DIR" \
  --eval-jsonl "${DATA}/val.jsonl" \
  --epochs "${MUNO21_RL_EPOCHS:-1}" \
  --learning-rate "${MUNO21_RL_LR:-5e-7}" \
  --batch-size "${MUNO21_RL_BATCH_SIZE:-4}" \
  --gradient-accumulation "${MUNO21_RL_GRAD_ACCUM:-4}" \
  --num-generations "${MUNO21_RL_NUM_GENERATIONS:-4}" \
  --contextual-ablation-only \
  --eval-steps "${MUNO21_RL_EVAL_STEPS:-50}" \
  --save-steps "${MUNO21_RL_SAVE_STEPS:-50}" \
  > "${RUN_DIR}/train.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "${RUN_DIR}/train.pid"
printf '%s\n' "$GPU" > "${RUN_DIR}/physical_gpu.txt"
echo "Started contextual constrained GRPO PID ${pid} on physical GPU ${GPU}"
