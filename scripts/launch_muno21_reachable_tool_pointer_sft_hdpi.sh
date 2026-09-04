#!/usr/bin/env bash
set -euo pipefail

# Independent pointer-action SFT. This never overwrites the historical
# opaque-handle adapter or its frozen validation receipts.
PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
GPU="${GPU:-1}"
SEED="${SEED:-20260842}"
TAG="${TAG:-v3}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-4B}"

STATES="${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl"
PAIR_ROOT="${STORE}/processed/muno21_v2/agent/post_acquisition_tool_pair_v1"
MAIN_TRAIN="${STORE}/processed/muno21_v2/agent/agent_data_v10_balanced_sparse_tools/train/sft_composed.jsonl"
MAIN_VAL="${STORE}/processed/muno21_v2/agent/agent_data_v10_balanced_sparse_tools/val/sft_composed.jsonl"
TOOL_ROOT="${TOOL_ROOT:-${STORE}/runs/agent/reachable_tool_pointer_sft_${TAG}}"
DATA_ROOT="${DATA_ROOT:-${STORE}/processed/muno21_v2/agent/reachable_tool_pointer_support_sft_${TAG}}"
RUN_DIR="${RUN_DIR:-${STORE}/runs/agent/muno21_qwen3_4b_reachable_tool_pointer_sft_${TAG}_seed${SEED}}"

for path in "${PY}" "${MODEL}/config.json" "${STATES}" "${PAIR_ROOT}/train.jsonl" \
  "${PAIR_ROOT}/val.jsonl" "${MAIN_TRAIN}" "${MAIN_VAL}"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
for path in "${TOOL_ROOT}" "${DATA_ROOT}" "${RUN_DIR}"; do
  [[ ! -e "${path}" ]] || { echo "Refusing existing output: ${path}" >&2; exit 3; }
done

mkdir -p "${TOOL_ROOT}" "${DATA_ROOT}/train" "${DATA_ROOT}/val" "${RUN_DIR}/control"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

"${PY}" scripts/build_reachable_tool_sft.py \
  "${STATES}" "${PAIR_ROOT}/train.jsonl" "${TOOL_ROOT}/train.jsonl" \
  --split train --max-pairs-per-task 1
"${PY}" scripts/build_reachable_tool_sft.py \
  "${STATES}" "${PAIR_ROOT}/val.jsonl" "${TOOL_ROOT}/val.jsonl" \
  --split val --max-pairs-per-task 1

"${PY}" scripts/compose_agent_sft.py \
  "${MAIN_TRAIN}" "${TOOL_ROOT}/train.jsonl" "${DATA_ROOT}/train/sft_composed.jsonl" \
  --training --use-tool-repeat 5 --no-tool-ratio 3.0 --seed "${SEED}"
"${PY}" scripts/compose_agent_sft.py \
  "${MAIN_VAL}" "${TOOL_ROOT}/val.jsonl" "${DATA_ROOT}/val/sft_composed.jsonl" \
  --keep-all-tool-sequences --seed "${SEED}"

cat >"${RUN_DIR}/protocol.json" <<EOF
{
  "schema_version": "muno21-reachable-tool-pointer-sft-${TAG}",
  "action_encoding": "selected_evidence_index_v1",
  "base_model": "${MODEL}",
  "main_train": "${MAIN_TRAIN}",
  "main_val": "${MAIN_VAL}",
  "reachable_tool_train": "${TOOL_ROOT}/train.jsonl",
  "reachable_tool_val": "${TOOL_ROOT}/val.jsonl",
  "use_tool_repeat": 5,
  "no_tool_ratio": 3.0,
  "seed": ${SEED},
  "physical_gpu": ${GPU},
  "test_assets_read": false
}
EOF

nohup env CUDA_VISIBLE_DEVICES="${GPU}" \
  PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH}" \
  "${PY}" "${PROJECT}/scripts/train_agent_sft.py" \
  "${MODEL}" "${DATA_ROOT}/train/sft_composed.jsonl" "${RUN_DIR}" \
  --eval-jsonl "${DATA_ROOT}/val/sft_composed.jsonl" \
  --epochs 2 --learning-rate 0.0002 --batch-size 1 \
  --gradient-accumulation 16 --max-length 2048 \
  --logging-steps 5 --eval-steps 100 --save-steps 100 \
  --save-total-limit 8 --lora-rank 16 --lora-alpha 32 --seed "${SEED}" \
  >"${RUN_DIR}/train.log" 2>&1 < /dev/null &
pid=$!
echo "${pid}" >"${RUN_DIR}/train.pid"
sleep 3
kill -0 "${pid}" 2>/dev/null || {
  echo "Pointer-action SFT failed to start; inspect ${RUN_DIR}/train.log" >&2
  exit 4
}
echo "Pointer-action SFT started: PID ${pid}, physical GPU ${GPU}, run ${RUN_DIR}"
