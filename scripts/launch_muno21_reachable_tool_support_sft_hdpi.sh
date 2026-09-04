#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
GPU="${GPU:-7}"
SEED="${SEED:-20260831}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-4B}"

MAIN_TRAIN="${STORE}/processed/muno21_v2/agent/agent_data_v10_balanced_sparse_tools/train/sft_composed.jsonl"
MAIN_VAL="${STORE}/processed/muno21_v2/agent/agent_data_v10_balanced_sparse_tools/val/sft_composed.jsonl"
TOOL_TRAIN="${STORE}/runs/agent/reachable_tool_sft_v1/train.jsonl"
TOOL_VAL="${STORE}/runs/agent/reachable_tool_sft_v1/val.jsonl"
DATA_ROOT="${STORE}/processed/muno21_v2/agent/reachable_tool_support_sft_v1"
RUN_DIR="${STORE}/runs/agent/muno21_qwen3_4b_reachable_tool_support_sft_seed${SEED}"

if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing existing run directory: ${RUN_DIR}" >&2
  exit 1
fi
for path in "${PY}" "${MODEL}/config.json" "${MAIN_TRAIN}" "${MAIN_VAL}" \
  "${TOOL_TRAIN}" "${TOOL_VAL}"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done

for path in "${MAIN_TRAIN}" "${MAIN_VAL}" "${TOOL_TRAIN}" "${TOOL_VAL}"; do
  [[ "$(grep -cve '^$' "${path}")" -gt 0 ]] || {
    echo "Empty JSONL input: ${path}" >&2
    exit 2
  }
done

mkdir -p "${DATA_ROOT}/train" "${DATA_ROOT}/val" "${RUN_DIR}/control"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

"${PY}" scripts/compose_agent_sft.py \
  "${MAIN_TRAIN}" "${TOOL_TRAIN}" "${DATA_ROOT}/train/sft_composed.jsonl" \
  --training --use-tool-repeat 5 --no-tool-ratio 3.0 --seed "${SEED}"
"${PY}" scripts/compose_agent_sft.py \
  "${MAIN_VAL}" "${TOOL_VAL}" "${DATA_ROOT}/val/sft_composed.jsonl" \
  --keep-all-tool-sequences --seed "${SEED}"

cat >"${RUN_DIR}/protocol.json" <<EOF
{
  "schema_version": "muno21-reachable-tool-support-sft-v1",
  "base_model": "${MODEL}",
  "main_train": "${MAIN_TRAIN}",
  "main_val": "${MAIN_VAL}",
  "reachable_tool_train": "${TOOL_TRAIN}",
  "reachable_tool_val": "${TOOL_VAL}",
  "use_tool_repeat": 5,
  "no_tool_ratio": 3.0,
  "seed": ${SEED},
  "physical_gpu": ${GPU},
  "test_assets_read": false
}
EOF

printf 'physical_gpu=%s\npython=%s\nmodel=%s\ntrain=%s\nval=%s\n' \
  "${GPU}" "${PY}" "${MODEL}" "${DATA_ROOT}/train/sft_composed.jsonl" \
  "${DATA_ROOT}/val/sft_composed.jsonl" >"${RUN_DIR}/launch.txt"

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
if ! kill -0 "${pid}" 2>/dev/null; then
  echo "Support SFT failed to start; inspect ${RUN_DIR}/train.log" >&2
  exit 1
fi
echo "Reachable-tool support SFT started: PID ${pid}, physical GPU ${GPU}, run ${RUN_DIR}"
