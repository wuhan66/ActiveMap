#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${ACTIVEMAP_PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_AGENT_ENV:-${STORAGE_ROOT}/envs/activemap-agent}/bin/python"
DATA_ROOT="${STORAGE_ROOT}/processed/muno21_v2/agent/post_acquisition_tool_pair_v1"
RUN_ROOT="${STORAGE_ROOT}/runs/agent"
SEED="${MUNO21_BELIEF_SEED:-20260821}"
UNGATED_GPU="${MUNO21_UNGATED_GPU:-1}"
GATED_GPU="${MUNO21_GATED_GPU:-4}"
UNGATED="${RUN_ROOT}/muno21_post_acquisition_ungated_seed${SEED}"
GATED="${RUN_ROOT}/muno21_post_acquisition_reliability_seed${SEED}"
AUDIT="${DATA_ROOT}/feature_audit.json"
LOG_ROOT="${STORAGE_ROOT}/logs/muno21_reliability_belief"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# shellcheck source=/dev/null
source scripts/server_hdpi_env.sh
# shellcheck source=/dev/null
source scripts/assert_allowed_gpu.sh
activemap_assert_allowed_gpu "${UNGATED_GPU}"
activemap_assert_allowed_gpu "${GATED_GPU}"
[[ "${UNGATED_GPU}" != "${GATED_GPU}" ]] || { echo "distinct GPUs required" >&2; exit 2; }

for path in "${DATA_ROOT}/train.jsonl" "${DATA_ROOT}/val.jsonl" "${AUDIT}"; do
  [[ -s "${path}" ]] || { echo "missing input: ${path}" >&2; exit 3; }
done
for gpu in "${UNGATED_GPU}" "${GATED_GPU}"; do
  [[ -z "$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)" ]] || {
    echo "GPU ${gpu} is occupied" >&2
    exit 4
  }
done
for output in "${UNGATED}" "${GATED}"; do
  [[ ! -e "${output}" ]] || { echo "refusing existing run: ${output}" >&2; exit 5; }
done
mkdir -p "${LOG_ROOT}"

common=(
  "${DATA_ROOT}/train.jsonl" "${DATA_ROOT}/val.jsonl"
  --device cuda:0 --epochs 200 --batch-size 128 --learning-rate 3e-4
  --hidden-dim 128 --dropout 0.1 --patience 20
  --safety-margin 0.02 --min-quality-delta 0.01 --min-tool-delta 0.01
  --false-edit-weight 2.0 --tool-contrastive-weight 0.2
  --tool-contrastive-margin 0.05 --seed "${SEED}"
)

CUDA_VISIBLE_DEVICES="${UNGATED_GPU}" "${PYTHON}" -u \
  scripts/train_post_acquisition_tool_belief.py \
  "${common[@]:0:2}" "${UNGATED}" "${common[@]:2}" \
  >"${LOG_ROOT}/ungated_seed${SEED}.log" 2>&1 &
ungated_pid=$!
CUDA_VISIBLE_DEVICES="${GATED_GPU}" "${PYTHON}" -u \
  scripts/train_post_acquisition_tool_belief.py \
  "${common[@]:0:2}" "${GATED}" "${common[@]:2}" \
  --reliability-gate --gate-bias -1.5 \
  >"${LOG_ROOT}/gated_seed${SEED}.log" 2>&1 &
gated_pid=$!

failed=0
wait "${ungated_pid}" || failed=1
wait "${gated_pid}" || failed=1
(( failed == 0 )) || exit 6
echo "paired reliability training complete: seed=${SEED}"
