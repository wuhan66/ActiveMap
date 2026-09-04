#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
GPU="${MUNO21_AGENT_GPU:-4}"
POLL_SECONDS="${POLL_SECONDS:-60}"

cd "$PROJECT_ROOT"
# shellcheck source=/dev/null
source scripts/server_hdpi_env.sh
# shellcheck source=/dev/null
source scripts/assert_allowed_gpu.sh
activemap_assert_allowed_gpu "$GPU"

for seed in 20260821 20260822 20260823; do
  promotion="${STORAGE_ROOT}/runs/agent/muno21_qwen3_4b_sparse_tool_sft_seed${seed}/evaluation/selection/promoted_adapter.json"
  while [[ ! -s "$promotion" ]]; do
    echo "$(date -Is) waiting for MUNO21 seed ${seed} promotion"
    sleep "$POLL_SECONDS"
  done
done

while [[ -n "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  echo "$(date -Is) waiting for GPU ${GPU}"
  sleep "$POLL_SECONDS"
done

export MUNO21_AGENT_GPU="$GPU"
export MUNO21_MAX_TOOL_CALLS="${MUNO21_MAX_TOOL_CALLS:-2}"
bash scripts/evaluate_agent_three_seeds.sh

OUTPUT_ROOT="${MUNO21_AGENT_ROLLOUT_ROOT:-${STORAGE_ROOT}/artifacts/paper_rollouts/agent_three_seed_val}"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-${ACTIVEMAP_AGENT_ENV}/bin/python}"
PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "$PYTHON" scripts/aggregate_agent_three_seeds.py \
  "$OUTPUT_ROOT" "$OUTPUT_ROOT/three_seed_bootstrap.json"
PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "$PYTHON" scripts/assess_agent_three_seed_promotion.py \
  "$OUTPUT_ROOT/three_seed_bootstrap.json" "$OUTPUT_ROOT/three_seed_promotion.json"
