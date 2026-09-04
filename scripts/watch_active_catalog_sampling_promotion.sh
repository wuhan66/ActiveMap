#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
RUN_ROOT="${RUN_ROOT:-/home/wh/ActiveMap/runs/sn7_active_catalog}"
POLL_SECONDS="${POLL_SECONDS:-60}"
REPLICATE_SEEDS="${REPLICATE_SEEDS:-20260718,20260719}"
GPU_POOL="${GPU_POOL:-1,3}"
EVAL_GPU="${EVAL_GPU:-1}"
SAMPLING_REPORT="${RUN_ROOT}/seed1_sampling_ablation.json"
AGGREGATE="${RUN_ROOT}/qwen3vl4b_promoted_three_seed/active_catalog_aoi_bootstrap.json"

cd "${PROJECT_ROOT}"
source scripts/server_hdpi_env.sh

while [[ ! -s "${SAMPLING_REPORT}" ]]; do
  echo "$(date -Is) waiting for validation-only sampling decision"
  sleep "${POLL_SECONDS}"
done

decision="$(${ACTIVEMAP_AGENT_ENV}/bin/python -c '
import json, sys
report = json.load(open(sys.argv[1]))
print(report.get("decision", "missing"))
' "${SAMPLING_REPORT}")"
case "${decision}" in
  weighted|unweighted) ;;
  *) echo "sampling comparison is not promotable: ${decision}" >&2; exit 20 ;;
esac

SEEDS="${REPLICATE_SEEDS}" GPU_POOL="${GPU_POOL}" RUN_ROOT="${RUN_ROOT}" \
  bash scripts/run_sn7_active_catalog_qwen.sh promoted_replicates

SEEDS="${REPLICATE_SEEDS}" GPU_SECOND="${EVAL_GPU}" RUN_ROOT="${RUN_ROOT}" \
  bash scripts/run_sn7_active_catalog_qwen.sh evaluate_promoted_replicates

[[ ! -e "${AGGREGATE}" ]] || {
  echo "refusing to overwrite promoted aggregate: ${AGGREGATE}" >&2
  exit 21
}
SEEDS="${REPLICATE_SEEDS}" RUN_ROOT="${RUN_ROOT}" \
  bash scripts/run_sn7_active_catalog_qwen.sh aggregate_promoted_three_seed
RUN_ROOT="${RUN_ROOT}" bash scripts/run_sn7_active_catalog_qwen.sh paper_table_promoted
