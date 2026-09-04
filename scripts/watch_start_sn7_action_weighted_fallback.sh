#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog}"
PYTHON="${ACTIVEMAP_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
REPORT="${RUN_ROOT}/seed1_sampling_ablation_step1000_sentinel.json"
GPU="${ACTION_WEIGHTED_GPU:-1}"
SEED="${SEED:-20260717}"
POLL_SECONDS="${POLL_SECONDS:-60}"

cd "$PROJECT_ROOT"
# shellcheck source=/dev/null
source scripts/server_hdpi_env.sh
# shellcheck source=/dev/null
source scripts/assert_allowed_gpu.sh
activemap_assert_allowed_gpu "$GPU"

while [[ ! -s "$REPORT" ]]; do
  echo "$(date -Is) waiting for step-1000 sentinel report"
  sleep "$POLL_SECONDS"
done
needs_fallback="$($PYTHON -c '
import json,sys
r=json.load(open(sys.argv[1]))
assert r.get("diagnostic_only") is True and r.get("test_assets_read") is False
w=r["weighted_metrics"]["predicted_call_rate"]
u=r["unweighted_metrics"]["predicted_call_rate"]
print(int(w == 0 and u == 0))
' "$REPORT")"
if [[ "$needs_fallback" != "1" ]]; then
  echo "$(date -Is) at least one baseline learned ACQUIRE; action-weighted fallback not launched"
  exit 0
fi

for family in qwen3vl4b_seed1_eval500 qwen3vl4b_unweighted_seed1; do
  state="${RUN_ROOT}/${family}/seed${SEED}/run_state.json"
  while [[ ! -s "$state" ]] || ! "$PYTHON" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("status")=="completed" else 1)' "$state"; do
    echo "$(date -Is) waiting for baseline completion: ${family}"
    sleep "$POLL_SECONDS"
  done
done
while [[ -n "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  echo "$(date -Is) waiting for GPU ${GPU}"
  sleep "$POLL_SECONDS"
done

fallback="${RUN_ROOT}/qwen3vl4b_action_weighted_seed1/seed${SEED}"
[[ ! -e "$fallback" ]] || { echo "refusing existing fallback run: ${fallback}" >&2; exit 4; }
echo "$(date -Is) launching action-weighted fallback on GPU ${GPU}"
if [[ ! -s "${RUN_ROOT}/action_weighted_smoke/seed${SEED}/process_result.json" ]]; then
  GPU_TRAIN="$GPU" RUN_ROOT="$RUN_ROOT" \
    bash scripts/run_sn7_active_catalog_qwen.sh action_weighted_smoke
fi
GPU_TRAIN="$GPU" RUN_ROOT="$RUN_ROOT" \
  bash scripts/run_sn7_active_catalog_qwen.sh action_weighted_seed1
