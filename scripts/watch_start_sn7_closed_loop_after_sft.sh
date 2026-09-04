#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
RUN_ROOT="${RUN_ROOT:-/home/wh/ActiveMap/runs/sn7_active_catalog}"
GPU="${SN7_CLOSED_LOOP_GPU:-6}"
PRIMARY_SEED="${ACTIVE_CATALOG_PRIMARY_SEED:-20260717}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MANIFEST="${RUN_ROOT}/qwen3vl4b_promoted_three_seed/promotion_manifest.json"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/ActiveMap/envs/activemap-agent/bin/python}"
EPISODES="${SN7_EPISODES:-/home/wh/ActiveMap/processed/sn7_v1/agent/sequential_selector_v1/full/episodes_train_val.jsonl}"
ASSET_ROOT="${SN7_ASSET_ROOT:-/home/wh/ActiveMap/datasets/sn7}"
ASSET_ROOT_MAP="${SN7_ASSET_ROOT_MAP:-/mnt/mydisk/wh/ActiveMap/datasets/sn7=${ASSET_ROOT}}"

cd "$PROJECT_ROOT"
# shellcheck source=/dev/null
source scripts/server_hdpi_env.sh
# shellcheck source=/dev/null
source scripts/assert_allowed_gpu.sh
activemap_assert_allowed_gpu "$GPU"

while [[ ! -s "$MANIFEST" ]]; do
  echo "$(date -Is) waiting for SN7 promoted three-seed manifest"
  sleep "$POLL_SECONDS"
done
decision="$($PYTHON -c '
import json,sys
r=json.load(open(sys.argv[1]))
assert r.get("test_assets_read") is False
print(r["sampling_decision"])
' "$MANIFEST")"
case "$decision" in
  weighted) family="qwen3vl4b_seed1_eval500" ;;
  unweighted) family="qwen3vl4b_unweighted_seed1" ;;
  *) echo "invalid promoted sampling decision: $decision" >&2; exit 2 ;;
esac
adapter="${RUN_ROOT}/${family}/seed${PRIMARY_SEED}/final"
[[ -s "${adapter}/adapter_config.json" ]] || { echo "missing promoted primary adapter" >&2; exit 3; }

while [[ -n "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  echo "$(date -Is) waiting for GPU ${GPU}"
  sleep "$POLL_SECONDS"
done

asset_preflight="${RUN_ROOT}/closed_loop_asset_preflight.json"
if [[ ! -s "$asset_preflight" ]]; then
  "$PYTHON" scripts/audit_episode_assets.py "$EPISODES" --split val \
    --asset-root-map "$ASSET_ROOT_MAP" --output "$asset_preflight"
fi

run_stage() {
  local artifact="$1"
  local stage="$2"
  shift 2
  if [[ ! -s "$artifact" ]]; then
    RUN_ROOT="$RUN_ROOT" GPU_SECOND="$GPU" CLOSED_LOOP_ADAPTER="$adapter" \
      WRITEBACK_ASSET_ROOT_MAP="$ASSET_ROOT_MAP" TOOL_ASSET_ROOT_MAP="$ASSET_ROOT_MAP" \
      bash scripts/run_sn7_active_catalog_qwen.sh "$stage" "$@"
  fi
}

run_stage /home/wh/ActiveMap/processed/sn7_v1/agent/sequential_selector_v1/full/closed_loop_v1/summary.json prepare_closed_loop_bundle
run_stage "${RUN_ROOT}/closed_loop_baselines/summary.json" closed_loop_baselines
run_stage "${RUN_ROOT}/closed_loop_smoke/process_result.json" closed_loop_smoke
run_stage "${RUN_ROOT}/closed_loop_seed1/evaluation/traces.jsonl" closed_loop_seed1
run_stage "${RUN_ROOT}/closed_loop_seed1/paired_aoi_comparison.json" compare_closed_loop_seed1
run_stage "${RUN_ROOT}/closed_loop_writeback_inputs/qwen.jsonl" prepare_closed_loop_writebacks
run_stage "${RUN_ROOT}/closed_loop_writeback_smoke/evaluation/summary.json" closed_loop_writeback_smoke
run_stage "${RUN_ROOT}/closed_loop_writeback_qwen/evaluation/summary.json" closed_loop_writeback_qwen
run_stage "${RUN_ROOT}/closed_loop_writeback_${WRITEBACK_BASELINE_POLICY:-uncertainty_gate}/evaluation/summary.json" closed_loop_writeback_baseline
run_stage "${RUN_ROOT}/closed_loop_writeback_qwen/paired_${WRITEBACK_BASELINE_POLICY:-uncertainty_gate}_aoi.json" compare_closed_loop_writeback
run_stage "${RUN_ROOT}/closed_loop_writeback_qwen/promotion_${WRITEBACK_BASELINE_POLICY:-uncertainty_gate}.json" assess_closed_loop_promotion
run_stage "${RUN_ROOT}/closed_loop_writeback_qwen/figures/metadata.json" render_closed_loop_examples
