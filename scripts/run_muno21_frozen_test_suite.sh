#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
source "${PROJECT_ROOT}/scripts/server_hdpi_env.sh"
export ACTIVEMAP_LAUNCHER_NAME="run_muno21_frozen_test_suite.sh"
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/acquire_training_slot.sh"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-${ACTIVEMAP_AGENT_ENV}/bin/python}"
GIS_PYTHON="${ACTIVEMAP_PYTHON:-${ACTIVEMAP_GIS_ENV}/bin/python}"
GPU="${MUNO21_AGENT_GPU:-${ACTIVEMAP_GPU_IDS%%,*}}"

ROOT="${ACTIVEMAP_PROCESSED_ROOT}/muno21_v2"
AGENT_ROOT="${ROOT}/agent"
UPDATER_MANIFEST="${ROOT}/updater/updater_samples.jsonl"
UPDATER_CHECKPOINT="${MUNO21_UPDATER_CHECKPOINT:-${ACTIVEMAP_RUN_ROOT}/updater/muno21_road_v7_topology_scratch_seed20260731/best.pt}"
EPISODES="${AGENT_ROOT}/episodes_test_v1.jsonl"
STATES="${AGENT_ROOT}/selector_states_test_v1.jsonl"
FLAT_TOOL="${AGENT_ROOT}/tool_belief_test_v1"
SEQUENCE_TOOL="${AGENT_ROOT}/tool_belief_sequence_test_v1"
TOOL_LABELS="${AGENT_ROOT}/sparse_tool_labels_test_v1"
TOOL_BELIEF="${ACTIVEMAP_RUN_ROOT}/agent/tool_belief_anchored_v4_seed20260821/best.pt"
TOOL_EVAL="${ACTIVEMAP_STORAGE_ROOT}/artifacts/frozen_test/tool_belief_sequence_test_v1"
ROLLOUT_ROOT="${ACTIVEMAP_STORAGE_ROOT}/artifacts/frozen_test/agent_three_seed"
WRITEBACK_ROOT="${ACTIVEMAP_STORAGE_ROOT}/artifacts/frozen_test/writeback_three_seed"
OFFICIAL_ROOT="${ACTIVEMAP_STORAGE_ROOT}/artifacts/frozen_test/official_graph_three_seed"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON}" -c \
  'from activemap.frozen_test import assert_frozen_test_access; assert_frozen_test_access()'

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/assert_allowed_gpu.sh"
activemap_assert_allowed_gpu "${GPU}"
gpu_pids="$(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ -n "${gpu_pids//[[:space:]]/}" ]]; then
  echo "Physical GPU ${GPU} has active compute processes: ${gpu_pids}" >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES="${GPU}"

for path in "${UPDATER_MANIFEST}" "${UPDATER_CHECKPOINT}" "${TOOL_BELIEF}"; do
  [[ -s "${path}" ]] || { echo "Missing frozen-test input: ${path}" >&2; exit 4; }
done
for path in \
  "${EPISODES}" "${STATES}" "${FLAT_TOOL}" "${SEQUENCE_TOOL}" \
  "${TOOL_LABELS}" "${TOOL_EVAL}" "${ROLLOUT_ROOT}" "${WRITEBACK_ROOT}"; do
  [[ ! -e "${path}" ]] || { echo "Refusing to reuse frozen-test output: ${path}" >&2; exit 5; }
done
[[ ! -e "${OFFICIAL_ROOT}" ]] || {
  echo "Refusing to reuse frozen-test output: ${OFFICIAL_ROOT}" >&2
  exit 5
}

"${GIS_PYTHON}" -m activemap.cli build-episodes-muno21 \
  "${UPDATER_MANIFEST}" "${EPISODES}" --splits test --frozen-test

"${GIS_PYTHON}" -m activemap.cli build-selector-oracle \
  "${UPDATER_CHECKPOINT}" "${EPISODES}" "${STATES}" \
  --device cuda:0 --image-size 512 --budgets 1.5,3.0,4.5 \
  --splits test --frozen-test
"${GIS_PYTHON}" scripts/audit_selector_states.py "${STATES}" \
  --output "${AGENT_ROOT}/selector_states_test_v1.audit.json"

"${GIS_PYTHON}" scripts/build_muno21_tool_belief_data.py \
  "${EPISODES}" "${STATES}" "${FLAT_TOOL}" \
  --out-size 512 --split test --frozen-test
"${GIS_PYTHON}" scripts/build_tool_belief_sequences.py \
  "${FLAT_TOOL}" "${EPISODES}" "${STATES}" "${SEQUENCE_TOOL}" \
  --splits test --frozen-test

"${GIS_PYTHON}" scripts/evaluate_tool_belief_sequences.py \
  "${SEQUENCE_TOOL}/test.jsonl" "${TOOL_BELIEF}" "${TOOL_EVAL}" \
  --device cpu --split test --frozen-test
"${GIS_PYTHON}" scripts/build_sparse_tool_agent_sft.py \
  "${SEQUENCE_TOOL}/test.jsonl" "${TOOL_EVAL}/details.jsonl" "${TOOL_LABELS}" \
  --frozen-test

MUNO21_AGENT_SPLIT=test \
MUNO21_AGENT_GPU="${GPU}" \
MUNO21_TOOL_SUPERVISION="${TOOL_LABELS}/sft.jsonl" \
MUNO21_AGENT_ROLLOUT_ROOT="${ROLLOUT_ROOT}" \
  bash scripts/evaluate_agent_three_seeds.sh

declare -A methods=(
  [generic_selector]=generic_selector
  [edit_conditioned_selector]=edit_conditioned_selector
  [agent_tool_to_belief]=qwen3_4b_sft_tool_to_belief
)
for seed in 20260821 20260822 20260823; do
  for method in generic_selector edit_conditioned_selector agent_tool_to_belief; do
    rollout_name="${methods[${method}]}"
    "${GIS_PYTHON}" scripts/evaluate_agent_map_writeback.py \
      "${UPDATER_CHECKPOINT}" "${EPISODES}" \
      "${ROLLOUT_ROOT}/seed${seed}/${rollout_name}.jsonl" \
      "${WRITEBACK_ROOT}/seed${seed}/${method}" \
      --device cuda:0 --split test --frozen-test --image-size 512 --threshold 0.5
    MUNO21_EVAL_SPLIT=test MUNO21_EPISODES="${EPISODES}" \
      bash scripts/run_muno21_official_graph_metrics.sh \
      "${WRITEBACK_ROOT}/seed${seed}/${method}/writeback.jsonl" \
      "${OFFICIAL_ROOT}/seed${seed}/${method}"
  done
done

"${PYTHON}" - "${ACTIVEMAP_FROZEN_TEST_LEDGER}" "${ROLLOUT_ROOT}" \
  "${WRITEBACK_ROOT}" "${OFFICIAL_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

ledger, rollouts, writebacks, official = map(Path, sys.argv[1:])
manifest = {
    "schema_version": "activemap-muno21-frozen-suite-v1",
    "ledger_started_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
    "rollout_root": str(rollouts.resolve()),
    "writeback_root": str(writebacks.resolve()),
    "official_graph_root": str(official.resolve()),
    "seeds": [20260821, 20260822, 20260823],
    "budgets": [1.5, 3.0, 4.5],
    "methods": ["generic_selector", "edit_conditioned_selector", "agent_tool_to_belief"],
    "heuristic_methods": [
        "random", "cheapest", "quality_first", "uncertainty", "mapex", "greedy_utility"
    ],
    "heuristic_protocol": "label_free_budget_filling_v1",
    "test_assets_read": True,
}
(rollouts.parent / "suite_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY

echo "[$(date --iso-8601=seconds)] MUNO21 frozen test suite complete"
