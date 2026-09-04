#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
MAIN_ROOT="${MAIN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_corruption_v1/full_seed20260729}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_corruption_uncertainty_v1/full_seed20260729}"
CORRUPTION_SEED="${CORRUPTION_SEED:-20260729}"
SEVERITY="${SEVERITY:?set SEVERITY}"
SEED="${SEED:?set SEED}"
QUANTILE="${QUANTILE:?set QUANTILE}"

case "${SEVERITY}" in 0|4|8|16) ;; *) exit 2 ;; esac
case "${SEED}" in 20260730|20260731|20260801) ;; *) exit 2 ;; esac
case "${QUANTILE}" in 05|15|30) ;; *) exit 2 ;; esac

BUNDLE="${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1"
if [[ "${SEVERITY}" == "0" ]]; then
  STATES="${BUNDLE}/states_val_step0.jsonl"
else
  STATES="${MAIN_ROOT}/severity${SEVERITY}/states_val_step0.jsonl"
fi
EPISODES="${BUNDLE}/episodes_val.jsonl"
UPDATER="${STORAGE_ROOT}/models/frozen_updater/sn7_v4_hierarchical_vector_change_seed20260716/best_quality.pt"
SELECTOR="${STORAGE_ROOT}/runs/selector/sn7_step0_two_stage_v2_seed${SEED}/edit_utility_seed${SEED}/best.pt"
TOOL_BELIEF="${STORAGE_ROOT}/runs/agent/sn7_step0_tool_belief_gated_seed${SEED}/best_promoted.pt"
ADAPTER="${STORAGE_ROOT}/runs/agent/sn7_step0_post_tool_adapter_updated_seed${SEED}/best.pt"
GATE="${STORAGE_ROOT}/runs/sn7_active_catalog/step0_selector_uncertainty_gate_seed${SEED}_q${QUANTILE}_frontier_v1"
JOB_ROOT="${RUN_ROOT}/severity${SEVERITY}/seed${SEED}/q${QUANTILE}"
CLOSED_LOOP="${JOB_ROOT}/closed_loop"
WRITEBACK_INPUT="${JOB_ROOT}/writeback_input.jsonl"
WRITEBACK="${JOB_ROOT}/writeback"

for path in \
  "${STATES}" "${EPISODES}" "${UPDATER}" "${SELECTOR}" \
  "${TOOL_BELIEF}" "${ADAPTER}" "${GATE}/gate.json" "${GATE}/summary.json"; do
  test -f "${path}"
done
mkdir -p "${JOB_ROOT}"
cd "${PROJECT_ROOT}"

if [[ ! -f "${CLOSED_LOOP}/summary.json" ]]; then
  test ! -e "${CLOSED_LOOP}" || {
    echo "incomplete closed-loop output exists: ${CLOSED_LOOP}" >&2
    exit 3
  }
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" scripts/evaluate_active_catalog_closed_loop_baselines.py \
      "${STATES}" "${CLOSED_LOOP}" \
      --episodes "${EPISODES}" \
      --learned-selector "edit_utility=${SELECTOR}" \
      --policy edit_utility \
      --seed "${SEED}" --device cuda:0 --split val \
      --belief-mode recurrent \
      --max-candidates 16 --max-acquisitions 2 \
      --bootstrap-repetitions 0 \
      --limit 6369 --sample-seed 20260729 \
      --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
      --tool-mode selective \
      --tool-belief-checkpoint "${TOOL_BELIEF}" \
      --post-tool-action-adapter "${ADAPTER}" \
      --tool-gate "${GATE}/gate.json" \
      --tool-gate-summary "${GATE}/summary.json" \
      --tool-artifact-root "${CLOSED_LOOP}/artifacts" \
      --tool-out-size 256 \
      >"${JOB_ROOT}/closed_loop.log" 2>&1
fi

if [[ ! -f "${WRITEBACK_INPUT}" ]]; then
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" scripts/convert_active_catalog_closed_loop_for_writeback.py \
      "${CLOSED_LOOP}/edit_utility.jsonl" "${WRITEBACK_INPUT}" \
      --split val --evidence-mode last >"${JOB_ROOT}/convert.log" 2>&1
fi

if [[ ! -f "${WRITEBACK}/summary.json" ]]; then
  test ! -e "${WRITEBACK}" || {
    echo "incomplete writeback output exists: ${WRITEBACK}" >&2
    exit 3
  }
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" scripts/evaluate_agent_map_writeback.py \
      "${UPDATER}" "${EPISODES}" "${WRITEBACK_INPUT}" "${WRITEBACK}" \
      --device cuda:0 --split val --image-size 512 \
      --threshold 0.5 --simplify-tolerance 0.0 \
      --min-delta-component-pixels 0 --delta-margin 0.15 \
      --confidence-floor 0.0 --evidence-fusion confidence_weighted \
      --protocol-name sn7-controller-prior-corruption-uncertainty-v1 \
      --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
      --prior-input-translation-pixels "${SEVERITY}" \
      --corruption-seed "${CORRUPTION_SEED}" \
      >"${JOB_ROOT}/writeback.log" 2>&1
fi

printf '{"status":"complete","severity":%s,"seed":%s,"variant":"uncertainty_q%s","test_assets_read":false}\n' \
  "${SEVERITY}" "${SEED}" "${QUANTILE}" >"${JOB_ROOT}/COMPLETE.json"
