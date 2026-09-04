#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_morphology_v1}"
MORPHOLOGY="${MORPHOLOGY:?set MORPHOLOGY}"
RADIUS="${RADIUS:?set RADIUS}"
SEED="${SEED:?set SEED}"
VARIANT="${VARIANT:?set VARIANT}"

case "${MORPHOLOGY}" in erode|dilate) ;; *) exit 2 ;; esac
[[ "${RADIUS}" =~ ^[1-9][0-9]*$ ]] || exit 2
case "${SEED}" in 20260730|20260731|20260801) ;; *) exit 2 ;; esac
case "${VARIANT}" in notool|forced|benefit) ;; *) exit 2 ;; esac

ROOT="${RUN_ROOT}/${MORPHOLOGY}${RADIUS}"
STATES="${ROOT}/states_val_step0.jsonl"
FROZEN_STATES="${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/states_val_step0.jsonl"
PARITY_AUDIT="${ROOT}/PARITY_AUDIT.json"
EPISODES="${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/episodes_val.jsonl"
UPDATER="${STORAGE_ROOT}/models/frozen_updater/sn7_v4_hierarchical_vector_change_seed20260716/best_quality.pt"
SELECTOR="${STORAGE_ROOT}/runs/selector/sn7_step0_two_stage_v2_seed${SEED}/edit_utility_seed${SEED}/best.pt"
TOOL_BELIEF="${STORAGE_ROOT}/runs/agent/sn7_step0_tool_belief_gated_seed${SEED}/best_promoted.pt"
ADAPTER="${STORAGE_ROOT}/runs/agent/sn7_step0_post_tool_adapter_updated_seed${SEED}/best.pt"
GATE="${STORAGE_ROOT}/runs/sn7_active_catalog/step0_tool_need_gate_seed${SEED}_benefit_gate_v1"
JOB_ROOT="${ROOT}/seed${SEED}/${VARIANT}"
CLOSED_LOOP="${JOB_ROOT}/closed_loop"
WRITEBACK_INPUT="${JOB_ROOT}/writeback_input.jsonl"
WRITEBACK="${JOB_ROOT}/writeback"

for path in "${STATES}" "${EPISODES}" "${UPDATER}" "${SELECTOR}"; do
  [[ -s "${path}" ]] || { echo "missing input: ${path}" >&2; exit 3; }
done
mkdir -p "${JOB_ROOT}"
exec 9>"${JOB_ROOT}/.job.lock"
flock 9
[[ ! -s "${JOB_ROOT}/COMPLETE.json" || ! -s "${JOB_ROOT}/AUDIT.json" ]] || exit 0
FAILURE_MARKER="${JOB_ROOT}/FAILED.json"
rm -f "${FAILURE_MARKER}"
record_failure() {
  local rc=$?
  trap - EXIT
  if ((rc != 0)); then
    local temporary="${FAILURE_MARKER}.tmp.$$"
    printf \
      '{"status":"failed","stage":"controller_writeback","morphology":"%s","radius":%s,"seed":%s,"variant":"%s","exit_code":%d,"test_assets_read":false}\n' \
      "${MORPHOLOGY}" "${RADIUS}" "${SEED}" "${VARIANT}" "${rc}" >"${temporary}"
    mv -f "${temporary}" "${FAILURE_MARKER}"
  fi
  exit "${rc}"
}
trap record_failure EXIT
cd "${PROJECT_ROOT}"

if [[ ! -s "${PARITY_AUDIT}" ]]; then
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" scripts/audit_sn7_frozen_anchor_parity.py \
      "${FROZEN_STATES}" "${STATES}" "${PARITY_AUDIT}"
fi

if [[ ! -f "${CLOSED_LOOP}/summary.json" ]]; then
  [[ ! -e "${CLOSED_LOOP}" ]] || exit 4
  command=(
    "${PYTHON}" scripts/evaluate_active_catalog_closed_loop_baselines.py
    "${STATES}" "${CLOSED_LOOP}"
    --episodes "${EPISODES}"
    --learned-selector "edit_utility=${SELECTOR}"
    --policy edit_utility --seed "${SEED}" --device cuda:0 --split val
    --belief-mode recurrent --max-candidates 16 --max-acquisitions 2
    --bootstrap-repetitions 0 --limit 6369 --sample-seed 20260729
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}"
  )
  if [[ "${VARIANT}" == notool ]]; then
    command+=(--tool-mode none)
  else
    for path in "${TOOL_BELIEF}" "${ADAPTER}"; do
      [[ -s "${path}" ]] || exit 3
    done
    command+=(
      --tool-belief-checkpoint "${TOOL_BELIEF}"
      --post-tool-action-adapter "${ADAPTER}"
      --tool-artifact-root "${CLOSED_LOOP}/artifacts" --tool-out-size 256
    )
    if [[ "${VARIANT}" == forced ]]; then
      command+=(--tool-mode forced)
    else
      command+=(--tool-mode selective --tool-gate "${GATE}/gate.joblib"
        --tool-gate-summary "${GATE}/summary.json")
    fi
  fi
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${command[@]}" >"${JOB_ROOT}/closed_loop.log" 2>&1
fi

if [[ ! -f "${WRITEBACK_INPUT}" ]]; then
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" scripts/convert_active_catalog_closed_loop_for_writeback.py \
      "${CLOSED_LOOP}/edit_utility.jsonl" "${WRITEBACK_INPUT}" \
      --split val --evidence-mode last >"${JOB_ROOT}/convert.log" 2>&1
fi

if [[ ! -f "${WRITEBACK}/summary.json" ]]; then
  [[ ! -e "${WRITEBACK}" ]] || exit 4
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" scripts/evaluate_agent_map_writeback.py \
      "${UPDATER}" "${EPISODES}" "${WRITEBACK_INPUT}" "${WRITEBACK}" \
      --device cuda:0 --split val --image-size 512 \
      --threshold 0.5 --simplify-tolerance 0.0 \
      --min-delta-component-pixels 0 --delta-margin 0.15 \
      --confidence-floor 0.0 --evidence-fusion confidence_weighted \
      --protocol-name sn7-controller-prior-morphology-v1 \
      --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
      --prior-input-morphology "${MORPHOLOGY}" \
      --prior-input-morphology-pixels "${RADIUS}" \
      >"${JOB_ROOT}/writeback.log" 2>&1
fi

if [[ ! -s "${JOB_ROOT}/AUDIT.json" ]]; then
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" scripts/audit_sn7_controller_writeback_job.py \
      "${JOB_ROOT}" "${JOB_ROOT}/AUDIT.json" \
      --protocol-name sn7-controller-prior-morphology-v1 \
      --morphology "${MORPHOLOGY}" --morphology-pixels "${RADIUS}" \
      >"${JOB_ROOT}/audit.log" 2>&1
fi

temporary="${JOB_ROOT}/COMPLETE.json.tmp.$$"
printf \
  '{"status":"complete","morphology":"%s","radius":%s,"seed":%s,"variant":"%s","test_assets_read":false}\n' \
  "${MORPHOLOGY}" "${RADIUS}" "${SEED}" "${VARIANT}" >"${temporary}"
mv -f "${temporary}" "${JOB_ROOT}/COMPLETE.json"
trap - EXIT
