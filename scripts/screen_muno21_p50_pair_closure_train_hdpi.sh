#!/usr/bin/env bash
set -euo pipefail

# Train-only calibration for the temporal-pair closure policy. This script
# never reads validation/test assets and never writes map-update predictions.
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-4B}"
ROOT="${ROOT:-${STORE}/runs/paper_evidence/muno21_p50_pair_closure_train_screen_v3}"
STATES="${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl"
CHECKPOINT="${STORE}/runs/selector/muno21_p50_promotion_v1/p50_seed12/best.pt"
MARGINS=(${PAIR_CLOSURE_MARGINS:-0.00 0.25 0.50 1.00 2.00})

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${ROOT}/logs"
exec 9>"${ROOT}/.lock"
flock -n 9 || exit 0

for path in "${PYTHON}" "${MODEL}/config.json" "${STATES}" "${CHECKPOINT}"; do
  [[ -s "${path}" ]] || { echo "missing input: ${path}" >&2; exit 2; }
done

if [[ ! -e "${ROOT}/protocol.json" ]]; then
  cat >"${ROOT}/protocol.json" <<'EOF'
{
  "schema_version": "muno21-p50-temporal-pair-closure-train-screen-v1",
  "split": "train",
  "test_assets_read": false,
  "writeback_executed": false,
  "margins": [0.0, 0.25, 0.5, 1.0, 2.0],
  "selection_rule": "choose a margin on train-only proxy safety/quality; validation is evaluated exactly once afterwards"
}
EOF
  sha256sum "${STATES}" "${CHECKPOINT}" >"${ROOT}/input_sha256.txt"
fi

for margin in "${MARGINS[@]}"; do
  tag="m${margin/./p}"
  output="${ROOT}/${tag}"
  [[ ! -e "${output}" ]] || { echo "refusing existing output: ${output}" >&2; exit 3; }
  "${PYTHON}" scripts/evaluate_agent_rollouts.py \
    "${MODEL}" "${STATES}" "${output}" \
    --checkpoint "${CHECKPOINT}" --split train --budgets 1.5,3.0,4.5 \
    --device cpu --selector-device cpu --seed 20260812 \
    --methods edit_conditioned_pair_closure \
    --temporal-pair-closure-margin "${margin}" \
    >"${ROOT}/logs/${tag}.log" 2>&1
done

"${PYTHON}" scripts/summarize_muno21_p50_pair_closure_train.py "${ROOT}"
touch "${ROOT}/COMPLETE.json"
