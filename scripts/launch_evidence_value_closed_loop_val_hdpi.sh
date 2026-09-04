#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/wh/projects/activemap-v1}"
ROOT="${ROOT:-/home/wh/ActiveMap/runs/evidence_value_closed_loop_val_20260725}"
PYTHON="${PYTHON:-/home/wh/ActiveMap/envs/activemap-agent/bin/python}"
STATES="${STATES:-/home/wh/ActiveMap/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/states_val_step0.jsonl}"
EPISODES="${EPISODES:-/home/wh/ActiveMap/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/episodes_val.jsonl}"
VALUE_ROOT="${VALUE_ROOT:-/home/wh/ActiveMap/runs/evidence_value_head_v1_20260725}"
RANKER="${RANKER:-/home/wh/ActiveMap/runs/sn7_active_catalog/candidate_ranker_v4/seed20260720/best.pt}"
OLD_SELECTOR="${OLD_SELECTOR:-/home/wh/ActiveMap/runs/selector/sn7_executable512_value_aware_gate_seed20260722_v6/edit_utility_seed20260722/best.pt}"

launch() {
  local gpu="$1"
  local name="$2"
  shift 2
  local output="${ROOT}/${name}"
  if [[ -e "${output}" ]]; then
    echo "Refusing to overwrite ${output}" >&2
    return 1
  fi
  mkdir -p "${ROOT}/logs" "${ROOT}/pids"
  (
    cd "${REPO}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src:. nohup "${PYTHON}" \
      scripts/evaluate_active_catalog_closed_loop_baselines.py \
      "${STATES}" "${output}" \
      --device cuda:0 \
      --max-acquisitions 2 \
      --bootstrap-repetitions 5000 \
      "$@" \
      >"${ROOT}/logs/${name}.log" 2>&1 < /dev/null &
    echo "$!" >"${ROOT}/pids/${name}.pid"
  )
  echo "${name}: pid=$(cat "${ROOT}/pids/${name}.pid") gpu=${gpu}"
}

mkdir -p "${ROOT}"
launch 0 seed1 \
  --evidence-value "value_seed1=${VALUE_ROOT}/seed1/run/best.pt" \
  --policy always_stop \
  --policy value_seed1 \
  --policy shortlist_oracle_upper_bound
launch 2 seed2 \
  --evidence-value "value_seed2=${VALUE_ROOT}/seed2/run/best.pt" \
  --policy value_seed2
launch 3 seed3 \
  --evidence-value "value_seed3=${VALUE_ROOT}/seed3/run/best.pt" \
  --policy value_seed3
launch 6 matched_baselines \
  --evidence-value "utility_only=${VALUE_ROOT}/utility_only_seed1/run/best.pt" \
  --learned-selector "old_value_aware=${OLD_SELECTOR}" \
  --ranker-checkpoint "${RANKER}" \
  --episodes "${EPISODES}" \
  --policy utility_only \
  --policy old_value_aware \
  --policy ranker_only
