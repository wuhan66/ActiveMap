#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
source "${PROJECT_ROOT}/scripts/server_hdpi_env.sh"
PYTHON="${ACTIVEMAP_PYTHON:-${ACTIVEMAP_GIS_ENV}/bin/python}"
STAGE="${1:-status}"
GPU="${MUNO21_PIPELINE_GPU:-${ACTIVEMAP_GPU_IDS%%,*}}"

ROOT="${ACTIVEMAP_PROCESSED_ROOT}/muno21_v2"
UPDATER_ROOT="${ROOT}/updater"
AGENT_ROOT="${ROOT}/agent"
UPDATER_MANIFEST="${UPDATER_ROOT}/updater_samples.jsonl"
UPDATER_CHECKPOINT="${MUNO21_UPDATER_CHECKPOINT:-${ACTIVEMAP_RUN_ROOT}/updater/muno21_road_v7_topology_scratch_seed20260731/best.pt}"
EPISODES="${AGENT_ROOT}/episodes_train_val_v1.jsonl"
STATES="${AGENT_ROOT}/selector_states_v1.jsonl"
SELECTOR_ROOT="${ACTIVEMAP_RUN_ROOT}/ablations/muno21_selector_structural"
BASE_AGENT="${AGENT_ROOT}/agent_data_v6_anonymized"
FLAT_TOOL="${AGENT_ROOT}/tool_belief_v1"
SEQUENCE_TOOL="${AGENT_ROOT}/tool_belief_sequence_v1"
TOOL_RUN="${ACTIVEMAP_RUN_ROOT}/agent/tool_belief_anchored_v4_seed20260821"
TOOL_EVAL_VAL="${ACTIVEMAP_RUN_ROOT}/agent/tool_belief_anchored_v4_seed20260821_sequence_eval"
TOOL_EVAL_TRAIN="${ACTIVEMAP_RUN_ROOT}/agent/tool_belief_anchored_v4_seed20260821_sequence_eval_train"
SPARSE_TOOL="${AGENT_ROOT}/sparse_tool_sft_v2"
COMPOSED_AGENT="${AGENT_ROOT}/agent_data_v9_natural_sparse_tools"
RL_BASE="${AGENT_ROOT}/agent_rl_states_v1"
RL_TOOL="${AGENT_ROOT}/agent_rl_tool_states_v1"
RL_COMPOSED="${AGENT_ROOT}/agent_rl_states_v2_grounded_tools"
LOG_ROOT="${ACTIVEMAP_LOG_ROOT}/muno21_agent_pipeline"

mkdir -p "${AGENT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

require_file() {
  [[ -s "$1" ]] || { echo "Required input is missing: $1" >&2; exit 3; }
}

require_passed_audit() {
  require_file "$1"
  "${PYTHON}" -c \
    'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("passed") else 1)' \
    "$1"
}

require_gpu() {
  if [[ ! "${GPU}" =~ ^(0|1)$ ]]; then
    echo "Refusing GPU ${GPU}: ActiveMap is restricted to physical GPUs 0 and 1" >&2
    exit 2
  fi
  local pids
  pids="$(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  if [[ -n "${pids//[[:space:]]/}" ]]; then
    echo "Physical GPU ${GPU} has active compute processes: ${pids}" >&2
    exit 4
  fi
  export ACTIVEMAP_LAUNCHER_NAME="prepare_muno21_agent_pipeline.sh:${STAGE}"
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/scripts/acquire_training_slot.sh"
  export CUDA_VISIBLE_DEVICES="${GPU}"
}

full_selector_checkpoints() {
  local values=()
  local seed
  for seed in 20260821 20260822 20260823; do
    local path="${SELECTOR_ROOT}/full__seed${seed}/best.pt"
    require_file "${path}"
    values+=("${path}")
  done
  local IFS=,
  printf '%s' "${values[*]}"
}

build_episodes() {
  require_passed_audit "${UPDATER_ROOT}/audit.json"
  require_file "${UPDATER_MANIFEST}"
  if [[ -s "${EPISODES}" && -s "${EPISODES%.jsonl}.summary.json" ]]; then
    echo "episodes already complete: ${EPISODES}"
    return
  fi
  [[ ! -e "${EPISODES}" ]] || { echo "Refusing partial episodes: ${EPISODES}" >&2; exit 5; }
  "${PYTHON}" -m activemap.cli build-episodes-muno21 \
    "${UPDATER_MANIFEST}" "${EPISODES}" --splits train,val
  "${PYTHON}" scripts/audit_muno21_episode_mask_parity.py \
    "${UPDATER_MANIFEST}" "${EPISODES}" --split val --count 32 \
    >"${AGENT_ROOT}/episode_mask_parity_val.json"
}

build_selector_states() {
  require_file "${EPISODES}"
  require_file "${UPDATER_CHECKPOINT}"
  if [[ -s "${STATES}" && -s "${STATES%.jsonl}.summary.json" ]]; then
    echo "selector states already complete: ${STATES}"
    return
  fi
  [[ ! -e "${STATES}" ]] || { echo "Refusing partial selector states: ${STATES}" >&2; exit 5; }
  require_gpu
  "${PYTHON}" -m activemap.cli build-selector-oracle \
    "${UPDATER_CHECKPOINT}" "${EPISODES}" "${STATES}" \
    --device cuda:0 --image-size 512 --budgets 1.5,3.0,4.5 --splits train,val
  "${PYTHON}" scripts/audit_selector_states.py "${STATES}" \
    --output "${AGENT_ROOT}/selector_states_v1.audit.json"
  "${PYTHON}" scripts/audit_selector_utility_structure.py "${STATES}" \
    --output "${AGENT_ROOT}/selector_states_v1.utility_audit.json"
}

train_selectors() {
  require_file "${STATES}"
  require_gpu
  bash scripts/run_ablations.sh selector paper 2>&1 | tee "${LOG_ROOT}/selectors.log"
  require_file "${SELECTOR_ROOT}/runs.json"
  full_selector_checkpoints >/dev/null
}

build_base_agent() {
  require_file "${STATES}"
  local checkpoints
  checkpoints="$(full_selector_checkpoints)"
  local split
  for split in train val; do
    local output="${BASE_AGENT}/${split}"
    if [[ ! -s "${output}/summary.json" ]]; then
      [[ ! -e "${output}" ]] || { echo "Refusing partial Agent data: ${output}" >&2; exit 5; }
      "${PYTHON}" -m activemap.cli build-agent-data \
        "${STATES}" "${output}" --split "${split}" \
        --selector-checkpoints "${checkpoints}" --device cpu --top-k 3
    fi
    "${PYTHON}" scripts/audit_agent_data.py \
      "${output}/sft.jsonl" "${output}/preferences.jsonl" \
      --expected-split "${split}" --output "${output}/audit.json"
    "${PYTHON}" scripts/audit_agent_identifier_leakage.py \
      "${output}/sft.jsonl" "${output}/preferences.jsonl"
  done
}

build_tool_data() {
  require_file "${EPISODES}"
  require_file "${STATES}"
  if [[ ! -s "${FLAT_TOOL}/summary.json" ]]; then
    [[ ! -e "${FLAT_TOOL}" ]] || { echo "Refusing partial tool data: ${FLAT_TOOL}" >&2; exit 5; }
    "${PYTHON}" scripts/build_muno21_tool_belief_data.py \
      "${EPISODES}" "${STATES}" "${FLAT_TOOL}" --out-size 512
  fi
  local train_count val_count
  train_count="$("${PYTHON}" -c 'import sys; print(sum(1 for x in open(sys.argv[1]) if x.strip()))' "${FLAT_TOOL}/train.jsonl")"
  val_count="$("${PYTHON}" -c 'import sys; print(sum(1 for x in open(sys.argv[1]) if x.strip()))' "${FLAT_TOOL}/val.jsonl")"
  "${PYTHON}" scripts/audit_tool_belief_data.py \
    "${FLAT_TOOL}/train.jsonl" "${FLAT_TOOL}/val.jsonl" \
    --expected-train "${train_count}" --expected-val "${val_count}" \
    --output "${FLAT_TOOL}/audit.json"
  if [[ ! -s "${SEQUENCE_TOOL}/summary.json" ]]; then
    [[ ! -e "${SEQUENCE_TOOL}" ]] || { echo "Refusing partial sequences: ${SEQUENCE_TOOL}" >&2; exit 5; }
    "${PYTHON}" scripts/build_tool_belief_sequences.py \
      "${FLAT_TOOL}" "${EPISODES}" "${STATES}" "${SEQUENCE_TOOL}"
  fi
}

train_tool_belief() {
  require_file "${SEQUENCE_TOOL}/train.jsonl"
  require_file "${SEQUENCE_TOOL}/val.jsonl"
  require_gpu
  PHYSICAL_GPU="${GPU}" DATA_ROOT="${ACTIVEMAP_STORAGE_ROOT}" \
    bash scripts/run_tool_belief_v4_pipeline.sh 2>&1 | tee "${LOG_ROOT}/tool_belief.log"
  require_file "${TOOL_RUN}/best.pt"
  require_file "${TOOL_EVAL_TRAIN}/details.jsonl"
  require_file "${TOOL_EVAL_VAL}/details.jsonl"
}

build_sparse_agent() {
  require_file "${TOOL_EVAL_TRAIN}/details.jsonl"
  require_file "${TOOL_EVAL_VAL}/details.jsonl"
  local split details
  for split in train val; do
    details="${TOOL_EVAL_VAL}/details.jsonl"
    [[ "${split}" == train ]] && details="${TOOL_EVAL_TRAIN}/details.jsonl"
    if [[ ! -s "${SPARSE_TOOL}/${split}/summary.json" ]]; then
      [[ ! -e "${SPARSE_TOOL}/${split}" ]] || { echo "Refusing partial sparse data" >&2; exit 5; }
      "${PYTHON}" scripts/build_sparse_tool_agent_sft.py \
        "${SEQUENCE_TOOL}/${split}.jsonl" "${details}" "${SPARSE_TOOL}/${split}"
    fi
  done
  if [[ ! -s "${COMPOSED_AGENT}/prior_audit.json" ]]; then
    MUNO21_AGENT_DATA_ROOT="${AGENT_ROOT}" \
      ACTIVEMAP_AGENT_PYTHON="${PYTHON}" \
      bash scripts/prepare_muno21_agent_data_v9_natural.sh
  fi
}

build_rl_data() {
  local split
  for split in train val; do
    mkdir -p "${RL_BASE}" "${RL_TOOL}" "${RL_COMPOSED}"
    if [[ ! -s "${RL_BASE}/${split}.jsonl.summary.json" ]]; then
      "${PYTHON}" scripts/build_agent_rl_states.py \
        "${BASE_AGENT}/${split}/trajectories.jsonl" "${RL_BASE}/${split}.jsonl" \
        --expected-split "${split}"
    fi
    if [[ ! -s "${RL_TOOL}/${split}.jsonl.summary.json" ]]; then
      "${PYTHON}" scripts/build_tool_agent_rl_states.py \
        "${SPARSE_TOOL}/${split}/sft.jsonl" \
        "${SPARSE_TOOL}/${split}/preferences.jsonl" \
        "${RL_TOOL}/${split}.jsonl" --expected-split "${split}"
    fi
    if [[ ! -s "${RL_COMPOSED}/${split}.jsonl.summary.json" ]]; then
      "${PYTHON}" scripts/compose_agent_rl_states.py \
        "${RL_COMPOSED}/${split}.jsonl" \
        "${RL_BASE}/${split}.jsonl" "${RL_TOOL}/${split}.jsonl" \
        --expected-split "${split}"
    fi
  done
}

status() {
  local path
  for path in \
    "${UPDATER_MANIFEST}" "${UPDATER_CHECKPOINT}" "${EPISODES}" "${STATES}" \
    "${SELECTOR_ROOT}/runs.json" "${BASE_AGENT}/train/summary.json" \
    "${FLAT_TOOL}/audit.json" "${SEQUENCE_TOOL}/summary.json" \
    "${TOOL_RUN}/best.pt" "${COMPOSED_AGENT}/prior_audit.json" \
    "${RL_COMPOSED}/train.jsonl.summary.json"; do
    if [[ -s "${path}" ]]; then printf 'READY\t%s\n' "${path}"; else printf 'MISSING\t%s\n' "${path}"; fi
  done
}

case "${STAGE}" in
  episodes) build_episodes ;;
  selector_states) build_selector_states ;;
  selectors) train_selectors ;;
  base_agent) build_base_agent ;;
  tool_data) build_tool_data ;;
  tool_belief) train_tool_belief ;;
  sparse_agent) build_sparse_agent ;;
  rl_data) build_rl_data ;;
  all)
    build_episodes
    build_selector_states
    train_selectors
    build_base_agent
    build_tool_data
    train_tool_belief
    build_sparse_agent
    build_rl_data
    ;;
  status) status ;;
  *)
    echo "Usage: $0 {status|episodes|selector_states|selectors|base_agent|tool_data|tool_belief|sparse_agent|rl_data|all}" >&2
    exit 2
    ;;
esac
