#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-status}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
DATA_ROOT="${DATA_ROOT:-${STORAGE_ROOT}/processed/sn7_v1/agent/sequential_selector_v1}"
SFT_ROOT="${SFT_ROOT:-${DATA_ROOT}/full/active_catalog_sft_v4}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog}"
BACKBONE_MODEL="${BACKBONE_MODEL:-/home/wh/hf_models/gemma-3-4b-it}"
BACKBONE_NAME="${BACKBONE_NAME:-gemma3_4b}"
BACKBONE_GPU="${BACKBONE_GPU:-4}"
BACKBONE_TOKEN_REPORT="${BACKBONE_TOKEN_REPORT:-${RUN_ROOT}/${BACKBONE_NAME}_token_audit/full.json}"
BACKBONE_SMOKE_ROOT="${BACKBONE_SMOKE_ROOT:-${RUN_ROOT}/${BACKBONE_NAME}_smoke}"
SAMPLING_REPORT="${SAMPLING_REPORT:-${RUN_ROOT}/seed1_sampling_ablation.json}"
SEED="${SEED:-20260717}"

cd "${PROJECT_ROOT}"
source scripts/server_hdpi_env.sh
export PYTHONPATH="src:.:${PYTHONPATH:-}"
[[ "${BACKBONE_NAME}" =~ ^[A-Za-z0-9_-]+$ ]] || {
  echo "invalid BACKBONE_NAME: ${BACKBONE_NAME}" >&2
  exit 3
}

require_file() {
  test -f "$1" || { echo "missing required file: $1" >&2; exit 4; }
}

require_free_gpu() {
  local pids
  pids="$(nvidia-smi -i "${BACKBONE_GPU}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${pids//[[:space:]]/}" ]] || {
    echo "GPU ${BACKBONE_GPU} has active compute processes: ${pids}" >&2
    exit 5
  }
}

sampling_decision() {
  [[ -f "${SAMPLING_REPORT}" ]] || return 0
  "${PYTHON}" -c \
    'import json,sys; value=json.load(open(sys.argv[1])).get("decision"); print(value if value in {"weighted","unweighted"} else "")' \
    "${SAMPLING_REPORT}"
}

decision="$(sampling_decision 2>/dev/null || true)"
family="${BACKBONE_NAME}_${decision:-pending}_seed1"
run_family="${RUN_ROOT}/${family}"

case "${STAGE}" in
  status)
    echo "backbone=${BACKBONE_NAME} decision=${decision:-pending} family=${family}"
    test -f "${run_family}/seed${SEED}/run_state.json" && \
      cat "${run_family}/seed${SEED}/run_state.json" || true
    ;;
  verify_gates)
    "${PYTHON}" scripts/verify_sn7_active_catalog_gates.py token \
      "${BACKBONE_TOKEN_REPORT}" --expected-train 33236 --expected-val 6697 \
      --max-length 2048
    "${PYTHON}" scripts/verify_sn7_active_catalog_gates.py smoke \
      "${BACKBONE_SMOKE_ROOT}" --seed "${SEED}"
    require_file "${SAMPLING_REPORT}"
    [[ -n "${decision}" ]] || {
      echo "sampling comparison is absent or inconclusive: ${SAMPLING_REPORT}" >&2
      exit 6
    }
    ;;
  train)
    bash "$0" verify_gates
    require_free_gpu
    [[ ! -e "${run_family}" ]] || {
      echo "refusing to overwrite backbone run: ${run_family}" >&2
      exit 7
    }
    sampling_args=()
    if [[ "${decision}" == "weighted" ]]; then
      sampling_args+=(--acquire-sampling-target 0.25)
    fi
    "${PYTHON}" scripts/launch_semantic_vlm_sft_seeds.py \
      "${BACKBONE_MODEL}" "${SFT_ROOT}/train.jsonl" "${SFT_ROOT}/val.jsonl" \
      "${run_family}" --gpu "${BACKBONE_GPU}" --seed "${SEED}" \
      --epochs 3 --learning-rate 1e-4 --batch-size 1 --gradient-accumulation 16 \
      --max-length 2048 --lora-rank 16 --lora-alpha 32 \
      --logging-steps 5 --eval-steps 500 --save-steps 500 --save-total-limit 2 \
      --early-stopping-patience 3 --monitor-interval 5 \
      "${sampling_args[@]}"
    ;;
  evaluate)
    bash "$0" verify_gates
    adapter="${run_family}/seed${SEED}/final"
    output="${run_family}/seed${SEED}/active_catalog_val"
    require_file "${adapter}/adapter_config.json"
    require_free_gpu
    [[ ! -e "${output}" ]] || {
      echo "refusing to overwrite backbone evaluation: ${output}" >&2
      exit 8
    }
    CUDA_VISIBLE_DEVICES="${BACKBONE_GPU}" "${PYTHON}" \
      scripts/evaluate_active_catalog_selector.py \
      "${BACKBONE_MODEL}" "${adapter}" "${SFT_ROOT}/val.jsonl" \
      "${SFT_ROOT}/val_evaluation_index.jsonl" "${output}" \
      --device cuda:0 --seed "${SEED}" --bootstrap-repetitions 2000
    ;;
  *)
    echo "unknown stage: ${STAGE}" >&2
    exit 2
    ;;
esac
