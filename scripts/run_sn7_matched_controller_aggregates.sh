#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"
OUTPUT="${STORAGE_ROOT}/artifacts/paper_evidence/sn7_matched_controller_3seed_v1"
SEEDS=(20260717 20260718 20260719)

mkdir -p "${OUTPUT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"

candidate_path() {
  local method="$1"
  local seed="$2"
  case "${method}" in
    react)
      echo "${RUN}/react_style_qwen_seed${seed}_n512_v2_writeback/evaluation/writeback.jsonl"
      ;;
    plan_execute)
      echo "${RUN}/plan_execute_qwen_seed${seed}_n512_v2_writeback/evaluation/writeback.jsonl"
      ;;
    geommagent_style)
      echo "${RUN}/geommagent_style_qwen_seed${seed}_full_v3_writeback/evaluation/writeback.jsonl"
      ;;
    sensesearch_style)
      echo "${RUN}/sensesearch_style_qwen_seed${seed}_full_v3_writeback/evaluation/writeback.jsonl"
      ;;
    active_map)
      echo "${RUN}/step0_matched_controller_support_v1/active_map_seed${seed}.jsonl"
      ;;
    *)
      echo "unsupported method: ${method}" >&2
      return 2
      ;;
  esac
}

for method in react plan_execute geommagent_style sensesearch_style active_map; do
  destination="${OUTPUT}/${method}_vs_direct.json"
  if [[ -s "${destination}" ]]; then
    echo "reuse ${destination}"
    continue
  fi
  args=()
  for seed in "${SEEDS[@]}"; do
    direct="${RUN}/closed_loop_sft_seed${seed}_n512_v2_writeback/evaluation/writeback.jsonl"
    candidate="$(candidate_path "${method}" "${seed}")"
    [[ -s "${direct}" ]] || {
      echo "missing Direct SFT writeback: ${direct}" >&2
      exit 3
    }
    [[ -s "${candidate}" ]] || {
      echo "missing ${method} writeback: ${candidate}" >&2
      exit 3
    }
    args+=(--pair "${seed}=${direct},${candidate}")
  done
  "${PYTHON}" scripts/aggregate_agent_writeback_pairs.py \
    "${destination}" "${args[@]}" --repetitions 5000 --seed 20260730
done

echo "completed matched three-seed controller aggregates: ${OUTPUT}"
