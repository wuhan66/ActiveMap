#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"
OUTPUT="${STORAGE_ROOT}/artifacts/paper_evidence/sn7_active_map_pairwise_3seed_v1"
SEEDS=(20260717 20260718 20260719)

mkdir -p "${OUTPUT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"

controller_path() {
  local method="$1"
  local seed="$2"
  case "${method}" in
    direct_sft)
      echo "${RUN}/closed_loop_sft_seed${seed}_n512_v2_writeback/evaluation/writeback.jsonl"
      ;;
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

pids=()
for baseline in direct_sft react plan_execute geommagent_style sensesearch_style; do
  destination="${OUTPUT}/active_map_vs_${baseline}.json"
  if [[ -s "${destination}" ]]; then
    echo "reuse ${destination}"
    continue
  fi
  args=()
  for seed in "${SEEDS[@]}"; do
    baseline_path="$(controller_path "${baseline}" "${seed}")"
    active_path="$(controller_path active_map "${seed}")"
    [[ -s "${baseline_path}" && -s "${active_path}" ]] || {
      echo "missing pair for ${baseline}, seed ${seed}" >&2
      exit 3
    }
    args+=(--pair "${seed}=${baseline_path},${active_path}")
  done
  "${PYTHON}" scripts/aggregate_agent_writeback_pairs.py \
    "${destination}" "${args[@]}" --repetitions 5000 --seed 20260730 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
(( status == 0 )) || exit "${status}"

echo "completed ActiveMap pairwise controller aggregates: ${OUTPUT}"
