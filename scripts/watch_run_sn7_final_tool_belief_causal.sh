#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
MODEL="/home/wh/hf_models/Qwen3-VL-4B-Instruct"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"
DATA="${STORAGE_ROOT}/processed/sn7_v1/agent/sequential_selector_v1/full"
STATES="${DATA}/closed_loop_v1/states_val_step0.jsonl"
EPISODES="${DATA}/closed_loop_v1/episodes_val.jsonl"
PROMPTS="${DATA}/active_catalog_sft_v4/val.jsonl"
INDEX="${DATA}/active_catalog_sft_v4/val_evaluation_index.jsonl"
UPDATER="${STORAGE_ROOT}/runs/updater/v4_hierarchical_vector_change_scratch_seed20260716/best_quality.pt"
TOOL_BELIEF="${STORAGE_ROOT}/runs/agent/tool_belief_anchored_v4_seed20260821/best.pt"
TOOL_GATE="${RUN}/uncertainty_tool_gate_train_q15/gate.json"
TOOL_GATE_SUMMARY="${RUN}/uncertainty_tool_gate_train_q15/summary.json"
SELECTION="${RUN}/conservative_rl_selection_n512_seed20260718/selection.json"
OUTPUT="${RUN}/final_tool_belief_causal_v1"
POLL_SECONDS="${POLL_SECONDS:-60}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"

for path in \
  "${MODEL}" "${STATES}" "${EPISODES}" "${PROMPTS}" "${INDEX}" \
  "${UPDATER}" "${TOOL_BELIEF}" "${TOOL_GATE}" "${TOOL_GATE_SUMMARY}"; do
  [[ -e "${path}" ]] || { echo "missing causal-ablation input: ${path}" >&2; exit 3; }
done

until [[ -s "${SELECTION}" ]]; do sleep "${POLL_SECONDS}"; done
selection_decision="$("${PYTHON}" -c "import json; print(json.load(open('${SELECTION}'))['decision'])")"
selected="$("${PYTHON}" -c "import json; print(json.load(open('${SELECTION}'))['selected_candidate'])")"
final_family="sft"
if [[ "${selection_decision}" == "promote_one" ]]; then
  assessment="${RUN}/promoted_${selected}_final_writeback_assessment.json"
  until [[ -s "${assessment}" ]]; do sleep "${POLL_SECONDS}"; done
  final_decision="$("${PYTHON}" -c "import json; print(json.load(open('${assessment}'))['decision'])")"
  [[ "${final_decision}" == "promote_rl" || "${final_decision}" == "retain_sft" ]] || {
    echo "invalid final RL assessment: ${final_decision}" >&2
    exit 4
  }
  [[ "${final_decision}" == "retain_sft" ]] || final_family="rl"
elif [[ "${selection_decision}" != "stop_rl" ]]; then
  echo "invalid RL selection decision: ${selection_decision}" >&2
  exit 5
fi

adapter_for_seed() {
  local seed="$1"
  if [[ "${final_family}" == "sft" ]]; then
    case "${seed}" in
      20260717) echo "${RUN}/qwen3vl4b_seed1_eval500/seed20260717/final" ;;
      20260718) echo "${RUN}/qwen3vl4b_weighted_replication_seed2/seed20260718/final" ;;
      20260719) echo "${RUN}/qwen3vl4b_weighted_replication_seed3/seed20260719/final" ;;
    esac
    return
  fi
  case "${seed}" in
    20260717|20260719)
      echo "${RUN}/promoted_${selected}_sftseed${seed}_n4096/seed${seed}/final"
      ;;
    20260718)
      case "${selected}" in
        rl_lr2p5_kl005_bal50) tag="lr2p5e6_kl005_bal50" ;;
        rl_lr2p5_kl010_bal50) tag="lr2p5e6_kl010_bal50" ;;
        rl_lr5_kl010_bal50) tag="lr5e6_kl010_bal50" ;;
        rl_lr5_kl005_bal25) tag="lr5e6_kl005_bal25" ;;
        rl_lr2p5_kl010_bal25) tag="lr2p5e6_kl010_bal25" ;;
        *) return 1 ;;
      esac
      echo "${RUN}/contextual_rl_conservative_${tag}_n4096/seed20260718/final"
      ;;
  esac
}

gpu_is_idle() {
  local pids
  pids="$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${pids//[[:space:]]/}" ]]
}

wait_gpu() {
  until gpu_is_idle "$1"; do sleep "${POLL_SECONDS}"; done
}

run_rollout() {
  local seed="$1" gpu="$2" label="$3" tool_mode="$4" belief_mode="$5"
  local adapter output
  adapter="$(adapter_for_seed "${seed}")"
  output="${OUTPUT}/seed${seed}/${label}"
  [[ -s "${adapter}/adapter_config.json" ]] || { echo "missing adapter: ${adapter}" >&2; return 6; }
  if [[ -s "${output}/evaluation/traces.jsonl" ]]; then return; fi
  [[ ! -e "${output}" ]] || { echo "refusing partial rollout: ${output}" >&2; return 7; }
  wait_gpu "${gpu}"
  args=(
    "${MODEL}" "${adapter}" "${STATES}" "${EPISODES}" "${PROMPTS}" "${INDEX}"
    "${output}" --gpu "${gpu}" --seed "${seed}" --policy-mode full_action
    --tool-mode "${tool_mode}" --belief-mode "${belief_mode}"
    --max-candidates 16 --max-acquisitions 2 --max-new-tokens 64
    --bootstrap-repetitions 500 --limit 512 --monitor-interval 5
  )
  if [[ "${tool_mode}" != "none" ]]; then
    args+=(
      --tool-belief-checkpoint "${TOOL_BELIEF}"
      --tool-artifact-root "${output}/tool_artifacts" --tool-out-size 256
    )
  fi
  if [[ "${tool_mode}" == "selective" ]]; then
    args+=(--tool-gate "${TOOL_GATE}" --tool-gate-summary "${TOOL_GATE_SUMMARY}")
  fi
  "${PYTHON}" scripts/launch_active_catalog_closed_loop.py "${args[@]}"
}

run_writeback() {
  local seed="$1" gpu="$2" label="$3"
  local trace input output
  trace="${OUTPUT}/seed${seed}/${label}/evaluation/traces.jsonl"
  input="${OUTPUT}/seed${seed}/${label}_writeback_input.jsonl"
  output="${OUTPUT}/seed${seed}/${label}_writeback"
  if [[ ! -s "${input}" ]]; then
    "${PYTHON}" scripts/convert_active_catalog_closed_loop_for_writeback.py \
      "${trace}" "${input}" --split val
  fi
  if [[ -s "${output}/evaluation/writeback.jsonl" ]]; then return; fi
  if [[ -s "${output}/process_result.json" ]]; then
    prior_status="$("${PYTHON}" -c \
      "import json; print(json.load(open('${output}/process_result.json'))['status'])")"
    if [[ "${prior_status}" == "failed" ]]; then
      archive="${output}_failed_$(date +%Y%m%d_%H%M%S)"
      echo "archiving failed writeback: ${output} -> ${archive}" >&2
      mv -- "${output}" "${archive}"
    fi
  fi
  [[ ! -e "${output}" ]] || { echo "refusing partial writeback: ${output}" >&2; return 8; }
  wait_gpu "${gpu}"
  "${PYTHON}" scripts/launch_active_catalog_writeback.py \
    "${UPDATER}" "${EPISODES}" "${input}" "${output}" \
    --gpu "${gpu}" --image-size 512 --threshold 0.5 \
    --protocol-name "sn7-final-${final_family}-${label}-writeback-v1" \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
    --split val --limit 512 --monitor-interval 5
}

run_seed() {
  local seed="$1" gpu="$2"
  run_rollout "${seed}" "${gpu}" no_tool none recurrent
  run_rollout "${seed}" "${gpu}" forced_recurrent forced recurrent
  run_rollout "${seed}" "${gpu}" selective_frozen_prior selective frozen_prior
  run_rollout "${seed}" "${gpu}" selective selective recurrent
  for label in no_tool forced_recurrent selective_frozen_prior selective; do
    run_writeback "${seed}" "${gpu}" "${label}"
  done
}

mkdir -p "${OUTPUT}/logs"
run_seed 20260717 0 >"${OUTPUT}/logs/seed20260717.log" 2>&1 & pid1="$!"
run_seed 20260718 1 >"${OUTPUT}/logs/seed20260718.log" 2>&1 & pid2="$!"
run_seed 20260719 2 >"${OUTPUT}/logs/seed20260719.log" 2>&1 & pid3="$!"
status=0
for pid in "${pid1}" "${pid2}" "${pid3}"; do wait "${pid}" || status=1; done
[[ "${status}" -eq 0 ]] || exit "${status}"

records=()
for seed in 20260717 20260718 20260719; do
  records+=(--records "${seed}:no_tool=${OUTPUT}/seed${seed}/no_tool/evaluation/traces.jsonl")
  records+=(--records "${seed}:forced=${OUTPUT}/seed${seed}/forced_recurrent/evaluation/traces.jsonl")
  records+=(--records "${seed}:selective_frozen_prior=${OUTPUT}/seed${seed}/selective_frozen_prior/evaluation/traces.jsonl")
  records+=(--records "${seed}:selective=${OUTPUT}/seed${seed}/selective/evaluation/traces.jsonl")
done
if [[ ! -s "${OUTPUT}/paired_tool_belief_causal.json" ]]; then
  "${PYTHON}" scripts/aggregate_active_catalog_tool_branches.py \
    "${OUTPUT}/paired_tool_belief_causal.json" "${records[@]}" \
    --repetitions 5000 --seed 20260729
fi
if [[ ! -s "${OUTPUT}/promotion.json" ]]; then
  "${PYTHON}" scripts/assess_active_catalog_tool_branch_promotion.py \
    "${OUTPUT}/paired_tool_belief_causal.json" "${OUTPUT}/promotion.json" \
    --false-edit-margin 0.02 --accuracy-margin 0.01 --min-seeds 3
fi

for reference in no_tool forced_recurrent selective_frozen_prior; do
  baseline_args=()
  candidate_args=()
  for seed in 20260717 20260718 20260719; do
    baseline_args+=(--baseline "${seed}=${OUTPUT}/seed${seed}/${reference}_writeback/evaluation/writeback.jsonl")
    candidate_args+=(--candidate "${seed}=${OUTPUT}/seed${seed}/selective_writeback/evaluation/writeback.jsonl")
  done
  if [[ ! -s "${OUTPUT}/writeback_selective_vs_${reference}.json" ]]; then
    "${PYTHON}" scripts/aggregate_active_catalog_tool_writebacks.py \
      "${OUTPUT}/writeback_selective_vs_${reference}.json" \
      "${baseline_args[@]}" "${candidate_args[@]}" \
      --repetitions 5000 --seed 20260729
  fi
done

if [[ ! -s "${OUTPUT}/writeback_promotion.json" ]]; then
  "${PYTHON}" scripts/assess_active_catalog_tool_writeback_promotion.py \
    "${OUTPUT}/promotion.json" \
    "${OUTPUT}/writeback_selective_vs_no_tool.json" \
    "${OUTPUT}/writeback_selective_vs_forced_recurrent.json" \
    "${OUTPUT}/writeback_promotion.json" \
    --replay-margin 0.01 --topology-margin 0.01 --min-seeds 3 \
    --record-failed-upstream
fi

if [[ ! -s "${OUTPUT}/manifest.json" ]]; then
  "${PYTHON}" - "${OUTPUT}" "${final_family}" "${selected}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
files = sorted(path for path in root.glob("*.json") if path.name != "manifest.json")
payload = {
    "schema_version": "sn7-final-tool-belief-causal-manifest-v1",
    "final_policy_family": sys.argv[2],
    "selected_rl_candidate": None if sys.argv[3] == "None" else sys.argv[3],
    "branches": [
        "no_tool",
        "forced_recurrent",
        "selective_frozen_prior",
        "selective",
    ],
    "model_seeds": [20260717, 20260718, 20260719],
    "split": "val",
    "test_assets_read": False,
    "artifacts": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    },
}
(root / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
fi
