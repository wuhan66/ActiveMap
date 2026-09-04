#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
DATA="${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded"
BUNDLE="${DATA}/closed_loop_val_bundle_v1"
STATES="${BUNDLE}/states_val_step0.jsonl"
EPISODES="${BUNDLE}/episodes_val.jsonl"
SELECTOR_ROOT="${SELECTOR_ROOT:-${STORAGE_ROOT}/runs/selector/sn7_step0_two_stage_v2}"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"
UPDATER="${STORAGE_ROOT}/runs/updater/v4_hierarchical_vector_change_scratch_seed20260716/best_quality.pt"
TOOL_BELIEF="${STORAGE_ROOT}/runs/agent/tool_belief_anchored_v4_seed20260821/best.pt"
TOOL_GATE="${TOOL_GATE:-${RUN}/uncertainty_tool_gate_train_q15/gate.json}"
TOOL_GATE_SUMMARY="${TOOL_GATE_SUMMARY:-${RUN}/uncertainty_tool_gate_train_q15/summary.json}"
OUTPUT="${OUTPUT:-${RUN}/structured_selector_tool_causal_v1}"
LIMIT="${LIMIT:-512}"
SAMPLE_SEED="${SAMPLE_SEED:-20260729}"
WRITEBACK="${WRITEBACK:-1}"
GPUS="${GPUS:-0,1,2}"
SEEDS_CSV="${SEEDS_CSV:-20260730,20260731,20260801}"
MIN_SEEDS="${MIN_SEEDS:-3}"
POLL_SECONDS="${POLL_SECONDS:-60}"
IFS=',' read -ra SEEDS <<< "${SEEDS_CSV}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

for path in \
  "${PYTHON}" "${STATES}" "${EPISODES}" "${UPDATER}" \
  "${TOOL_BELIEF}" "${TOOL_GATE}" "${TOOL_GATE_SUMMARY}"; do
  [[ -e "${path}" ]] || { echo "missing structured-causal input: ${path}" >&2; exit 3; }
done
[[ "${LIMIT}" =~ ^[1-9][0-9]*$ ]] || { echo "LIMIT must be positive" >&2; exit 4; }
[[ "${WRITEBACK}" == "0" || "${WRITEBACK}" == "1" ]] || {
  echo "WRITEBACK must be 0 or 1" >&2
  exit 4
}

IFS=',' read -ra GPU_LIST <<< "${GPUS}"
[[ "${#GPU_LIST[@]}" -eq "${#SEEDS[@]}" ]] || {
  echo "GPU and selector seed counts must match" >&2
  exit 4
}
[[ "${#SEEDS[@]}" -ge 1 && "${#SEEDS[@]}" -le 3 ]] || {
  echo "one to three selector seeds are required" >&2
  exit 4
}
[[ "${MIN_SEEDS}" =~ ^[1-3]$ && "${MIN_SEEDS}" -le "${#SEEDS[@]}" ]] || {
  echo "MIN_SEEDS must be between 1 and the selector seed count" >&2
  exit 4
}
for seed in "${SEEDS[@]}"; do
  [[ "${seed}" =~ ^[0-9]+$ ]] || {
    echo "invalid selector seed: ${seed}" >&2
    exit 4
  }
done

selector_for_seed() {
  echo "${SELECTOR_ROOT}_seed$1/edit_utility_seed$1/best.pt"
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
  local seed="$1" label="$2" tool_mode="$3" belief_mode="$4"
  local selector branch
  selector="$(selector_for_seed "${seed}")"
  branch="${OUTPUT}/seed${seed}/${label}"
  [[ -s "${selector}" ]] || { echo "missing selector: ${selector}" >&2; return 5; }
  if [[ -s "${branch}/edit_utility.jsonl" && -s "${branch}/summary.json" ]]; then
    return
  fi
  if [[ -e "${branch}" ]]; then
    archive="${branch}_failed_$(date +%Y%m%d_%H%M%S)"
    echo "archiving partial structured rollout: ${branch} -> ${archive}" >&2
    mv -- "${branch}" "${archive}"
  fi
  args=(
    "${STATES}" "${branch}"
    --episodes "${EPISODES}"
    --learned-selector "edit_utility=${selector}"
    --policy edit_utility
    --seed "${seed}" --device cpu --split val
    --tool-mode "${tool_mode}" --belief-mode "${belief_mode}"
    --max-candidates 16 --max-acquisitions 2
    --bootstrap-repetitions 500 --limit "${LIMIT}"
    --sample-seed "${SAMPLE_SEED}"
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}"
  )
  if [[ "${tool_mode}" != "none" ]]; then
    args+=(
      --tool-belief-checkpoint "${TOOL_BELIEF}"
      --tool-artifact-root "${branch}/tool_artifacts"
      --tool-out-size 256
    )
  fi
  if [[ "${tool_mode}" == "selective" ]]; then
    args+=(--tool-gate "${TOOL_GATE}" --tool-gate-summary "${TOOL_GATE_SUMMARY}")
  fi
  "${PYTHON}" scripts/evaluate_active_catalog_closed_loop_baselines.py "${args[@]}"
}

run_writeback() {
  local seed="$1" gpu="$2" label="$3"
  local trace input output
  trace="${OUTPUT}/seed${seed}/${label}/edit_utility.jsonl"
  input="${OUTPUT}/seed${seed}/${label}_writeback_input.jsonl"
  output="${OUTPUT}/seed${seed}/${label}_writeback"
  if [[ ! -s "${input}" ]]; then
    "${PYTHON}" scripts/convert_active_catalog_closed_loop_for_writeback.py \
      "${trace}" "${input}" --split val
  fi
  if [[ -s "${output}/evaluation/writeback.jsonl" ]]; then return; fi
  [[ ! -e "${output}" ]] || {
    echo "refusing partial structured writeback: ${output}" >&2
    return 7
  }
  wait_gpu "${gpu}"
  "${PYTHON}" scripts/launch_active_catalog_writeback.py \
    "${UPDATER}" "${EPISODES}" "${input}" "${output}" \
    --gpu "${gpu}" --image-size 512 --threshold 0.5 \
    --protocol-name "sn7-structured-selector-${label}-writeback-v1" \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
    --split val --limit "${LIMIT}" --monitor-interval 5
}

run_seed() {
  local seed="$1" gpu="$2"
  run_rollout "${seed}" no_tool none recurrent
  run_rollout "${seed}" forced_recurrent forced recurrent
  run_rollout "${seed}" selective_frozen_prior selective frozen_prior
  run_rollout "${seed}" selective_recurrent selective recurrent
  if [[ "${WRITEBACK}" == "1" ]]; then
    for label in no_tool forced_recurrent selective_frozen_prior selective_recurrent; do
      run_writeback "${seed}" "${gpu}" "${label}"
    done
  fi
}

mkdir -p "${OUTPUT}/logs"
pids=()
for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[${index}]}"
  run_seed "${seed}" "${GPU_LIST[${index}]}" \
    >"${OUTPUT}/logs/seed${seed}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
[[ "${status}" -eq 0 ]] || exit "${status}"

if [[ "${#SEEDS[@]}" -eq 1 ]]; then
  "${PYTHON}" -c '
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
seed = int(sys.argv[2])
summaries = sorted(root.glob(f"seed{seed}/*/summary.json"))
payload = {
    "schema_version": "sn7-structured-selector-tool-causal-smoke-manifest-v1",
    "selector_seeds": [seed],
    "branches": [path.parent.name for path in summaries],
    "record_limit": int(sys.argv[3]),
    "writeback": bool(int(sys.argv[4])),
    "deterministic_sample_seed": int(sys.argv[5]),
    "split": "val",
    "test_assets_read": False,
    "artifacts": {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in summaries
    },
}
(root / "smoke_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
' "${OUTPUT}" "${SEEDS[0]}" "${LIMIT}" "${WRITEBACK}" "${SAMPLE_SEED}"
  exit 0
fi

records=()
for seed in "${SEEDS[@]}"; do
  records+=(--records "${seed}:no_tool=${OUTPUT}/seed${seed}/no_tool/edit_utility.jsonl")
  records+=(--records "${seed}:forced=${OUTPUT}/seed${seed}/forced_recurrent/edit_utility.jsonl")
  records+=(--records "${seed}:selective_frozen_prior=${OUTPUT}/seed${seed}/selective_frozen_prior/edit_utility.jsonl")
  records+=(--records "${seed}:selective=${OUTPUT}/seed${seed}/selective_recurrent/edit_utility.jsonl")
done
"${PYTHON}" scripts/aggregate_active_catalog_tool_branches.py \
  "${OUTPUT}/paired_tool_belief_causal.json" "${records[@]}" \
  --repetitions 5000 --seed 20260729
"${PYTHON}" scripts/assess_active_catalog_tool_branch_promotion.py \
  "${OUTPUT}/paired_tool_belief_causal.json" "${OUTPUT}/promotion.json" \
  --false-edit-margin 0.02 --accuracy-margin 0.01 --min-seeds "${MIN_SEEDS}"

if [[ "${WRITEBACK}" == "1" ]]; then
  for reference in no_tool forced_recurrent selective_frozen_prior; do
    baseline_args=()
    candidate_args=()
    for seed in "${SEEDS[@]}"; do
      baseline_args+=(--baseline "${seed}=${OUTPUT}/seed${seed}/${reference}_writeback/evaluation/writeback.jsonl")
      candidate_args+=(--candidate "${seed}=${OUTPUT}/seed${seed}/selective_recurrent_writeback/evaluation/writeback.jsonl")
    done
    "${PYTHON}" scripts/aggregate_active_catalog_tool_writebacks.py \
      "${OUTPUT}/writeback_selective_vs_${reference}.json" \
      "${baseline_args[@]}" "${candidate_args[@]}" \
      --repetitions 5000 --seed 20260729
  done
  "${PYTHON}" scripts/assess_active_catalog_tool_writeback_promotion.py \
    "${OUTPUT}/promotion.json" \
    "${OUTPUT}/writeback_selective_vs_no_tool.json" \
    "${OUTPUT}/writeback_selective_vs_forced_recurrent.json" \
    "${OUTPUT}/writeback_promotion.json" \
    --replay-margin 0.01 --topology-margin 0.01 --min-seeds "${MIN_SEEDS}" \
    --record-failed-upstream
fi

"${PYTHON}" -c '
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
files = sorted(path for path in root.glob("*.json") if path.name != "manifest.json")
payload = {
    "schema_version": "sn7-structured-selector-tool-causal-manifest-v1",
    "selector_seeds": [int(value) for value in sys.argv[5].split(",")],
    "branches": ["no_tool", "forced_recurrent", "selective_frozen_prior", "selective_recurrent"],
    "record_limit": int(sys.argv[2]),
    "writeback": bool(int(sys.argv[3])),
    "deterministic_sample_seed": int(sys.argv[4]),
    "minimum_seed_count": int(sys.argv[6]),
    "split": "val",
    "test_assets_read": False,
    "artifacts": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
}
(root / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
' "${OUTPUT}" "${LIMIT}" "${WRITEBACK}" "${SAMPLE_SEED}" "${SEEDS_CSV}" "${MIN_SEEDS}"
