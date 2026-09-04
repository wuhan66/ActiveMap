#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-status}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/ActiveMap/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
DATA_ROOT="${DATA_ROOT:-/home/wh/ActiveMap/processed/sn7_v1/agent/sequential_selector_v1}"
SFT_ROOT="${SFT_ROOT:-${DATA_ROOT}/full/active_catalog_sft_v4}"
RUN_ROOT="${RUN_ROOT:-/home/wh/ActiveMap/runs/sn7_active_catalog}"
GPU_TRAIN="${GPU_TRAIN:-0}"
GPU_SECOND="${GPU_SECOND:-2}"
GPU_POOL="${GPU_POOL:-${GPU_TRAIN},${GPU_SECOND}}"
MAX_ACTIVE_GPUS="${ACTIVEMAP_MAX_GPUS:-2}"
SEEDS="${SEEDS:-20260717,20260718,20260719}"
ACQUIRE_TARGET="${ACQUIRE_TARGET:-0.25}"
ACTION_WEIGHTED_ACQUIRE_LOSS_WEIGHT="${ACTION_WEIGHTED_ACQUIRE_LOSS_WEIGHT:-3.0}"
MIN_FREE_DISK_GIB="${MIN_FREE_DISK_GIB:-80}"
DPO_ADAPTER="${DPO_ADAPTER:-}"
CLOSED_LOOP_ADAPTER="${CLOSED_LOOP_ADAPTER:-}"
CLOSED_LOOP_ROOT="${CLOSED_LOOP_ROOT:-${DATA_ROOT}/full/closed_loop_v1}"
FULL_SELECTOR_STATES="${FULL_SELECTOR_STATES:-${DATA_ROOT}/selector_states_train_val.jsonl}"
TRAIN_VAL_EPISODES="${TRAIN_VAL_EPISODES:-${DATA_ROOT}/full/episodes_train_val.jsonl}"
UPDATER_CHECKPOINT="${UPDATER_CHECKPOINT:-${STORAGE_ROOT}/runs/updater/v4_hierarchical_vector_change_scratch_seed20260716/best_quality.pt}"
WRITEBACK_BASELINE_POLICY="${WRITEBACK_BASELINE_POLICY:-uncertainty_gate}"
WRITEBACK_ASSET_ROOT_MAP="${WRITEBACK_ASSET_ROOT_MAP:-}"
TOOL_ASSET_ROOT_MAP="${TOOL_ASSET_ROOT_MAP:-}"
RL_BASE_ADAPTER="${RL_BASE_ADAPTER:-}"
RL_ADAPTER="${RL_ADAPTER:-}"
RL_RUN_NAME="${RL_RUN_NAME:-qwen3vl4b_contextual_rl_seed1}"
RL_LEARNING_RATE="${RL_LEARNING_RATE:-1e-6}"
RL_ACTION_LIMIT="${RL_ACTION_LIMIT:-4}"
RL_KL_BETA="${RL_KL_BETA:-0.05}"
RL_ENTROPY_WEIGHT="${RL_ENTROPY_WEIGHT:-0.01}"
RL_UTILITY_SCALE="${RL_UTILITY_SCALE:-1.0}"
ONLINE_BASE_ADAPTER="${ONLINE_BASE_ADAPTER:-}"
ONLINE_ADAPTER="${ONLINE_ADAPTER:-}"
ONLINE_ROUND="${ONLINE_ROUND:-1}"
ONLINE_ROLLOUTS="${ONLINE_ROLLOUTS:-4}"
ONLINE_TRAIN_STATES="${ONLINE_TRAIN_STATES:-2048}"
ONLINE_VAL_STATES="${ONLINE_VAL_STATES:-512}"
ONLINE_TEMPERATURE="${ONLINE_TEMPERATURE:-0.8}"
ONLINE_TOP_P="${ONLINE_TOP_P:-0.95}"
ONLINE_REFERENCE_TRACE="${ONLINE_REFERENCE_TRACE:-}"
ONLINE_REFERENCE_WRITEBACK="${ONLINE_REFERENCE_WRITEBACK:-}"
SEED_TAG="${SEED_TAG:-seed1}"
SFT_RUN_NAME="${SFT_RUN_NAME:-qwen3vl4b_seed1_eval500}"
SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-500}"
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-500}"
TOOL_BELIEF_FAMILY="${TOOL_BELIEF_FAMILY:-joint_tool_belief}"
TOOL_GATE_FAMILY="${TOOL_GATE_FAMILY:-joint_tool_gate}"
TOOL_GATE_LABEL_BELIEF_FAMILY="${TOOL_GATE_LABEL_BELIEF_FAMILY:-joint_tool_belief}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

require_file() {
  test -f "$1" || { echo "missing required file: $1" >&2; exit 1; }
}

require_promotion() {
  require_file "$1"
  local promoted
  promoted="$(${PYTHON} -c 'import json,sys; d=json.load(open(sys.argv[1])); print(int(d.get("promote", False)))' "$1")"
  [[ "${promoted}" == "1" ]] || { echo "promotion gate failed: $1" >&2; exit 7; }
}

require_free_gpu() {
  local gpu="$1"
  local pids
  pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${pids//[[:space:]]/}" ]] || {
    echo "GPU ${gpu} has active compute processes: ${pids}" >&2
    exit 4
  }
}

build_gpu_args() {
  local allowed=",${ACTIVEMAP_GPU_IDS:-0,1},"
  local -A seen=()
  local gpu
  gpu_args=()
  IFS=',' read -r -a gpu_values <<< "${GPU_POOL}"
  (( ${#gpu_values[@]} > 0 )) || { echo "GPU_POOL is empty" >&2; exit 3; }
  (( ${#gpu_values[@]} <= MAX_ACTIVE_GPUS )) || {
    echo "GPU_POOL requests ${#gpu_values[@]} GPUs; limit is ${MAX_ACTIVE_GPUS}" >&2
    exit 3
  }
  for gpu in "${gpu_values[@]}"; do
    [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "invalid GPU id: ${gpu}" >&2; exit 3; }
    [[ "${allowed}" == *",${gpu},"* ]] || {
      echo "GPU ${gpu} is outside ACTIVEMAP_GPU_IDS=${ACTIVEMAP_GPU_IDS:-0,1}" >&2
      exit 3
    }
    [[ -z "${seen[${gpu}]:-}" ]] || { echo "duplicate GPU id: ${gpu}" >&2; exit 3; }
    seen[${gpu}]=1
    require_free_gpu "${gpu}"
    gpu_args+=(--gpu "${gpu}")
  done
}

preflight() {
  require_file "${SFT_ROOT}/train.jsonl"
  require_file "${SFT_ROOT}/val.jsonl"
  require_file "${SFT_ROOT}/val_evaluation_index.jsonl"
  require_file "${SFT_ROOT}/audit.json"
  require_free_gpu "${GPU_TRAIN}"
  require_free_gpu "${GPU_SECOND}"
  local available_kib available_gib
  available_kib="$(df -Pk "${RUN_ROOT}" 2>/dev/null | awk 'NR==2 {print $4}' || true)"
  if [[ -z "${available_kib}" ]]; then
    available_kib="$(df -Pk "$(dirname "${RUN_ROOT}")" | awk 'NR==2 {print $4}')"
  fi
  available_gib="$((available_kib / 1024 / 1024))"
  (( available_gib >= MIN_FREE_DISK_GIB )) || {
    echo "only ${available_gib} GiB free; require ${MIN_FREE_DISK_GIB} GiB" >&2
    exit 6
  }
  echo "preflight passed"
  echo "train_records=$(wc -l < "${SFT_ROOT}/train.jsonl")"
  echo "val_records=$(wc -l < "${SFT_ROOT}/val.jsonl")"
  echo "sft_size=$(du -sh "${SFT_ROOT}" | awk '{print $1}')"
  echo "free_disk_gib=${available_gib}"
  echo "reserved_gpus=${GPU_TRAIN},${GPU_SECOND}"
  echo "measured_single_run_peak_mib=13482"
}

evaluate_seed() {
  local seed="$1"
  local run_family="$2"
  local adapter="${RUN_ROOT}/${run_family}/seed${seed}/final"
  local output="${RUN_ROOT}/${run_family}/seed${seed}/active_catalog_val"
  require_file "${adapter}/adapter_config.json"
  require_file "${SFT_ROOT}/val.jsonl"
  require_file "${SFT_ROOT}/val_evaluation_index.jsonl"
  [[ ! -e "${output}" ]] || {
    echo "refusing to overwrite evaluation: ${output}" >&2
    exit 5
  }
  require_free_gpu "${GPU_SECOND}"
  CUDA_VISIBLE_DEVICES="${GPU_SECOND}" "${PYTHON}" \
    scripts/evaluate_active_catalog_selector.py \
    "${MODEL}" "${adapter}" "${SFT_ROOT}/val.jsonl" \
    "${SFT_ROOT}/val_evaluation_index.jsonl" "${output}" \
    --device cuda:0 --seed "${seed}" --bootstrap-repetitions 2000
}

seed_args=()
IFS=',' read -r -a seed_values <<< "${SEEDS}"
for seed in "${seed_values[@]}"; do
  seed_args+=(--seed "${seed}")
done
ACTIVE_SEED="${ACTIVE_SEED:-${seed_values[0]}}"
PRIMARY_SEED="${ACTIVE_CATALOG_PRIMARY_SEED:-${seed_values[0]}}"
seed_tag_for() {
  local seed="$1"
  if [[ "${seed}" == "${PRIMARY_SEED}" ]]; then
    printf '%s\n' "seed1"
  else
    printf 'seed%s\n' "${seed}"
  fi
}
[[ "${SEED_TAG}" =~ ^[A-Za-z0-9_-]+$ ]] || {
  echo "SEED_TAG must contain only letters, digits, underscore, or hyphen" >&2
  exit 3
}
BRANCH_SUFFIX=""
[[ "${SEED_TAG}" == "seed1" ]] || BRANCH_SUFFIX="_${SEED_TAG}"
GROUNDED_ROOT="${RUN_ROOT}/joint_grounded_tools${BRANCH_SUFFIX}"

common_train_args=(
  --epochs 3
  --learning-rate 1e-4
  --batch-size 1
  --gradient-accumulation 16
  --max-length 2048
  --lora-rank 16
  --lora-alpha 32
  --logging-steps 5
  --eval-steps "${SFT_EVAL_STEPS}"
  --save-steps "${SFT_SAVE_STEPS}"
  --save-total-limit 2
  --early-stopping-patience 3
  --monitor-interval 5
)
weighted_train_args=(
  "${common_train_args[@]}"
  --acquire-sampling-target "${ACQUIRE_TARGET}"
)

sampling_decision() {
  local report="${RUN_ROOT}/seed1_sampling_ablation.json"
  require_file "${report}"
  "${PYTHON}" -c '
import json, sys
report = json.load(open(sys.argv[1]))
if report.get("schema_version") != "active-catalog-sampling-ablation-v1":
    raise SystemExit("unexpected sampling report schema")
if report.get("test_assets_read") is not False:
    raise SystemExit("sampling report is not validation-only")
if report.get("promotion_eligible") is not True:
    raise SystemExit("sampling report is diagnostic-only")
decision = report.get("decision")
if decision not in {"weighted", "unweighted"}:
    raise SystemExit(f"sampling decision is not promotable: {decision}")
print(decision)
' "${report}"
}

promoted_replicate_family() {
  printf 'qwen3vl4b_promoted_%s_replicates\n' "$1"
}

case "${STAGE}" in
  status)
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader
    find "${RUN_ROOT}" -name run_state.json -type f -print -exec tail -n 20 {} \; \
      2>/dev/null || true
    ;;
  token_audit)
    require_file "${SFT_ROOT}/train.jsonl"
    require_file "${SFT_ROOT}/val.jsonl"
    mkdir -p "${RUN_ROOT}/token_audit"
    CUDA_VISIBLE_DEVICES= "${PYTHON}" scripts/audit_vlm_sft_token_lengths.py \
      "${MODEL}" "${SFT_ROOT}/train.jsonl" "${SFT_ROOT}/val.jsonl" \
      --max-length 2048 \
      --output "${RUN_ROOT}/token_audit/full_qwen3vl4b.json"
    ;;
  verify_token_audit)
    "${PYTHON}" scripts/verify_sn7_active_catalog_gates.py token \
      "${RUN_ROOT}/token_audit/full_qwen3vl4b.json" \
      --expected-train 33236 --expected-val 6697 --max-length 2048
    ;;
  preflight)
    preflight
    ;;
  smoke)
    require_file "${SFT_ROOT}/smoke_train.jsonl"
    require_file "${SFT_ROOT}/smoke_val.jsonl"
    "${PYTHON}" scripts/launch_semantic_vlm_sft_seeds.py \
      "${MODEL}" "${SFT_ROOT}/smoke_train.jsonl" "${SFT_ROOT}/smoke_val.jsonl" \
      "${RUN_ROOT}/smoke_full_protocol" \
      --gpu "${GPU_TRAIN}" --seed "${seed_values[0]}" \
      --epochs 1 --learning-rate 1e-4 --batch-size 1 --gradient-accumulation 1 \
      --max-length 2048 --lora-rank 16 --lora-alpha 32 \
      --logging-steps 1 --eval-steps 1 --save-steps 1 --save-total-limit 1 \
      --early-stopping-patience 1 --monitor-interval 2 \
      --acquire-sampling-target 0.5 \
      --expected-train-records 2 --expected-val-records 2 \
      --expected-train-action STOP=1 --expected-train-action ACQUIRE=1 \
      --expected-val-action STOP=1 --expected-val-action ACQUIRE=1
    ;;
  action_weighted_smoke)
    require_file "${SFT_ROOT}/smoke_train.jsonl"
    require_file "${SFT_ROOT}/smoke_val.jsonl"
    "${PYTHON}" scripts/launch_semantic_vlm_sft_seeds.py \
      "${MODEL}" "${SFT_ROOT}/smoke_train.jsonl" "${SFT_ROOT}/smoke_val.jsonl" \
      "${RUN_ROOT}/action_weighted_smoke" \
      --gpu "${GPU_TRAIN}" --seed "${seed_values[0]}" \
      --epochs 1 --learning-rate 1e-4 --batch-size 1 --gradient-accumulation 1 \
      --max-length 2048 --lora-rank 16 --lora-alpha 32 \
      --logging-steps 1 --eval-steps 1 --save-steps 1 --save-total-limit 1 \
      --early-stopping-patience 0 --monitor-interval 2 \
      --acquire-sampling-target 0.5 \
      --acquire-loss-weight "${ACTION_WEIGHTED_ACQUIRE_LOSS_WEIGHT}" \
      --expected-train-records 2 --expected-val-records 2 \
      --expected-train-action STOP=1 --expected-train-action ACQUIRE=1 \
      --expected-val-action STOP=1 --expected-val-action ACQUIRE=1
    ;;
  verify_smoke)
    "${PYTHON}" scripts/verify_sn7_active_catalog_gates.py smoke \
      "${RUN_ROOT}/smoke_full_protocol" --seed "${seed_values[0]}"
    ;;
  seed1)
    require_file "${SFT_ROOT}/train.jsonl"
    require_file "${SFT_ROOT}/val.jsonl"
    "${PYTHON}" scripts/launch_semantic_vlm_sft_seeds.py \
      "${MODEL}" "${SFT_ROOT}/train.jsonl" "${SFT_ROOT}/val.jsonl" \
      "${RUN_ROOT}/${SFT_RUN_NAME}" \
      --gpu "${GPU_TRAIN}" --seed "${seed_values[0]}" \
      "${weighted_train_args[@]}"
    ;;
  unweighted_seed1)
    require_file "${SFT_ROOT}/train.jsonl"
    require_file "${SFT_ROOT}/val.jsonl"
    "${PYTHON}" scripts/launch_semantic_vlm_sft_seeds.py \
      "${MODEL}" "${SFT_ROOT}/train.jsonl" "${SFT_ROOT}/val.jsonl" \
      "${RUN_ROOT}/qwen3vl4b_unweighted_seed1" \
      --gpu "${GPU_TRAIN}" --seed "${seed_values[0]}" \
      --epochs 3 --learning-rate 1e-4 --batch-size 1 --gradient-accumulation 16 \
      --max-length 2048 --lora-rank 16 --lora-alpha 32 \
      --logging-steps 5 --eval-steps "${SFT_EVAL_STEPS}" \
      --save-steps "${SFT_SAVE_STEPS}" --save-total-limit 2 \
      --early-stopping-patience 3 --monitor-interval 5
    ;;
  action_weighted_seed1)
    require_file "${SFT_ROOT}/train.jsonl"
    require_file "${SFT_ROOT}/val.jsonl"
    "${PYTHON}" scripts/launch_semantic_vlm_sft_seeds.py \
      "${MODEL}" "${SFT_ROOT}/train.jsonl" "${SFT_ROOT}/val.jsonl" \
      "${RUN_ROOT}/qwen3vl4b_action_weighted_seed1" \
      --gpu "${GPU_TRAIN}" --seed "${seed_values[0]}" \
      "${weighted_train_args[@]}" \
      --acquire-loss-weight "${ACTION_WEIGHTED_ACQUIRE_LOSS_WEIGHT}"
    ;;
  three_seed)
    require_file "${SFT_ROOT}/train.jsonl"
    require_file "${SFT_ROOT}/val.jsonl"
    build_gpu_args
    "${PYTHON}" scripts/launch_semantic_vlm_sft_seeds.py \
      "${MODEL}" "${SFT_ROOT}/train.jsonl" "${SFT_ROOT}/val.jsonl" \
      "${RUN_ROOT}/qwen3vl4b_three_seed" \
      "${gpu_args[@]}" \
      "${seed_args[@]}" "${weighted_train_args[@]}"
    ;;
  promoted_replicates)
    require_file "${SFT_ROOT}/train.jsonl"
    require_file "${SFT_ROOT}/val.jsonl"
    decision="$(sampling_decision)"
    family="$(promoted_replicate_family "${decision}")"
    build_gpu_args
    promoted_args=("${common_train_args[@]}")
    if [[ "${decision}" == "weighted" ]]; then
      promoted_args+=(--acquire-sampling-target "${ACQUIRE_TARGET}")
    fi
    "${PYTHON}" scripts/launch_semantic_vlm_sft_seeds.py \
      "${MODEL}" "${SFT_ROOT}/train.jsonl" "${SFT_ROOT}/val.jsonl" \
      "${RUN_ROOT}/${family}" \
      "${gpu_args[@]}" \
      "${seed_args[@]}" "${promoted_args[@]}"
    ;;
  evaluate_seed1)
    evaluate_seed "${seed_values[0]}" "${SFT_RUN_NAME}"
    ;;
  evaluate_unweighted_seed1)
    evaluate_seed "${seed_values[0]}" qwen3vl4b_unweighted_seed1
    ;;
  compare_seed1_sampling)
    weighted_trace="${RUN_ROOT}/${SFT_RUN_NAME}/seed${seed_values[0]}/active_catalog_val/traces.jsonl"
    unweighted_trace="${RUN_ROOT}/qwen3vl4b_unweighted_seed1/seed${seed_values[0]}/active_catalog_val/traces.jsonl"
    output="${RUN_ROOT}/seed1_sampling_ablation.json"
    require_file "${weighted_trace}"
    require_file "${unweighted_trace}"
    [[ ! -e "${output}" ]] || { echo "refusing to overwrite sampling ablation" >&2; exit 45; }
    "${PYTHON}" scripts/compare_active_catalog_sampling_ablation.py \
      "${weighted_trace}" "${unweighted_trace}" "${output}" \
      --repetitions 2000 --seed "${seed_values[0]}"
    ;;
  evaluate_three_seed)
    for seed in "${seed_values[@]}"; do
      evaluate_seed "${seed}" qwen3vl4b_three_seed
    done
    ;;
  aggregate_three_seed)
    aggregate_args=()
    for seed in "${seed_values[@]}"; do
      trace="${RUN_ROOT}/qwen3vl4b_three_seed/seed${seed}/active_catalog_val/traces.jsonl"
      require_file "${trace}"
      aggregate_args+=(--records "${seed}=${trace}")
    done
    "${PYTHON}" scripts/aggregate_active_catalog_seeds.py \
      "${RUN_ROOT}/qwen3vl4b_three_seed/active_catalog_aoi_bootstrap.json" \
      "${aggregate_args[@]}" --repetitions 2000 --bootstrap-seed 20260717
    ;;
  evaluate_promoted_replicates)
    decision="$(sampling_decision)"
    family="$(promoted_replicate_family "${decision}")"
    for seed in "${seed_values[@]}"; do
      evaluate_seed "${seed}" "${family}"
    done
    ;;
  aggregate_promoted_three_seed)
    decision="$(sampling_decision)"
    family="$(promoted_replicate_family "${decision}")"
    first_seed="${ACTIVE_CATALOG_PRIMARY_SEED:-20260717}"
    if [[ "${decision}" == "weighted" ]]; then
      first_family="${SFT_RUN_NAME}"
    else
      first_family="qwen3vl4b_unweighted_seed1"
    fi
    aggregate_root="${RUN_ROOT}/qwen3vl4b_promoted_three_seed"
    aggregate_output="${aggregate_root}/active_catalog_aoi_bootstrap.json"
    [[ ! -e "${aggregate_output}" ]] || {
      echo "refusing to overwrite promoted aggregation: ${aggregate_output}" >&2
      exit 46
    }
    first_trace="${RUN_ROOT}/${first_family}/seed${first_seed}/active_catalog_val/traces.jsonl"
    require_file "${first_trace}"
    aggregate_args=(--records "${first_seed}=${first_trace}")
    for seed in "${seed_values[@]}"; do
      [[ "${seed}" != "${first_seed}" ]] || {
        echo "replicate seed list repeats primary seed ${first_seed}" >&2
        exit 47
      }
      trace="${RUN_ROOT}/${family}/seed${seed}/active_catalog_val/traces.jsonl"
      require_file "${trace}"
      aggregate_args+=(--records "${seed}=${trace}")
    done
    (( ${#aggregate_args[@]} == 6 )) || {
      echo "promoted aggregation requires exactly one primary and two replicate seeds" >&2
      exit 48
    }
    "${PYTHON}" scripts/aggregate_active_catalog_seeds.py \
      "${aggregate_output}" "${aggregate_args[@]}" \
      --repetitions 2000 --bootstrap-seed "${first_seed}"
    "${PYTHON}" -c '
import hashlib, json, pathlib, sys
report, output, decision = map(str, sys.argv[1:])
report_path, output_path = pathlib.Path(report), pathlib.Path(output)
payload = {
    "schema_version": "active-catalog-promoted-seeds-v1",
    "sampling_decision": decision,
    "sampling_report": str(report_path.resolve()),
    "sampling_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    "aggregate": str(output_path.resolve()),
    "aggregate_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    "test_assets_read": False,
}
(output_path.parent / "promotion_manifest.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
' "${RUN_ROOT}/seed1_sampling_ablation.json" "${aggregate_output}" "${decision}"
    ;;
  paper_table_promoted)
    "${PYTHON}" scripts/export_active_catalog_paper_table.py \
      "${RUN_ROOT}/qwen3vl4b_promoted_three_seed/active_catalog_aoi_bootstrap.json" \
      "${RUN_ROOT}/qwen3vl4b_promoted_three_seed/paper_export"
    ;;
  paper_table)
    "${PYTHON}" scripts/export_active_catalog_paper_table.py \
      "${RUN_ROOT}/qwen3vl4b_three_seed/active_catalog_aoi_bootstrap.json" \
      "${RUN_ROOT}/qwen3vl4b_three_seed/paper_export"
    ;;
  build_preferences)
    preference_root="${RUN_ROOT}/active_catalog_preferences"
    [[ ! -e "${preference_root}" ]] || {
      echo "refusing to overwrite preferences: ${preference_root}" >&2
      exit 8
    }
    mkdir -p "${preference_root}"
    "${PYTHON}" scripts/build_active_catalog_vlm_preferences.py \
      "${SFT_ROOT}/train.jsonl" "${SFT_ROOT}/train_evaluation_index.jsonl" \
      "${preference_root}/train.jsonl" --expected-split train --pairs-per-state 2
    "${PYTHON}" scripts/build_active_catalog_vlm_preferences.py \
      "${SFT_ROOT}/val.jsonl" "${SFT_ROOT}/val_evaluation_index.jsonl" \
      "${preference_root}/val.jsonl" --expected-split val --pairs-per-state 2
    ;;
  dpo_smoke|dpo_seed1)
    [[ -n "${DPO_ADAPTER}" ]] || {
      echo "DPO_ADAPTER must explicitly name the validation-promoted SFT adapter" >&2
      exit 9
    }
    require_file "${DPO_ADAPTER}/adapter_config.json"
    preference_root="${RUN_ROOT}/active_catalog_preferences"
    require_file "${preference_root}/train.jsonl"
    require_file "${preference_root}/val.jsonl"
    require_free_gpu "${GPU_TRAIN}"
    dpo_output="${RUN_ROOT}/qwen3vl4b_dpo_seed1"
    dpo_args=(
      "${DPO_ADAPTER}" "${preference_root}/train.jsonl"
      "${preference_root}/val.jsonl" "${dpo_output}"
      --gpu "${GPU_TRAIN}"
      --epochs 1 --learning-rate 5e-6 --batch-size 1 --gradient-accumulation 16
      --max-length 2048 --beta 0.1 --seed "${seed_values[0]}"
      --logging-steps 5 --eval-steps 100 --save-steps 100 --monitor-interval 5
    )
    if [[ "${STAGE}" == "dpo_smoke" ]]; then
      dpo_args=(
        "${DPO_ADAPTER}" "${preference_root}/train.jsonl"
        "${preference_root}/val.jsonl" "${RUN_ROOT}/dpo_smoke"
        --gpu "${GPU_TRAIN}"
        --epochs 1 --learning-rate 5e-6 --batch-size 1 --gradient-accumulation 1
        --max-length 2048 --beta 0.1 --seed "${seed_values[0]}"
        --logging-steps 1 --eval-steps 1 --save-steps 1
        --max-train-samples 2 --max-eval-samples 2 --monitor-interval 2
      )
    fi
    "${PYTHON}" scripts/launch_active_catalog_vlm_dpo.py "${dpo_args[@]}"
    ;;
  build_rl_states)
    rl_data="${RUN_ROOT}/active_catalog_rl_states"
    [[ ! -e "${rl_data}" ]] || { echo "refusing to overwrite RL states" >&2; exit 37; }
    mkdir -p "${rl_data}"
    "${PYTHON}" scripts/build_active_catalog_vlm_rl_states.py \
      "${SFT_ROOT}/train.jsonl" "${SFT_ROOT}/train_evaluation_index.jsonl" \
      "${rl_data}/train.jsonl" --expected-split train --max-candidates 16
    "${PYTHON}" scripts/build_active_catalog_vlm_rl_states.py \
      "${SFT_ROOT}/val.jsonl" "${SFT_ROOT}/val_evaluation_index.jsonl" \
      "${rl_data}/val.jsonl" --expected-split val --max-candidates 16
    ;;
  assess_rl_readiness)
    readiness="${RUN_ROOT}/active_catalog_rl_readiness.json"
    require_file "${RUN_ROOT}/qwen3vl4b_three_seed/paper_export/promotion_gate.json"
    require_file "${RUN_ROOT}/tool_branch_three_seed/promotion.json"
    require_file "${RUN_ROOT}/tool_writeback_three_seed/promotion.json"
    [[ ! -e "${readiness}" ]] || { echo "refusing to overwrite RL readiness" >&2; exit 38; }
    "${PYTHON}" scripts/assess_active_catalog_rl_readiness.py \
      "${RUN_ROOT}/qwen3vl4b_three_seed/paper_export/promotion_gate.json" \
      "${RUN_ROOT}/tool_branch_three_seed/promotion.json" \
      "${RUN_ROOT}/tool_writeback_three_seed/promotion.json" "${readiness}"
    ;;
  rl_smoke|rl_seed1)
    [[ -n "${RL_BASE_ADAPTER}" ]] || { echo "RL_BASE_ADAPTER must name the promoted SFT adapter" >&2; exit 39; }
    require_file "${RL_BASE_ADAPTER}/adapter_config.json"
    require_file "${RUN_ROOT}/active_catalog_rl_states/train.jsonl"
    require_file "${RUN_ROOT}/active_catalog_rl_states/val.jsonl"
    require_free_gpu "${GPU_TRAIN}"
    rl_output="${RUN_ROOT}/${RL_RUN_NAME}"
    rl_extra=()
    if [[ "${STAGE}" == "rl_smoke" ]]; then
      rl_output="${RUN_ROOT}/contextual_rl_smoke"
      rl_extra=(--epochs 1 --max-train-samples 2 --max-eval-samples 2 --logging-steps 1 --eval-steps 1 --save-steps 1)
    else
      require_file "${RUN_ROOT}/active_catalog_rl_readiness.json"
      ready="$(${PYTHON} -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["ready_for_single_seed_rl"]))' "${RUN_ROOT}/active_catalog_rl_readiness.json")"
      [[ "${ready}" == "1" ]] || { echo "three-seed safety gates do not permit RL" >&2; exit 40; }
    fi
    "${PYTHON}" scripts/launch_active_catalog_vlm_rl.py \
      "${RL_BASE_ADAPTER}" "${RUN_ROOT}/active_catalog_rl_states/train.jsonl" \
      "${RUN_ROOT}/active_catalog_rl_states/val.jsonl" "${rl_output}" \
      --gpu "${GPU_TRAIN}" --seed "${ACTIVE_SEED}" --epochs 1 \
      --learning-rate "${RL_LEARNING_RATE}" --gradient-accumulation 16 --max-length 2048 \
      --action-limit "${RL_ACTION_LIMIT}" --kl-beta "${RL_KL_BETA}" \
      --entropy-weight "${RL_ENTROPY_WEIGHT}" --temperature 1.0 \
      --utility-scale "${RL_UTILITY_SCALE}" \
      --monitor-interval 5 "${rl_extra[@]}"
    ;;
  closed_loop_rl_seed1)
    [[ -n "${RL_ADAPTER}" ]] || { echo "RL_ADAPTER must name the trained RL adapter" >&2; exit 41; }
    require_file "${RL_ADAPTER}/adapter_config.json"
    require_free_gpu "${GPU_SECOND}"
    "${PYTHON}" scripts/launch_active_catalog_closed_loop.py \
      "${MODEL}" "${RL_ADAPTER}" "${CLOSED_LOOP_ROOT}/states_val_step0.jsonl" \
      "${CLOSED_LOOP_ROOT}/episodes_val.jsonl" "${SFT_ROOT}/val.jsonl" \
      "${SFT_ROOT}/val_evaluation_index.jsonl" "${RUN_ROOT}/closed_loop_rl_seed1" \
      --gpu "${GPU_SECOND}" --seed "${ACTIVE_SEED}" --max-candidates 16 \
      --max-acquisitions 2 --max-new-tokens 64 --bootstrap-repetitions 2000 \
      --monitor-interval 5
    ;;
  prepare_rl_writeback)
    trace="${RUN_ROOT}/closed_loop_rl_seed1/evaluation/traces.jsonl"
    require_file "${trace}"
    "${PYTHON}" scripts/convert_active_catalog_closed_loop_for_writeback.py \
      "${trace}" "${RUN_ROOT}/closed_loop_writeback_inputs/rl_seed1.jsonl"
    ;;
  closed_loop_writeback_rl)
    require_file "${UPDATER_CHECKPOINT}"
    require_file "${RUN_ROOT}/closed_loop_writeback_inputs/rl_seed1.jsonl"
    require_free_gpu "${GPU_SECOND}"
    root_map_args=()
    [[ -z "${WRITEBACK_ASSET_ROOT_MAP}" ]] || root_map_args=(--asset-root-map "${WRITEBACK_ASSET_ROOT_MAP}")
    "${PYTHON}" scripts/launch_active_catalog_writeback.py \
      "${UPDATER_CHECKPOINT}" "${TRAIN_VAL_EPISODES}" \
      "${RUN_ROOT}/closed_loop_writeback_inputs/rl_seed1.jsonl" \
      "${RUN_ROOT}/closed_loop_writeback_rl_seed1" --gpu "${GPU_SECOND}" \
      --image-size 512 --threshold 0.5 --protocol-name sn7-active-catalog-rl-writeback-v1 \
      --monitor-interval 5 "${root_map_args[@]}"
    ;;
  compare_rl_vs_sft)
    rl_trace="${RUN_ROOT}/closed_loop_rl_seed1/evaluation/traces.jsonl"
    sft_trace="${RUN_ROOT}/closed_loop_seed1/evaluation/traces.jsonl"
    rl_writeback="${RUN_ROOT}/closed_loop_writeback_rl_seed1/evaluation/writeback.jsonl"
    sft_writeback="${RUN_ROOT}/closed_loop_writeback_no_tool/evaluation/writeback.jsonl"
    require_file "${rl_trace}"; require_file "${sft_trace}"
    require_file "${rl_writeback}"; require_file "${sft_writeback}"
    "${PYTHON}" scripts/compare_active_catalog_closed_loop.py \
      "${RUN_ROOT}/closed_loop_rl_seed1/paired_sft_aoi.json" --candidate rl \
      --records "rl=${rl_trace}" --records "sft=${sft_trace}" \
      --repetitions 2000 --seed "${ACTIVE_SEED}"
    "${PYTHON}" scripts/compare_agent_writebacks.py \
      "${sft_writeback}" "${rl_writeback}" \
      "${RUN_ROOT}/closed_loop_writeback_rl_seed1/paired_sft_aoi.json" \
      --bootstrap 2000 --seed "${ACTIVE_SEED}" --group-key aoi_id
    ;;
  assess_rl_promotion)
    "${PYTHON}" scripts/assess_active_catalog_closed_loop_promotion.py \
      "${RUN_ROOT}/closed_loop_rl_seed1/paired_sft_aoi.json" \
      "${RUN_ROOT}/closed_loop_writeback_rl_seed1/paired_sft_aoi.json" \
      "${RUN_ROOT}/closed_loop_writeback_rl_seed1/promotion.json" \
      --candidate rl --baseline sft --false-edit-margin 0.01 --topology-margin 0.01
    ;;
  prepare_onpolicy_states)
    root="${RUN_ROOT}/onpolicy_round${ONLINE_ROUND}"
    [[ ! -e "${root}/states" ]] || { echo "refusing to overwrite on-policy states" >&2; exit 42; }
    "${PYTHON}" scripts/sample_active_catalog_online_states.py \
      "${FULL_SELECTOR_STATES}" "${root}/states/train.jsonl" --split train \
      --count "${ONLINE_TRAIN_STATES}" --seed "${ACTIVE_SEED}"
    "${PYTHON}" scripts/sample_active_catalog_online_states.py \
      "${FULL_SELECTOR_STATES}" "${root}/states/val.jsonl" --split val \
      --count "${ONLINE_VAL_STATES}" --seed "${ACTIVE_SEED}"
    ;;
  collect_onpolicy_train|collect_onpolicy_val)
    [[ -n "${ONLINE_BASE_ADAPTER}" ]] || { echo "ONLINE_BASE_ADAPTER is required" >&2; exit 43; }
    require_file "${ONLINE_BASE_ADAPTER}/adapter_config.json"
    require_promotion "${RUN_ROOT}/closed_loop_writeback_rl_seed1/promotion.json"
    require_file "${RUN_ROOT}/${TOOL_BELIEF_FAMILY}_seed1/best_promoted.pt"
    require_file "${RUN_ROOT}/${TOOL_GATE_FAMILY}_seed1/gate.joblib"
    require_file "${RUN_ROOT}/${TOOL_GATE_FAMILY}_seed1/summary.json"
    split="${STAGE#collect_onpolicy_}"
    root="${RUN_ROOT}/onpolicy_round${ONLINE_ROUND}"
    states="${root}/states/${split}.jsonl"
    sft="${SFT_ROOT}/${split}.jsonl"
    index="${SFT_ROOT}/${split}_evaluation_index.jsonl"
    require_file "${states}"; require_file "${sft}"; require_file "${index}"
    for ((rollout=0; rollout<ONLINE_ROLLOUTS; rollout++)); do
      output="${root}/${split}_rollout${rollout}"
      require_free_gpu "${GPU_SECOND}"
      "${PYTHON}" scripts/launch_active_catalog_closed_loop.py \
        "${MODEL}" "${ONLINE_BASE_ADAPTER}" "${states}" "${TRAIN_VAL_EPISODES}" \
        "${sft}" "${index}" "${output}" --gpu "${GPU_SECOND}" \
        --seed "$((ACTIVE_SEED + rollout))" --split "${split}" --max-candidates 16 \
        --max-acquisitions 2 --max-new-tokens 64 --bootstrap-repetitions 0 \
        --tool-mode selective \
        --tool-belief-checkpoint "${RUN_ROOT}/${TOOL_BELIEF_FAMILY}_seed1/best_promoted.pt" \
        --tool-gate "${RUN_ROOT}/${TOOL_GATE_FAMILY}_seed1/gate.joblib" \
        --tool-gate-summary "${RUN_ROOT}/${TOOL_GATE_FAMILY}_seed1/summary.json" \
        --tool-artifact-root "${output}/tool_artifacts" --tool-out-size 256 \
        --do-sample --temperature "${ONLINE_TEMPERATURE}" --top-p "${ONLINE_TOP_P}" \
        --monitor-interval 5
    done
    ;;
  build_onpolicy_preferences)
    root="${RUN_ROOT}/onpolicy_round${ONLINE_ROUND}"
    for split in train val; do
      trace_args=()
      for ((rollout=0; rollout<ONLINE_ROLLOUTS; rollout++)); do
        trace="${root}/${split}_rollout${rollout}/evaluation/traces.jsonl"
        require_file "${trace}"
        trace_args+=(--traces "${trace}")
      done
      "${PYTHON}" scripts/build_active_catalog_onpolicy_preferences.py \
        "${SFT_ROOT}/${split}.jsonl" "${SFT_ROOT}/${split}_evaluation_index.jsonl" \
        "${root}/preferences/${split}.jsonl" "${trace_args[@]}" \
        --expected-split "${split}" --minimum-margin 0.001
    done
    ;;
  online_dpo_smoke|online_dpo_round)
    [[ -n "${ONLINE_BASE_ADAPTER}" ]] || { echo "ONLINE_BASE_ADAPTER is required" >&2; exit 44; }
    require_file "${ONLINE_BASE_ADAPTER}/adapter_config.json"
    root="${RUN_ROOT}/onpolicy_round${ONLINE_ROUND}"
    require_file "${root}/preferences/train.jsonl"
    require_file "${root}/preferences/val.jsonl"
    require_free_gpu "${GPU_TRAIN}"
    output="${root}/online_dpo"
    extra=()
    if [[ "${STAGE}" == "online_dpo_smoke" ]]; then
      output="${root}/online_dpo_smoke"
      extra=(--max-train-samples 2 --max-eval-samples 2 --logging-steps 1 --eval-steps 1 --save-steps 1)
    else
      require_promotion "${RUN_ROOT}/closed_loop_writeback_rl_seed1/promotion.json"
    fi
    "${PYTHON}" scripts/launch_active_catalog_vlm_dpo.py \
      "${ONLINE_BASE_ADAPTER}" "${root}/preferences/train.jsonl" \
      "${root}/preferences/val.jsonl" "${output}" --gpu "${GPU_TRAIN}" \
      --epochs 1 --learning-rate 2e-6 --batch-size 1 --gradient-accumulation 16 \
      --max-length 2048 --beta 0.05 --seed "${ACTIVE_SEED}" \
      --logging-steps 5 --eval-steps 100 --save-steps 100 --monitor-interval 5 \
      "${extra[@]}"
    ;;
  evaluate_online_round)
    [[ -n "${ONLINE_ADAPTER}" ]] || { echo "ONLINE_ADAPTER is required" >&2; exit 45; }
    require_file "${ONLINE_ADAPTER}/adapter_config.json"
    root="${RUN_ROOT}/onpolicy_round${ONLINE_ROUND}"
    require_free_gpu "${GPU_SECOND}"
    "${PYTHON}" scripts/launch_active_catalog_closed_loop.py \
      "${MODEL}" "${ONLINE_ADAPTER}" "${CLOSED_LOOP_ROOT}/states_val_step0.jsonl" \
      "${CLOSED_LOOP_ROOT}/episodes_val.jsonl" "${SFT_ROOT}/val.jsonl" \
      "${SFT_ROOT}/val_evaluation_index.jsonl" "${root}/evaluation" \
      --gpu "${GPU_SECOND}" --seed "${ACTIVE_SEED}" --max-candidates 16 \
      --max-acquisitions 2 --max-new-tokens 64 --bootstrap-repetitions 2000 \
      --tool-mode selective \
      --tool-belief-checkpoint "${RUN_ROOT}/${TOOL_BELIEF_FAMILY}_seed1/best_promoted.pt" \
      --tool-gate "${RUN_ROOT}/${TOOL_GATE_FAMILY}_seed1/gate.joblib" \
      --tool-gate-summary "${RUN_ROOT}/${TOOL_GATE_FAMILY}_seed1/summary.json" \
      --tool-artifact-root "${root}/evaluation/tool_artifacts" --tool-out-size 256 \
      --monitor-interval 5
    ;;
  prepare_online_writeback)
    root="${RUN_ROOT}/onpolicy_round${ONLINE_ROUND}"
    "${PYTHON}" scripts/convert_active_catalog_closed_loop_for_writeback.py \
      "${root}/evaluation/evaluation/traces.jsonl" "${root}/writeback_input.jsonl"
    ;;
  online_writeback)
    root="${RUN_ROOT}/onpolicy_round${ONLINE_ROUND}"
    require_file "${root}/writeback_input.jsonl"; require_file "${UPDATER_CHECKPOINT}"
    require_free_gpu "${GPU_SECOND}"
    root_map_args=()
    [[ -z "${WRITEBACK_ASSET_ROOT_MAP}" ]] || root_map_args=(--asset-root-map "${WRITEBACK_ASSET_ROOT_MAP}")
    "${PYTHON}" scripts/launch_active_catalog_writeback.py \
      "${UPDATER_CHECKPOINT}" "${TRAIN_VAL_EPISODES}" "${root}/writeback_input.jsonl" \
      "${root}/writeback" --gpu "${GPU_SECOND}" --image-size 512 --threshold 0.5 \
      --protocol-name sn7-active-catalog-online-writeback-v1 --monitor-interval 5 \
      "${root_map_args[@]}"
    ;;
  compare_online_round)
    [[ -n "${ONLINE_REFERENCE_TRACE}" && -n "${ONLINE_REFERENCE_WRITEBACK}" ]] || {
      echo "ONLINE_REFERENCE_TRACE and ONLINE_REFERENCE_WRITEBACK are required" >&2; exit 46;
    }
    root="${RUN_ROOT}/onpolicy_round${ONLINE_ROUND}"
    candidate_trace="${root}/evaluation/evaluation/traces.jsonl"
    candidate_writeback="${root}/writeback/evaluation/writeback.jsonl"
    require_file "${ONLINE_REFERENCE_TRACE}"; require_file "${ONLINE_REFERENCE_WRITEBACK}"
    require_file "${candidate_trace}"; require_file "${candidate_writeback}"
    "${PYTHON}" scripts/audit_active_catalog_policy_pair.py \
      "$(dirname "${candidate_trace}")/summary.json" \
      "$(dirname "${ONLINE_REFERENCE_TRACE}")/summary.json" "${root}/pair_audit.json"
    "${PYTHON}" scripts/compare_active_catalog_closed_loop.py \
      "${root}/paired_reference_aoi.json" --candidate online \
      --records "online=${candidate_trace}" --records "reference=${ONLINE_REFERENCE_TRACE}" \
      --repetitions 2000 --seed "${ACTIVE_SEED}"
    "${PYTHON}" scripts/compare_agent_writebacks.py \
      "${ONLINE_REFERENCE_WRITEBACK}" "${candidate_writeback}" \
      "${root}/writeback/paired_reference_aoi.json" --bootstrap 2000 \
      --seed "${ACTIVE_SEED}" --group-key aoi_id
    ;;
  assess_online_round)
    root="${RUN_ROOT}/onpolicy_round${ONLINE_ROUND}"
    "${PYTHON}" scripts/assess_active_catalog_closed_loop_promotion.py \
      "${root}/paired_reference_aoi.json" "${root}/writeback/paired_reference_aoi.json" \
      "${root}/promotion.json" --candidate online --baseline reference \
      --false-edit-margin 0.01 --topology-margin 0.01
    ;;
  prepare_closed_loop_bundle)
    require_file "${FULL_SELECTOR_STATES}"
    require_file "${TRAIN_VAL_EPISODES}"
    [[ ! -e "${CLOSED_LOOP_ROOT}" ]] || {
      echo "refusing to overwrite closed-loop bundle: ${CLOSED_LOOP_ROOT}" >&2
      exit 10
    }
    "${PYTHON}" scripts/prepare_active_catalog_closed_loop_bundle.py \
      "${FULL_SELECTOR_STATES}" "${TRAIN_VAL_EPISODES}" "${CLOSED_LOOP_ROOT}"
    ;;
  closed_loop_baselines)
    require_file "${CLOSED_LOOP_ROOT}/states_val_step0.jsonl"
    baseline_output="${RUN_ROOT}/closed_loop_baselines"
    [[ ! -e "${baseline_output}" ]] || {
      echo "refusing to overwrite closed-loop baselines: ${baseline_output}" >&2
      exit 12
    }
    CUDA_VISIBLE_DEVICES= "${PYTHON}" \
      scripts/evaluate_active_catalog_closed_loop_baselines.py \
      "${CLOSED_LOOP_ROOT}/states_val_step0.jsonl" "${baseline_output}" \
      --max-candidates 16 --max-acquisitions 2 \
      --bootstrap-repetitions 2000 --seed "${seed_values[0]}"
    ;;
  closed_loop_smoke|closed_loop_seed1|closed_loop_seed)
    [[ -n "${CLOSED_LOOP_ADAPTER}" ]] || {
      echo "CLOSED_LOOP_ADAPTER must explicitly name a validation-promoted adapter" >&2
      exit 11
    }
    require_file "${CLOSED_LOOP_ADAPTER}/adapter_config.json"
    require_file "${CLOSED_LOOP_ROOT}/states_val_step0.jsonl"
    require_file "${CLOSED_LOOP_ROOT}/episodes_val.jsonl"
    require_file "${CLOSED_LOOP_ROOT}/summary.json"
    require_file "${SFT_ROOT}/val.jsonl"
    require_file "${SFT_ROOT}/val_evaluation_index.jsonl"
    require_free_gpu "${GPU_SECOND}"
    closed_loop_output="${RUN_ROOT}/closed_loop_${SEED_TAG}"
    closed_loop_args=(
      "${MODEL}" "${CLOSED_LOOP_ADAPTER}"
      "${CLOSED_LOOP_ROOT}/states_val_step0.jsonl"
      "${CLOSED_LOOP_ROOT}/episodes_val.jsonl"
      "${SFT_ROOT}/val.jsonl" "${SFT_ROOT}/val_evaluation_index.jsonl"
      "${closed_loop_output}"
      --gpu "${GPU_SECOND}" --seed "${ACTIVE_SEED}"
      --max-candidates 16 --max-acquisitions 2 --max-new-tokens 64
      --bootstrap-repetitions 2000 --monitor-interval 5
    )
    if [[ "${STAGE}" == "closed_loop_smoke" ]]; then
      closed_loop_args=(
        "${MODEL}" "${CLOSED_LOOP_ADAPTER}"
        "${CLOSED_LOOP_ROOT}/states_val_step0.jsonl"
        "${CLOSED_LOOP_ROOT}/episodes_val.jsonl"
        "${SFT_ROOT}/val.jsonl" "${SFT_ROOT}/val_evaluation_index.jsonl"
        "${RUN_ROOT}/closed_loop_smoke"
        --gpu "${GPU_SECOND}" --seed "${seed_values[0]}"
        --max-candidates 16 --max-acquisitions 2 --max-new-tokens 64
        --bootstrap-repetitions 0 --limit 2 --monitor-interval 2
      )
    fi
    "${PYTHON}" scripts/launch_active_catalog_closed_loop.py \
      "${closed_loop_args[@]}"
    ;;
  joint_transition_train_smoke|joint_transition_train)
    [[ -n "${CLOSED_LOOP_ADAPTER}" ]] || {
      echo "CLOSED_LOOP_ADAPTER must explicitly name a validation-promoted adapter" >&2
      exit 18
    }
    require_file "${CLOSED_LOOP_ADAPTER}/adapter_config.json"
    require_file "${FULL_SELECTOR_STATES}"
    require_file "${TRAIN_VAL_EPISODES}"
    require_file "${SFT_ROOT}/train.jsonl"
    require_file "${SFT_ROOT}/train_evaluation_index.jsonl"
    require_free_gpu "${GPU_SECOND}"
    joint_output="${RUN_ROOT}/joint_transition_train_${SEED_TAG}"
    joint_extra=()
    if [[ "${STAGE}" == "joint_transition_train_smoke" ]]; then
      joint_output="${RUN_ROOT}/joint_transition_train_smoke"
      joint_extra=(--limit 2 --monitor-interval 2)
    fi
    [[ ! -e "${joint_output}" ]] || {
      echo "refusing to overwrite joint transition rollout: ${joint_output}" >&2
      exit 19
    }
    "${PYTHON}" scripts/launch_active_catalog_closed_loop.py \
      "${MODEL}" "${CLOSED_LOOP_ADAPTER}" \
      "${FULL_SELECTOR_STATES}" "${TRAIN_VAL_EPISODES}" \
      "${SFT_ROOT}/train.jsonl" "${SFT_ROOT}/train_evaluation_index.jsonl" \
      "${joint_output}" --gpu "${GPU_SECOND}" --seed "${ACTIVE_SEED}" \
      --split train --max-candidates 16 --max-acquisitions 2 \
      --max-new-tokens 64 --bootstrap-repetitions 0 "${joint_extra[@]}"
    ;;
  audit_joint_transitions)
    joint_train="${RUN_ROOT}/joint_transition_train_${SEED_TAG}/evaluation/joint_transitions.jsonl"
    joint_val="${RUN_ROOT}/closed_loop_${SEED_TAG}/evaluation/joint_transitions.jsonl"
    joint_audit="${RUN_ROOT}/joint_transition_audit_${SEED_TAG}.json"
    require_file "${joint_train}"
    require_file "${joint_val}"
    [[ ! -e "${joint_audit}" ]] || {
      echo "refusing to overwrite joint transition audit: ${joint_audit}" >&2
      exit 20
    }
    CUDA_VISIBLE_DEVICES= "${PYTHON}" \
      scripts/audit_active_catalog_joint_transitions.py \
      "${joint_train}" "${joint_val}" "${joint_audit}"
    ;;
  ground_joint_tools_smoke|ground_joint_tools)
    grounded_root="${GROUNDED_ROOT}"
    grounded_train_transitions="${RUN_ROOT}/joint_transition_train_${SEED_TAG}/evaluation/joint_transitions.jsonl"
    grounded_val_transitions="${RUN_ROOT}/closed_loop_${SEED_TAG}/evaluation/joint_transitions.jsonl"
    grounded_limit=()
    if [[ "${STAGE}" == "ground_joint_tools_smoke" ]]; then
      grounded_root="${RUN_ROOT}/joint_grounded_tools_smoke"
      grounded_train_transitions="${RUN_ROOT}/joint_transition_train_smoke/evaluation/joint_transitions.jsonl"
      grounded_val_transitions="${RUN_ROOT}/closed_loop_smoke/evaluation/joint_transitions.jsonl"
      grounded_limit=(--limit 2)
    fi
    [[ ! -e "${grounded_root}" ]] || {
      echo "refusing to overwrite grounded tool data: ${grounded_root}" >&2
      exit 21
    }
    require_file "${TRAIN_VAL_EPISODES}"
    require_file "${grounded_train_transitions}"
    require_file "${grounded_val_transitions}"
    if [[ "${STAGE}" == "ground_joint_tools" ]]; then
      require_file "${RUN_ROOT}/joint_transition_audit_${SEED_TAG}.json"
    fi
    root_map_args=()
    if [[ -n "${TOOL_ASSET_ROOT_MAP}" ]]; then
      root_map_args=(--asset-root-map "${TOOL_ASSET_ROOT_MAP}")
    fi
    CUDA_VISIBLE_DEVICES= "${PYTHON}" \
      scripts/build_active_catalog_grounded_tool_data.py \
      "${grounded_train_transitions}" \
      "${TRAIN_VAL_EPISODES}" "${grounded_root}/train" \
      --split train --out-size 256 --label-smoothing 0.05 \
      "${root_map_args[@]}" "${grounded_limit[@]}"
    CUDA_VISIBLE_DEVICES= "${PYTHON}" \
      scripts/build_active_catalog_grounded_tool_data.py \
      "${grounded_val_transitions}" \
      "${TRAIN_VAL_EPISODES}" "${grounded_root}/val" \
      --split val --out-size 256 --label-smoothing 0.05 \
      "${root_map_args[@]}" "${grounded_limit[@]}"
    ;;
  audit_grounded_joint_tools)
    grounded_root="${GROUNDED_ROOT}"
    require_file "${grounded_root}/train/train.jsonl"
    require_file "${grounded_root}/val/val.jsonl"
    grounded_audit="${grounded_root}/audit.json"
    [[ ! -e "${grounded_audit}" ]] || {
      echo "refusing to overwrite grounded tool audit: ${grounded_audit}" >&2
      exit 22
    }
    CUDA_VISIBLE_DEVICES= "${PYTHON}" \
      scripts/audit_active_catalog_grounded_tool_data.py \
      "${grounded_root}/train/train.jsonl" \
      "${grounded_root}/val/val.jsonl" "${grounded_audit}"
    ;;
  joint_tool_belief_smoke|joint_tool_belief_seed1|joint_tool_belief_seed|joint_tool_belief_gated_smoke|joint_tool_belief_gated_seed1|joint_tool_belief_gated_seed)
    grounded_root="${GROUNDED_ROOT}"
    belief_family="joint_tool_belief"
    reliability_args=()
    if [[ "${STAGE}" == joint_tool_belief_gated_* ]]; then
      belief_family="joint_tool_belief_gated"
      reliability_args=(--reliability-gate --gate-bias -1.5)
    fi
    tool_belief_output="${RUN_ROOT}/${belief_family}_${SEED_TAG}"
    tool_belief_args=(
      --device cpu --epochs 200 --batch-size 128 --learning-rate 3e-4
      --hidden-dim 128 --dropout 0.1 --patience 20
      --safety-margin 0.02 --min-quality-delta 0.01 --min-tool-delta 0.01
      --tool-contrastive-weight 0.2 --tool-contrastive-margin 0.05
      --seed "${ACTIVE_SEED}"
    )
    if [[ "${STAGE}" == "joint_tool_belief_smoke" || "${STAGE}" == "joint_tool_belief_gated_smoke" ]]; then
      grounded_root="${RUN_ROOT}/joint_grounded_tools_smoke"
      tool_belief_output="${RUN_ROOT}/${belief_family}_smoke"
      tool_belief_args=(
        --device cpu --epochs 1 --batch-size 2 --learning-rate 3e-4
        --hidden-dim 32 --dropout 0.0 --patience 1
        --safety-margin 1.0 --min-quality-delta 0.0 --min-tool-delta 0.0
        --tool-contrastive-weight 0.2 --seed "${seed_values[0]}"
      )
    else
      require_file "${grounded_root}/audit.json"
    fi
    require_file "${grounded_root}/train/train.jsonl"
    require_file "${grounded_root}/val/val.jsonl"
    [[ ! -e "${tool_belief_output}" ]] || {
      echo "refusing to overwrite Tool-Belief run: ${tool_belief_output}" >&2
      exit 23
    }
    CUDA_VISIBLE_DEVICES= "${PYTHON}" \
      scripts/train_post_acquisition_tool_belief.py \
      "${grounded_root}/train/train.jsonl" \
      "${grounded_root}/val/val.jsonl" "${tool_belief_output}" \
      "${tool_belief_args[@]}" "${reliability_args[@]}"
    ;;
  build_joint_tool_gate_features)
    grounded_root="${GROUNDED_ROOT}"
    # Keep CALL/SKIP supervision fixed when comparing belief updater variants.
    tool_belief_checkpoint="${RUN_ROOT}/${TOOL_GATE_LABEL_BELIEF_FAMILY}_${SEED_TAG}/best_promoted.pt"
    gate_feature_root="${RUN_ROOT}/${TOOL_GATE_FAMILY}_features_${SEED_TAG}"
    require_file "${grounded_root}/audit.json"
    require_file "${grounded_root}/train/train.jsonl"
    require_file "${grounded_root}/val/val.jsonl"
    require_file "${tool_belief_checkpoint}"
    [[ ! -e "${gate_feature_root}" ]] || {
      echo "refusing to overwrite tool-gate features: ${gate_feature_root}" >&2
      exit 24
    }
    "${PYTHON}" scripts/build_active_catalog_tool_gate_features.py \
      "${grounded_root}/train/train.jsonl" "${tool_belief_checkpoint}" \
      "${gate_feature_root}/train" --split train --device cpu \
      --correctness-weight 0.5 --tool-cost-weight 0.18 \
      --false-edit-weight 0.35 --missed-edit-weight 0.20
    "${PYTHON}" scripts/build_active_catalog_tool_gate_features.py \
      "${grounded_root}/val/val.jsonl" "${tool_belief_checkpoint}" \
      "${gate_feature_root}/val" --split val --device cpu \
      --correctness-weight 0.5 --tool-cost-weight 0.18 \
      --false-edit-weight 0.35 --missed-edit-weight 0.20
    ;;
  train_joint_tool_gate)
    gate_feature_root="${RUN_ROOT}/${TOOL_GATE_FAMILY}_features_${SEED_TAG}"
    gate_output="${RUN_ROOT}/${TOOL_GATE_FAMILY}_${SEED_TAG}"
    require_file "${gate_feature_root}/train/features.npy"
    require_file "${gate_feature_root}/train/records.jsonl"
    require_file "${gate_feature_root}/train/summary.json"
    require_file "${gate_feature_root}/val/features.npy"
    require_file "${gate_feature_root}/val/records.jsonl"
    require_file "${gate_feature_root}/val/summary.json"
    [[ ! -e "${gate_output}" ]] || {
      echo "refusing to overwrite selective tool gate: ${gate_output}" >&2
      exit 25
    }
    CUDA_VISIBLE_DEVICES= "${PYTHON}" scripts/train_visual_tool_gate.py \
      "${gate_feature_root}/train" "${gate_feature_root}/val" \
      "${gate_output}" --seed "${ACTIVE_SEED}" \
      --selection-objective proxy_utility --fit-weighting utility_risk \
      --max-call-rate 0.50 --min-oof-recall 0.10
    ;;
  closed_loop_forced_tools|closed_loop_selective_tools)
    [[ -n "${CLOSED_LOOP_ADAPTER}" ]] || {
      echo "CLOSED_LOOP_ADAPTER must name the promoted selector" >&2
      exit 26
    }
    tool_mode="forced"
    tool_branch_suffix=""
    if [[ "${TOOL_BELIEF_FAMILY}" != "joint_tool_belief" ]]; then
      tool_branch_suffix="_${TOOL_BELIEF_FAMILY}"
    fi
    tool_output="${RUN_ROOT}/closed_loop_forced_tools${tool_branch_suffix}_${SEED_TAG}"
    gate_args=()
    if [[ "${STAGE}" == "closed_loop_selective_tools" ]]; then
      tool_mode="selective"
      tool_output="${RUN_ROOT}/closed_loop_selective_tools${tool_branch_suffix}_${SEED_TAG}"
      require_file "${RUN_ROOT}/${TOOL_GATE_FAMILY}_${SEED_TAG}/gate.joblib"
      require_file "${RUN_ROOT}/${TOOL_GATE_FAMILY}_${SEED_TAG}/summary.json"
      gate_args=(
        --tool-gate "${RUN_ROOT}/${TOOL_GATE_FAMILY}_${SEED_TAG}/gate.joblib"
        --tool-gate-summary "${RUN_ROOT}/${TOOL_GATE_FAMILY}_${SEED_TAG}/summary.json"
      )
    fi
    require_file "${CLOSED_LOOP_ADAPTER}/adapter_config.json"
    require_file "${RUN_ROOT}/${TOOL_BELIEF_FAMILY}_${SEED_TAG}/best_promoted.pt"
    require_file "${CLOSED_LOOP_ROOT}/states_val_step0.jsonl"
    require_file "${CLOSED_LOOP_ROOT}/episodes_val.jsonl"
    require_file "${SFT_ROOT}/val.jsonl"
    require_file "${SFT_ROOT}/val_evaluation_index.jsonl"
    require_free_gpu "${GPU_SECOND}"
    [[ ! -e "${tool_output}" ]] || {
      echo "refusing to overwrite tool-agent rollout: ${tool_output}" >&2
      exit 27
    }
    "${PYTHON}" scripts/launch_active_catalog_closed_loop.py \
      "${MODEL}" "${CLOSED_LOOP_ADAPTER}" \
      "${CLOSED_LOOP_ROOT}/states_val_step0.jsonl" \
      "${CLOSED_LOOP_ROOT}/episodes_val.jsonl" \
      "${SFT_ROOT}/val.jsonl" "${SFT_ROOT}/val_evaluation_index.jsonl" \
      "${tool_output}" --gpu "${GPU_SECOND}" --seed "${ACTIVE_SEED}" \
      --max-candidates 16 --max-acquisitions 2 --max-new-tokens 64 \
      --bootstrap-repetitions 2000 --monitor-interval 5 \
      --tool-mode "${tool_mode}" \
      --tool-belief-checkpoint "${RUN_ROOT}/${TOOL_BELIEF_FAMILY}_${SEED_TAG}/best_promoted.pt" \
      --tool-artifact-root "${tool_output}/tool_artifacts" --tool-out-size 256 \
      "${gate_args[@]}"
    ;;
  compare_closed_loop_reliability_gate)
    ungated_root="${RUN_ROOT}/closed_loop_selective_tools_${SEED_TAG}/evaluation"
    gated_root="${RUN_ROOT}/closed_loop_selective_tools_joint_tool_belief_gated_${SEED_TAG}/evaluation"
    comparison="${RUN_ROOT}/closed_loop_reliability_gate_${SEED_TAG}.json"
    require_file "${ungated_root}/summary.json"
    require_file "${ungated_root}/traces.jsonl"
    require_file "${gated_root}/summary.json"
    require_file "${gated_root}/traces.jsonl"
    [[ ! -e "${comparison}" ]] || {
      echo "refusing to overwrite reliability-gate comparison: ${comparison}" >&2
      exit 28
    }
    CUDA_VISIBLE_DEVICES= "${PYTHON}" scripts/compare_closed_loop_reliability_gate.py \
      "${comparison}" \
      --ungated-summary "${ungated_root}/summary.json" \
      --ungated-traces "${ungated_root}/traces.jsonl" \
      --gated-summary "${gated_root}/summary.json" \
      --gated-traces "${gated_root}/traces.jsonl" \
      --repetitions 2000 --seed "${ACTIVE_SEED}"
    ;;
  compare_tool_belief_reliability_gate_three_seed)
    ungated_args=()
    gated_args=()
    for seed in "${seed_values[@]}"; do
      tag="$(seed_tag_for "${seed}")"
      ungated="${RUN_ROOT}/joint_tool_belief_${tag}/summary.json"
      gated="${RUN_ROOT}/joint_tool_belief_gated_${tag}/summary.json"
      require_file "${ungated}"
      require_file "${gated}"
      ungated_args+=("${ungated}")
      gated_args+=("${gated}")
    done
    output="${RUN_ROOT}/reliability_gate_three_seed/component_ablation.json"
    [[ ! -e "${output}" ]] || { echo "refusing to overwrite reliability component aggregate" >&2; exit 28; }
    "${PYTHON}" scripts/compare_tool_belief_reliability_gate.py "${output}" \
      --ungated "${ungated_args[@]}" --gated "${gated_args[@]}"
    ;;
  aggregate_reliability_gate_three_seed)
    aggregate_args=()
    for seed in "${seed_values[@]}"; do
      tag="$(seed_tag_for "${seed}")"
      comparison="${RUN_ROOT}/closed_loop_reliability_gate_${tag}.json"
      require_file "${comparison}"
      aggregate_args+=(--closed-loop "${seed}=${comparison}")
    done
    component="${RUN_ROOT}/reliability_gate_three_seed/component_ablation.json"
    output="${RUN_ROOT}/reliability_gate_three_seed/promotion.json"
    require_file "${component}"
    [[ ! -e "${output}" ]] || { echo "refusing to overwrite reliability promotion aggregate" >&2; exit 28; }
    "${PYTHON}" scripts/aggregate_reliability_gate_ablation.py "${output}" \
      --component "${component}" "${aggregate_args[@]}"
    ;;
  compare_closed_loop_tool_branches)
    no_tool="${RUN_ROOT}/closed_loop_${SEED_TAG}/evaluation/traces.jsonl"
    forced="${RUN_ROOT}/closed_loop_forced_tools_${SEED_TAG}/evaluation/traces.jsonl"
    selective="${RUN_ROOT}/closed_loop_selective_tools_${SEED_TAG}/evaluation/traces.jsonl"
    output="${RUN_ROOT}/closed_loop_selective_tools_${SEED_TAG}/paired_tool_branches.json"
    require_file "${no_tool}"
    require_file "${forced}"
    require_file "${selective}"
    [[ ! -e "${output}" ]] || {
      echo "refusing to overwrite tool-branch comparison: ${output}" >&2
      exit 28
    }
    "${PYTHON}" scripts/compare_active_catalog_closed_loop.py \
      "${output}" --candidate selective --repetitions 2000 \
      --seed "${ACTIVE_SEED}" \
      --records "selective=${selective}" \
      --records "forced=${forced}" --records "no_tool=${no_tool}"
    ;;
  assess_closed_loop_tool_branches)
    comparison="${RUN_ROOT}/closed_loop_selective_tools_${SEED_TAG}/paired_tool_branches.json"
    promotion="${RUN_ROOT}/closed_loop_selective_tools_${SEED_TAG}/promotion.json"
    require_file "${comparison}"
    [[ ! -e "${promotion}" ]] || {
      echo "refusing to overwrite tool-branch promotion: ${promotion}" >&2
      exit 29
    }
    "${PYTHON}" scripts/assess_active_catalog_tool_branch_promotion.py \
      "${comparison}" "${promotion}" \
      --false-edit-margin 0.02 --accuracy-margin 0.01
    ;;
  prepare_tool_branch_writebacks)
    input_root="${RUN_ROOT}/closed_loop_writeback_inputs"
    require_file "${RUN_ROOT}/closed_loop_${SEED_TAG}/evaluation/traces.jsonl"
    require_file "${RUN_ROOT}/closed_loop_forced_tools_${SEED_TAG}/evaluation/traces.jsonl"
    require_file "${RUN_ROOT}/closed_loop_selective_tools_${SEED_TAG}/evaluation/traces.jsonl"
    mkdir -p "${input_root}"
    for policy in no_tool forced_tools selective_tools; do
      [[ ! -e "${input_root}/${policy}${BRANCH_SUFFIX}.jsonl" ]] || {
        echo "refusing to overwrite tool writeback input: ${policy}" >&2
        exit 30
      }
    done
    "${PYTHON}" scripts/convert_active_catalog_closed_loop_for_writeback.py \
      "${RUN_ROOT}/closed_loop_${SEED_TAG}/evaluation/traces.jsonl" "${input_root}/no_tool${BRANCH_SUFFIX}.jsonl"
    "${PYTHON}" scripts/convert_active_catalog_closed_loop_for_writeback.py \
      "${RUN_ROOT}/closed_loop_forced_tools_${SEED_TAG}/evaluation/traces.jsonl" "${input_root}/forced_tools${BRANCH_SUFFIX}.jsonl"
    "${PYTHON}" scripts/convert_active_catalog_closed_loop_for_writeback.py \
      "${RUN_ROOT}/closed_loop_selective_tools_${SEED_TAG}/evaluation/traces.jsonl" "${input_root}/selective_tools${BRANCH_SUFFIX}.jsonl"
    ;;
  closed_loop_writeback_no_tool|closed_loop_writeback_forced_tools|closed_loop_writeback_selective_tools)
    require_file "${UPDATER_CHECKPOINT}"
    require_file "${TRAIN_VAL_EPISODES}"
    require_free_gpu "${GPU_SECOND}"
    policy="${STAGE#closed_loop_writeback_}"
    input="${RUN_ROOT}/closed_loop_writeback_inputs/${policy}${BRANCH_SUFFIX}.jsonl"
    output="${RUN_ROOT}/closed_loop_writeback_${policy}${BRANCH_SUFFIX}"
    require_file "${input}"
    root_map_args=()
    if [[ -n "${WRITEBACK_ASSET_ROOT_MAP}" ]]; then
      root_map_args=(--asset-root-map "${WRITEBACK_ASSET_ROOT_MAP}")
    fi
    "${PYTHON}" scripts/launch_active_catalog_writeback.py \
      "${UPDATER_CHECKPOINT}" "${TRAIN_VAL_EPISODES}" "${input}" "${output}" \
      --gpu "${GPU_SECOND}" --image-size 512 --threshold 0.5 \
      --protocol-name "sn7-active-catalog-${policy}-vector-writeback-v1" \
      --monitor-interval 5 "${root_map_args[@]}"
    ;;
  compare_tool_branch_writebacks)
    selective="${RUN_ROOT}/closed_loop_writeback_selective_tools${BRANCH_SUFFIX}/evaluation/writeback.jsonl"
    no_tool="${RUN_ROOT}/closed_loop_writeback_no_tool${BRANCH_SUFFIX}/evaluation/writeback.jsonl"
    forced="${RUN_ROOT}/closed_loop_writeback_forced_tools${BRANCH_SUFFIX}/evaluation/writeback.jsonl"
    output_root="${RUN_ROOT}/closed_loop_writeback_selective_tools${BRANCH_SUFFIX}"
    require_file "${selective}"
    require_file "${no_tool}"
    require_file "${forced}"
    for reference in no_tool forced_tools; do
      [[ ! -e "${output_root}/paired_${reference}_aoi.json" ]] || {
        echo "refusing to overwrite tool writeback comparison: ${reference}" >&2
        exit 31
      }
    done
    "${PYTHON}" scripts/compare_agent_writebacks.py \
      "${no_tool}" "${selective}" "${output_root}/paired_no_tool_aoi.json" \
      --bootstrap 2000 --seed "${ACTIVE_SEED}" --group-key aoi_id
    "${PYTHON}" scripts/compare_agent_writebacks.py \
      "${forced}" "${selective}" "${output_root}/paired_forced_tools_aoi.json" \
      --bootstrap 2000 --seed "${ACTIVE_SEED}" --group-key aoi_id
    ;;
  assess_tool_writeback_promotion)
    output_root="${RUN_ROOT}/closed_loop_writeback_selective_tools${BRANCH_SUFFIX}"
    require_file "${RUN_ROOT}/closed_loop_selective_tools_${SEED_TAG}/promotion.json"
    require_file "${output_root}/paired_no_tool_aoi.json"
    require_file "${output_root}/paired_forced_tools_aoi.json"
    [[ ! -e "${output_root}/promotion.json" ]] || {
      echo "refusing to overwrite tool writeback promotion" >&2
      exit 32
    }
    "${PYTHON}" scripts/assess_active_catalog_tool_writeback_promotion.py \
      "${RUN_ROOT}/closed_loop_selective_tools_${SEED_TAG}/promotion.json" \
      "${output_root}/paired_no_tool_aoi.json" \
      "${output_root}/paired_forced_tools_aoi.json" \
      "${output_root}/promotion.json" --replay-margin 0.01 --topology-margin 0.01
    ;;
  aggregate_tool_branches_three_seed)
    aggregate_args=()
    for seed in "${seed_values[@]}"; do
      tag="$(seed_tag_for "${seed}")"
      for label in no_tool forced selective; do
        case "${label}" in
          no_tool) trace="${RUN_ROOT}/closed_loop_${tag}/evaluation/traces.jsonl" ;;
          forced) trace="${RUN_ROOT}/closed_loop_forced_tools_${tag}/evaluation/traces.jsonl" ;;
          selective) trace="${RUN_ROOT}/closed_loop_selective_tools_${tag}/evaluation/traces.jsonl" ;;
        esac
        require_file "${trace}"
        aggregate_args+=(--records "${seed}:${label}=${trace}")
      done
    done
    output="${RUN_ROOT}/tool_branch_three_seed/paired_tool_branches.json"
    [[ ! -e "${output}" ]] || { echo "refusing to overwrite three-seed tool comparison" >&2; exit 33; }
    "${PYTHON}" scripts/aggregate_active_catalog_tool_branches.py \
      "${output}" "${aggregate_args[@]}" --repetitions 2000 --seed "${seed_values[0]}"
    ;;
  assess_tool_branches_three_seed)
    comparison="${RUN_ROOT}/tool_branch_three_seed/paired_tool_branches.json"
    promotion="${RUN_ROOT}/tool_branch_three_seed/promotion.json"
    require_file "${comparison}"
    [[ ! -e "${promotion}" ]] || { echo "refusing to overwrite three-seed promotion" >&2; exit 34; }
    "${PYTHON}" scripts/assess_active_catalog_tool_branch_promotion.py \
      "${comparison}" "${promotion}" --false-edit-margin 0.02 --accuracy-margin 0.01 \
      --min-seeds 3
    ;;
  aggregate_tool_writebacks_three_seed)
    no_tool_args=()
    forced_args=()
    selective_args=()
    for seed in "${seed_values[@]}"; do
      tag="$(seed_tag_for "${seed}")"
      no_tool="${RUN_ROOT}/closed_loop_writeback_no_tool_${tag}/evaluation/writeback.jsonl"
      forced="${RUN_ROOT}/closed_loop_writeback_forced_tools_${tag}/evaluation/writeback.jsonl"
      selective="${RUN_ROOT}/closed_loop_writeback_selective_tools_${tag}/evaluation/writeback.jsonl"
      require_file "${no_tool}"
      require_file "${forced}"
      require_file "${selective}"
      no_tool_args+=(--baseline "${seed}=${no_tool}")
      forced_args+=(--baseline "${seed}=${forced}")
      selective_args+=(--candidate "${seed}=${selective}")
    done
    output_root="${RUN_ROOT}/tool_writeback_three_seed"
    [[ ! -e "${output_root}" ]] || { echo "refusing to overwrite three-seed writeback aggregate" >&2; exit 35; }
    mkdir -p "${output_root}"
    "${PYTHON}" scripts/aggregate_active_catalog_tool_writebacks.py \
      "${output_root}/paired_no_tool_aoi.json" "${no_tool_args[@]}" "${selective_args[@]}" \
      --repetitions 2000 --seed "${seed_values[0]}"
    "${PYTHON}" scripts/aggregate_active_catalog_tool_writebacks.py \
      "${output_root}/paired_forced_tools_aoi.json" "${forced_args[@]}" "${selective_args[@]}" \
      --repetitions 2000 --seed "${seed_values[0]}"
    ;;
  assess_tool_writebacks_three_seed)
    output_root="${RUN_ROOT}/tool_writeback_three_seed"
    require_file "${RUN_ROOT}/tool_branch_three_seed/promotion.json"
    require_file "${output_root}/paired_no_tool_aoi.json"
    require_file "${output_root}/paired_forced_tools_aoi.json"
    [[ ! -e "${output_root}/promotion.json" ]] || { echo "refusing to overwrite three-seed writeback promotion" >&2; exit 36; }
    "${PYTHON}" scripts/assess_active_catalog_tool_writeback_promotion.py \
      "${RUN_ROOT}/tool_branch_three_seed/promotion.json" \
      "${output_root}/paired_no_tool_aoi.json" \
      "${output_root}/paired_forced_tools_aoi.json" \
      "${output_root}/promotion.json" --replay-margin 0.01 --topology-margin 0.01 \
      --min-seeds 3
    ;;
  compare_closed_loop_seed1)
    model_trace="${RUN_ROOT}/closed_loop_seed1/evaluation/traces.jsonl"
    baseline_root="${RUN_ROOT}/closed_loop_baselines"
    require_file "${model_trace}"
    for policy in always_stop cheapest clear_per_cost uncertainty_gate random shortlist_oracle_upper_bound; do
      require_file "${baseline_root}/${policy}.jsonl"
    done
    comparison_output="${RUN_ROOT}/closed_loop_seed1/paired_aoi_comparison.json"
    [[ ! -e "${comparison_output}" ]] || {
      echo "refusing to overwrite comparison: ${comparison_output}" >&2
      exit 13
    }
    "${PYTHON}" scripts/compare_active_catalog_closed_loop.py \
      "${comparison_output}" --candidate qwen --repetitions 2000 \
      --seed "${seed_values[0]}" \
      --records "qwen=${model_trace}" \
      --records "always_stop=${baseline_root}/always_stop.jsonl" \
      --records "cheapest=${baseline_root}/cheapest.jsonl" \
      --records "clear_per_cost=${baseline_root}/clear_per_cost.jsonl" \
      --records "uncertainty_gate=${baseline_root}/uncertainty_gate.jsonl" \
      --records "random=${baseline_root}/random.jsonl" \
      --records "shortlist_oracle_upper_bound=${baseline_root}/shortlist_oracle_upper_bound.jsonl"
    ;;
  writeback_preflight)
    require_file "${UPDATER_CHECKPOINT}"
    require_file "${TRAIN_VAL_EPISODES}"
    "${PYTHON}" -c \
      'import affine, rasterio, shapely, torch; print("writeback dependencies passed")'
    ;;
  prepare_closed_loop_writebacks)
    model_trace="${RUN_ROOT}/closed_loop_seed1/evaluation/traces.jsonl"
    baseline_trace="${RUN_ROOT}/closed_loop_baselines/${WRITEBACK_BASELINE_POLICY}.jsonl"
    require_file "${model_trace}"
    require_file "${baseline_trace}"
    writeback_input_root="${RUN_ROOT}/closed_loop_writeback_inputs"
    [[ ! -e "${writeback_input_root}" ]] || {
      echo "refusing to overwrite writeback inputs: ${writeback_input_root}" >&2
      exit 14
    }
    mkdir -p "${writeback_input_root}"
    "${PYTHON}" scripts/convert_active_catalog_closed_loop_for_writeback.py \
      "${model_trace}" "${writeback_input_root}/qwen.jsonl"
    "${PYTHON}" scripts/convert_active_catalog_closed_loop_for_writeback.py \
      "${baseline_trace}" "${writeback_input_root}/${WRITEBACK_BASELINE_POLICY}.jsonl"
    ;;
  closed_loop_writeback_smoke|closed_loop_writeback_qwen|closed_loop_writeback_baseline)
    require_file "${UPDATER_CHECKPOINT}"
    require_file "${TRAIN_VAL_EPISODES}"
    require_free_gpu "${GPU_SECOND}"
    writeback_input_root="${RUN_ROOT}/closed_loop_writeback_inputs"
    writeback_policy="qwen"
    writeback_output="${RUN_ROOT}/closed_loop_writeback_qwen"
    writeback_limit=()
    if [[ "${STAGE}" == "closed_loop_writeback_smoke" ]]; then
      writeback_output="${RUN_ROOT}/closed_loop_writeback_smoke"
      writeback_limit=(--limit 2)
    elif [[ "${STAGE}" == "closed_loop_writeback_baseline" ]]; then
      writeback_policy="${WRITEBACK_BASELINE_POLICY}"
      writeback_output="${RUN_ROOT}/closed_loop_writeback_${WRITEBACK_BASELINE_POLICY}"
    fi
    require_file "${writeback_input_root}/${writeback_policy}.jsonl"
    root_map_args=()
    if [[ -n "${WRITEBACK_ASSET_ROOT_MAP}" ]]; then
      root_map_args=(--asset-root-map "${WRITEBACK_ASSET_ROOT_MAP}")
    fi
    "${PYTHON}" scripts/launch_active_catalog_writeback.py \
      "${UPDATER_CHECKPOINT}" "${TRAIN_VAL_EPISODES}" \
      "${writeback_input_root}/${writeback_policy}.jsonl" "${writeback_output}" \
      --gpu "${GPU_SECOND}" --image-size 512 --threshold 0.5 \
      --protocol-name sn7-active-catalog-vector-writeback-v1 \
      --monitor-interval 5 "${root_map_args[@]}" "${writeback_limit[@]}"
    ;;
  compare_closed_loop_writeback)
    candidate="${RUN_ROOT}/closed_loop_writeback_qwen/evaluation/writeback.jsonl"
    baseline="${RUN_ROOT}/closed_loop_writeback_${WRITEBACK_BASELINE_POLICY}/evaluation/writeback.jsonl"
    require_file "${candidate}"
    require_file "${baseline}"
    output="${RUN_ROOT}/closed_loop_writeback_qwen/paired_${WRITEBACK_BASELINE_POLICY}_aoi.json"
    [[ ! -e "${output}" ]] || {
      echo "refusing to overwrite writeback comparison: ${output}" >&2
      exit 15
    }
    "${PYTHON}" scripts/compare_agent_writebacks.py \
      "${baseline}" "${candidate}" "${output}" \
      --bootstrap 2000 --seed "${seed_values[0]}" --group-key aoi_id
    ;;
  assess_closed_loop_promotion)
    closed_loop_comparison="${RUN_ROOT}/closed_loop_seed1/paired_aoi_comparison.json"
    writeback_comparison="${RUN_ROOT}/closed_loop_writeback_qwen/paired_${WRITEBACK_BASELINE_POLICY}_aoi.json"
    promotion_output="${RUN_ROOT}/closed_loop_writeback_qwen/promotion_${WRITEBACK_BASELINE_POLICY}.json"
    require_file "${closed_loop_comparison}"
    require_file "${writeback_comparison}"
    [[ ! -e "${promotion_output}" ]] || {
      echo "refusing to overwrite promotion result: ${promotion_output}" >&2
      exit 16
    }
    "${PYTHON}" scripts/assess_active_catalog_closed_loop_promotion.py \
      "${closed_loop_comparison}" "${writeback_comparison}" \
      "${promotion_output}" --candidate qwen \
      --baseline "${WRITEBACK_BASELINE_POLICY}" \
      --false-edit-margin 0.02 --topology-margin 0.01
    ;;
  render_closed_loop_examples)
    model_trace="${RUN_ROOT}/closed_loop_seed1/evaluation/traces.jsonl"
    writeback_trace="${RUN_ROOT}/closed_loop_writeback_qwen/evaluation/writeback.jsonl"
    figure_output="${RUN_ROOT}/closed_loop_writeback_qwen/figures"
    require_file "${TRAIN_VAL_EPISODES}"
    require_file "${model_trace}"
    require_file "${writeback_trace}"
    [[ ! -e "${figure_output}" ]] || {
      echo "refusing to overwrite closed-loop figures: ${figure_output}" >&2
      exit 17
    }
    root_map_args=()
    if [[ -n "${WRITEBACK_ASSET_ROOT_MAP}" ]]; then
      root_map_args=(--asset-root-map "${WRITEBACK_ASSET_ROOT_MAP}")
    fi
    CUDA_VISIBLE_DEVICES= "${PYTHON}" \
      scripts/render_active_catalog_closed_loop_examples.py \
      "${TRAIN_VAL_EPISODES}" "${model_trace}" "${writeback_trace}" \
      "${figure_output}" --count 4 "${root_map_args[@]}"
    ;;
  *)
    echo "unknown stage: ${STAGE}" >&2
    exit 2
    ;;
esac
