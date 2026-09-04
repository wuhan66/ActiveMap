#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
MODEL="/home/wh/hf_models/Qwen3-VL-4B-Instruct"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"
DATA="${STORAGE_ROOT}/processed/sn7_v1/agent/sequential_selector_v1/full"
RL_DATA="${RUN}/active_catalog_rl_states"
EPISODES="${DATA}/closed_loop_v1/episodes_val.jsonl"
UPDATER="${STORAGE_ROOT}/runs/updater/v4_hierarchical_vector_change_scratch_seed20260716/best_quality.pt"
SELECTION="${RUN}/conservative_rl_selection_n512_seed20260718/selection.json"
LOG_ROOT="${STORAGE_ROOT}/logs/sn7_promoted_rl_three_seed"
POLL_SECONDS="${POLL_SECONDS:-60}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
mkdir -p "${LOG_ROOT}"

until [[ -s "${SELECTION}" ]]; do
  sleep "${POLL_SECONDS}"
done

decision="$("${PYTHON}" -c \
  "import json; print(json.load(open('${SELECTION}'))['decision'])")"
selected="$("${PYTHON}" -c \
  "import json; print(json.load(open('${SELECTION}'))['selected_candidate'])")"
if [[ "${decision}" == "stop_rl" ]]; then
  echo "Frozen gate selected stop_rl; no replication is launched."
  exit 0
fi
[[ "${decision}" == "promote_one" && "${selected}" != "None" ]] || {
  echo "invalid frozen RL selection: decision=${decision}, selected=${selected}" >&2
  exit 2
}

case "${selected}" in
  rl_lr2p5_kl005_bal50)
    learning_rate="2.5e-6"; kl_beta="0.05"; acquire_fraction="0.50"
    seed2_trace="${RUN}/closed_loop_rl_conservative_lr2p5e6_kl005_bal50_n4096_seed20260718_n512/evaluation/traces.jsonl"
    ;;
  rl_lr2p5_kl010_bal50)
    learning_rate="2.5e-6"; kl_beta="0.10"; acquire_fraction="0.50"
    seed2_trace="${RUN}/closed_loop_rl_conservative_lr2p5e6_kl010_bal50_n4096_seed20260718_n512/evaluation/traces.jsonl"
    ;;
  rl_lr5_kl010_bal50)
    learning_rate="5e-6"; kl_beta="0.10"; acquire_fraction="0.50"
    seed2_trace="${RUN}/closed_loop_rl_conservative_lr5e6_kl010_bal50_n4096_seed20260718_n512/evaluation/traces.jsonl"
    ;;
  rl_lr5_kl005_bal25)
    learning_rate="5e-6"; kl_beta="0.05"; acquire_fraction="0.25"
    seed2_trace="${RUN}/closed_loop_rl_conservative_lr5e6_kl005_bal25_n4096_seed20260718_n512/evaluation/traces.jsonl"
    ;;
  rl_lr2p5_kl010_bal25)
    learning_rate="2.5e-6"; kl_beta="0.10"; acquire_fraction="0.25"
    seed2_trace="${RUN}/closed_loop_rl_conservative_lr2p5e6_kl010_bal25_n4096_seed20260718_n512/evaluation/traces.jsonl"
    ;;
  *)
    echo "unrecognized selected candidate: ${selected}" >&2
    exit 3
    ;;
esac

gpu_is_idle() {
  local gpu="$1"
  local pids
  pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${pids//[[:space:]]/}" ]]
}

adapter_for_seed() {
  case "$1" in
    20260717)
      echo "${RUN}/qwen3vl4b_seed1_eval500/seed20260717/final"
      ;;
    20260719)
      echo "${RUN}/qwen3vl4b_weighted_replication_seed3/seed20260719/final"
      ;;
    *)
      return 1
      ;;
  esac
}

run_writeback_pair() {
  local model_seed="$1"
  local gpu="$2"
  local rl_trace="$3"
  local sft_trace="$4"
  local prefix="${RUN}/promoted_${selected}_seed${model_seed}"
  local rl_input="${prefix}_writeback_input.jsonl"
  local sft_input="${RUN}/sft_seed${model_seed}_promotion_writeback_input.jsonl"
  local rl_output="${prefix}_writeback"
  local sft_output="${RUN}/sft_seed${model_seed}_promotion_writeback"
  local comparison="${prefix}_writeback_paired_vs_sft.json"

  if [[ ! -s "${rl_input}" ]]; then
    "${PYTHON}" scripts/convert_active_catalog_closed_loop_for_writeback.py \
      "${rl_trace}" "${rl_input}"
  fi
  if [[ ! -s "${sft_input}" ]]; then
    "${PYTHON}" scripts/convert_active_catalog_closed_loop_for_writeback.py \
      "${sft_trace}" "${sft_input}"
  fi
  until gpu_is_idle "${gpu}"; do
    sleep "${POLL_SECONDS}"
  done
  if [[ ! -s "${sft_output}/evaluation/summary.json" ]]; then
    [[ ! -e "${sft_output}" ]] || {
      echo "refusing partial SFT writeback: ${sft_output}" >&2
      return 8
    }
    "${PYTHON}" scripts/launch_active_catalog_writeback.py \
      "${UPDATER}" "${EPISODES}" "${sft_input}" "${sft_output}" \
      --gpu "${gpu}" --image-size 512 --threshold 0.5 \
      --protocol-name "sn7-sft-seed${model_seed}-promotion-writeback-v1" \
      --monitor-interval 5
  fi
  until gpu_is_idle "${gpu}"; do
    sleep "${POLL_SECONDS}"
  done
  if [[ ! -s "${rl_output}/evaluation/summary.json" ]]; then
    [[ ! -e "${rl_output}" ]] || {
      echo "refusing partial promoted RL writeback: ${rl_output}" >&2
      return 9
    }
    "${PYTHON}" scripts/launch_active_catalog_writeback.py \
      "${UPDATER}" "${EPISODES}" "${rl_input}" "${rl_output}" \
      --gpu "${gpu}" --image-size 512 --threshold 0.5 \
      --protocol-name "sn7-promoted-${selected}-seed${model_seed}-writeback-v1" \
      --monitor-interval 5
  fi
  if [[ ! -s "${comparison}" ]]; then
    "${PYTHON}" scripts/compare_agent_writebacks.py \
      "${sft_output}/evaluation/writeback.jsonl" \
      "${rl_output}/evaluation/writeback.jsonl" \
      "${comparison}" --bootstrap 2000 --seed "${model_seed}" \
      --group-key aoi_id
  fi
}

run_seed() {
  local model_seed="$1"
  local gpu="$2"
  local sft train_root train_result rl_adapter rl_rollout sft_rollout
  sft="$(adapter_for_seed "${model_seed}")"
  train_root="${RUN}/promoted_${selected}_sftseed${model_seed}_n4096"
  train_result="${train_root}/process_result.json"
  rl_adapter="${train_root}/seed${model_seed}/final"
  rl_rollout="${RUN}/closed_loop_promoted_${selected}_seed${model_seed}_n512"
  sft_rollout="${RUN}/closed_loop_sft_seed${model_seed}_n512"

  [[ -s "${sft}/adapter_config.json" ]] || {
    echo "missing independent SFT adapter: ${sft}" >&2
    return 4
  }
  until gpu_is_idle "${gpu}"; do
    sleep "${POLL_SECONDS}"
  done
  if [[ ! -s "${train_result}" ]]; then
    [[ ! -e "${train_root}" ]] || {
      echo "refusing partial promotion training: ${train_root}" >&2
      return 5
    }
    "${PYTHON}" scripts/launch_active_catalog_vlm_rl.py \
      "${sft}" "${RL_DATA}/train.jsonl" "${RL_DATA}/val.jsonl" "${train_root}" \
      --gpu "${gpu}" --seed "${model_seed}" --epochs 1 \
      --learning-rate "${learning_rate}" --gradient-accumulation 16 \
      --max-length 2048 --action-limit 4 --kl-beta "${kl_beta}" \
      --entropy-weight 0.01 --temperature 1.0 --utility-scale 30 \
      --acquire-fraction "${acquire_fraction}" --pairwise-weight 0.1 \
      --pairwise-margin 0.2 --length-normalize-action-score \
      --logging-steps 10 --eval-steps 64 --save-steps 64 \
      --max-train-samples 4096 --max-eval-samples 512 --monitor-interval 5
  fi
  "${PYTHON}" -c \
    "import json; assert json.load(open('${train_result}'))['status'] == 'completed'"

  until gpu_is_idle "${gpu}"; do
    sleep "${POLL_SECONDS}"
  done
  if [[ ! -s "${rl_rollout}/evaluation/traces.jsonl" ]]; then
    [[ ! -e "${rl_rollout}" ]] || {
      echo "refusing partial RL rollout: ${rl_rollout}" >&2
      return 6
    }
    "${PYTHON}" scripts/launch_active_catalog_closed_loop.py \
      "${MODEL}" "${rl_adapter}" \
      "${DATA}/closed_loop_v1/states_val_step0.jsonl" \
      "${DATA}/closed_loop_v1/episodes_val.jsonl" \
      "${DATA}/active_catalog_sft_v4/val.jsonl" \
      "${DATA}/active_catalog_sft_v4/val_evaluation_index.jsonl" \
      "${rl_rollout}" --gpu "${gpu}" --seed "${model_seed}" \
      --max-candidates 16 --max-acquisitions 2 --max-new-tokens 64 \
      --bootstrap-repetitions 500 --limit 512 --monitor-interval 5
  fi

  until gpu_is_idle "${gpu}"; do
    sleep "${POLL_SECONDS}"
  done
  if [[ ! -s "${sft_rollout}/evaluation/traces.jsonl" ]]; then
    [[ ! -e "${sft_rollout}" ]] || {
      echo "refusing partial SFT rollout: ${sft_rollout}" >&2
      return 7
    }
    "${PYTHON}" scripts/launch_active_catalog_closed_loop.py \
      "${MODEL}" "${sft}" \
      "${DATA}/closed_loop_v1/states_val_step0.jsonl" \
      "${DATA}/closed_loop_v1/episodes_val.jsonl" \
      "${DATA}/active_catalog_sft_v4/val.jsonl" \
      "${DATA}/active_catalog_sft_v4/val_evaluation_index.jsonl" \
      "${sft_rollout}" --gpu "${gpu}" --seed "${model_seed}" \
      --max-candidates 16 --max-acquisitions 2 --max-new-tokens 64 \
      --bootstrap-repetitions 500 --limit 512 --monitor-interval 5
  fi

  "${PYTHON}" scripts/compare_active_catalog_closed_loop.py \
    "${RUN}/promoted_${selected}_seed${model_seed}_paired_vs_sft.json" \
    --reference sft --repetitions 2000 --seed "${model_seed}" \
    --records "sft=${sft_rollout}/evaluation/traces.jsonl" \
    --records "rl=${rl_rollout}/evaluation/traces.jsonl"

  run_writeback_pair \
    "${model_seed}" "${gpu}" \
    "${rl_rollout}/evaluation/traces.jsonl" \
    "${sft_rollout}/evaluation/traces.jsonl"
}

run_seed 20260717 0 >"${LOG_ROOT}/seed20260717.log" 2>&1 &
pid1="$!"
run_seed 20260719 1 >"${LOG_ROOT}/seed20260719.log" 2>&1 &
pid3="$!"
status=0
for pid in "${pid1}" "${pid3}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
[[ "${status}" -eq 0 ]] || exit "${status}"

until gpu_is_idle 2; do
  sleep "${POLL_SECONDS}"
done
run_writeback_pair \
  20260718 2 \
  "${seed2_trace}" \
  "${RUN}/closed_loop_sft_seed20260718_n512/evaluation/traces.jsonl"

"${PYTHON}" scripts/aggregate_active_catalog_paired_policy_seeds.py \
  "${RUN}/promoted_${selected}_three_seed_vs_matched_sft.json" \
  --pair "20260717=${RUN}/closed_loop_promoted_${selected}_seed20260717_n512/evaluation/traces.jsonl,${RUN}/closed_loop_sft_seed20260717_n512/evaluation/traces.jsonl" \
  --pair "20260718=${seed2_trace},${RUN}/closed_loop_sft_seed20260718_n512/evaluation/traces.jsonl" \
  --pair "20260719=${RUN}/closed_loop_promoted_${selected}_seed20260719_n512/evaluation/traces.jsonl,${RUN}/closed_loop_sft_seed20260719_n512/evaluation/traces.jsonl" \
  --repetitions 5000 --seed 20260729

"${PYTHON}" scripts/aggregate_agent_writeback_pairs.py \
  "${RUN}/promoted_${selected}_three_seed_writeback_vs_matched_sft.json" \
  --pair "20260717=${RUN}/sft_seed20260717_promotion_writeback/evaluation/writeback.jsonl,${RUN}/promoted_${selected}_seed20260717_writeback/evaluation/writeback.jsonl" \
  --pair "20260718=${RUN}/sft_seed20260718_promotion_writeback/evaluation/writeback.jsonl,${RUN}/promoted_${selected}_seed20260718_writeback/evaluation/writeback.jsonl" \
  --pair "20260719=${RUN}/sft_seed20260719_promotion_writeback/evaluation/writeback.jsonl,${RUN}/promoted_${selected}_seed20260719_writeback/evaluation/writeback.jsonl" \
  --repetitions 5000 --seed 20260729

"${PYTHON}" scripts/assess_promoted_rl_writeback.py \
  "${RUN}/promoted_${selected}_three_seed_writeback_vs_matched_sft.json" \
  "${RUN}/promoted_${selected}_final_writeback_assessment.json" \
  --min-safety-utility-ci-low 0.0 \
  --min-raster-gain-ci-low 0.0 \
  --max-false-edit-ci-high 0.0 \
  --min-topology-observed 0.0
