#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog}"
DATA_ROOT="${DATA_ROOT:-${STORAGE_ROOT}/processed/sn7_v1/agent/sequential_selector_v1}"
SFT_ROOT="${SFT_ROOT:-${DATA_ROOT}/full/active_catalog_sft_v4}"
SENTINEL="${SFT_ROOT}/balanced_sentinel_v1"
PYTHON="${ACTIVEMAP_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
QWEN="${QWEN_MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
GEMMA="${GEMMA_MODEL:-/home/wh/hf_models/gemma-3-4b-it}"
POLL_SECONDS="${POLL_SECONDS:-60}"
CONTROL_ROOT="${RUN_ROOT}/long_matrix_control"

cd "${PROJECT_ROOT}"
# shellcheck source=/dev/null
source scripts/server_hdpi_env.sh
# shellcheck source=/dev/null
source scripts/assert_allowed_gpu.sh
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${CONTROL_ROOT}"

for gpu in 3 4 5 7; do activemap_assert_allowed_gpu "${gpu}"; done

wait_for_completed_run() {
  local family="$1" seed="$2"
  local state="${RUN_ROOT}/${family}/seed${seed}/run_state.json"
  while true; do
    if [[ -s "${state}" ]]; then
      status="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status","unknown"))' "${state}")"
      case "${status}" in
        completed) return 0 ;;
        failed) echo "training failed: ${state}" >&2; return 7 ;;
      esac
    fi
    echo "$(date -Is) waiting for ${family} seed ${seed}"
    sleep "${POLL_SECONDS}"
  done
}

wait_for_gpu() {
  local gpu="$1"
  while [[ -n "$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
    echo "$(date -Is) waiting for GPU ${gpu}"
    sleep "${POLL_SECONDS}"
  done
}

select_checkpoint() {
  local family="$1" model="$2" seed="$3" gpu="$4"
  local seed_root="${RUN_ROOT}/${family}/seed${seed}"
  local candidate_root="${seed_root}/sentinel_checkpoint_selection"
  local selection="${candidate_root}/selection.json"
  if [[ -s "${selection}" ]]; then
    "${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r.get("passed") and not r.get("test_assets_read"); print(r["selected"]["adapter"])' "${selection}"
    return
  fi
  mkdir -p "${candidate_root}"
  local -a adapters=() selector_args=()
  [[ -s "${seed_root}/diagnostic_adapters/checkpoint-1000/adapter_model.safetensors" ]] && \
    adapters+=("${seed_root}/diagnostic_adapters/checkpoint-1000")
  while IFS= read -r adapter; do adapters+=("${adapter}"); done < <(
    find "${seed_root}/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V
  )
  adapters+=("${seed_root}/final")
  local ordinal=0 adapter label output
  for adapter in "${adapters[@]}"; do
    [[ -s "${adapter}/adapter_model.safetensors" ]] || continue
    ordinal=$((ordinal + 1))
    label="candidate-${ordinal}-$(basename "${adapter}")"
    output="${candidate_root}/${label}"
    if [[ ! -s "${output}/summary.json" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/evaluate_active_catalog_selector.py \
        "${model}" "${adapter}" "${SENTINEL}/val.jsonl" \
        "${SENTINEL}/val_evaluation_index.jsonl" "${output}" \
        --device cuda:0 --seed "${seed}" --bootstrap-repetitions 500 >&2
    fi
    selector_args+=(--candidate "${label}" "${adapter}" "${output}/traces.jsonl")
  done
  (( ordinal >= 2 )) || { echo "fewer than two checkpoint candidates: ${seed_root}" >&2; return 8; }
  "${PYTHON}" scripts/select_active_catalog_sentinel_checkpoint.py "${selection}" "${selector_args[@]}" >&2
  "${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r.get("passed") and not r.get("test_assets_read"); print(r["selected"]["adapter"])' "${selection}"
}

evaluate_family() {
  local family="$1" model="$2" seed="$3" gpu="$4"
  local seed_root="${RUN_ROOT}/${family}/seed${seed}"
  local output="${seed_root}/active_catalog_val"
  wait_for_completed_run "${family}" "${seed}"
  wait_for_gpu "${gpu}"
  if [[ -s "${output}/summary.json" ]]; then
    echo "reuse completed evaluation: ${output}"
    return
  fi
  [[ ! -e "${output}" ]] || { echo "incomplete evaluation exists: ${output}" >&2; return 9; }
  adapter="$(select_checkpoint "${family}" "${model}" "${seed}" "${gpu}")"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/evaluate_active_catalog_selector.py \
    "${model}" "${adapter}" "${SFT_ROOT}/val.jsonl" \
    "${SFT_ROOT}/val_evaluation_index.jsonl" "${output}" \
    --device cuda:0 --seed "${seed}" --bootstrap-repetitions 2000
}

run_gpu3() {
  evaluate_family qwen3vl4b_unweighted_seed1 "${QWEN}" 20260717 3
  evaluate_family qwen3vl4b_seed1_eval500 "${QWEN}" 20260717 3
}

run_gpu3 & pid3=$!
evaluate_family qwen3vl4b_weighted_replication_seed3 "${QWEN}" 20260719 4 & pid4=$!
evaluate_family gemma3_4b_weighted_seed1 "${GEMMA}" 20260717 5 & pid5=$!
evaluate_family qwen3vl4b_weighted_replication_seed2 "${QWEN}" 20260718 7 & pid7=$!

failed=0
for pid in "${pid3}" "${pid4}" "${pid5}" "${pid7}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || { echo "one or more matrix evaluations failed" >&2; exit 10; }

weighted1="${RUN_ROOT}/qwen3vl4b_seed1_eval500/seed20260717/active_catalog_val/traces.jsonl"
weighted2="${RUN_ROOT}/qwen3vl4b_weighted_replication_seed2/seed20260718/active_catalog_val/traces.jsonl"
weighted3="${RUN_ROOT}/qwen3vl4b_weighted_replication_seed3/seed20260719/active_catalog_val/traces.jsonl"
unweighted="${RUN_ROOT}/qwen3vl4b_unweighted_seed1/seed20260717/active_catalog_val/traces.jsonl"
gemma="${RUN_ROOT}/gemma3_4b_weighted_seed1/seed20260717/active_catalog_val/traces.jsonl"

sampling="${RUN_ROOT}/seed1_sampling_ablation.json"
if [[ ! -s "${sampling}" ]]; then
  "${PYTHON}" scripts/compare_active_catalog_sampling_ablation.py \
    "${weighted1}" "${unweighted}" "${sampling}" --repetitions 2000 --seed 20260717
fi

aggregate="${RUN_ROOT}/qwen3vl4b_weighted_matrix_three_seed/active_catalog_aoi_bootstrap.json"
if [[ ! -s "${aggregate}" ]]; then
  "${PYTHON}" scripts/aggregate_active_catalog_seeds.py "${aggregate}" \
    --records "20260717=${weighted1}" --records "20260718=${weighted2}" \
    --records "20260719=${weighted3}" --repetitions 2000 --bootstrap-seed 20260717
fi

backbone="${RUN_ROOT}/backbone_matrix/qwen_vs_gemma_seed20260717.json"
if [[ ! -s "${backbone}" ]]; then
  "${PYTHON}" scripts/compare_active_catalog_backbones.py \
    "${gemma}" "${weighted1}" "${backbone}" \
    --candidate-name gemma-3-4b-it --reference-name qwen3-vl-4b-instruct \
    --repetitions 2000 --seed 20260717
fi

printf 'completed_at=%s\nsampling=%s\naggregate=%s\nbackbone=%s\ntest_assets_read=false\n' \
  "$(date -Is)" "${sampling}" "${aggregate}" "${backbone}" \
  >"${CONTROL_ROOT}/complete.txt"
