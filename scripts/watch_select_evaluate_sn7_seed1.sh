#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog}"
DATA_ROOT="${DATA_ROOT:-${STORAGE_ROOT}/processed/sn7_v1/agent/sequential_selector_v1}"
SFT_ROOT="${SFT_ROOT:-${DATA_ROOT}/full/active_catalog_sft_v4}"
SENTINEL="${SFT_ROOT}/balanced_sentinel_v1"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
PYTHON="${ACTIVEMAP_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
SEED="${SEED:-20260717}"
GPU_WEIGHTED="${SEED1_SELECTION_GPU:-6}"
GPU_UNWEIGHTED="${SEED1_FULL_EVAL_SECOND_GPU:-7}"
POLL_SECONDS="${POLL_SECONDS:-60}"
WEIGHTED_FAMILY="${WEIGHTED_FAMILY:-qwen3vl4b_seed1_eval500}"
UNWEIGHTED_FAMILY="${UNWEIGHTED_FAMILY:-qwen3vl4b_unweighted_seed1}"

cd "$PROJECT_ROOT"
# shellcheck source=/dev/null
source scripts/server_hdpi_env.sh
# shellcheck source=/dev/null
source scripts/assert_allowed_gpu.sh
activemap_assert_allowed_gpu "$GPU_WEIGHTED"
activemap_assert_allowed_gpu "$GPU_UNWEIGHTED"
[[ "$GPU_WEIGHTED" != "$GPU_UNWEIGHTED" ]] || { echo "two distinct full-evaluation GPUs required" >&2; exit 2; }

for family in "$WEIGHTED_FAMILY" "$UNWEIGHTED_FAMILY"; do
  state="${RUN_ROOT}/${family}/seed${SEED}/run_state.json"
  while [[ ! -s "$state" ]] || ! "$PYTHON" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("status")=="completed" else 1)' "$state"; do
    echo "$(date -Is) waiting for completed training: ${family}"
    sleep "$POLL_SECONDS"
  done
done
while [[ ! -s "${RUN_ROOT}/seed1_sampling_ablation_step1000_sentinel.json" ]]; do
  echo "$(date -Is) waiting for step-1000 balanced sentinel"
  sleep "$POLL_SECONDS"
done
while screen -ls 2>/dev/null | grep -q activemap_muno21_replicates_after_sn7_step500; do
  echo "$(date -Is) waiting for MUNO21 replicate allocation"
  sleep "$POLL_SECONDS"
done
for gpu in "$GPU_WEIGHTED" "$GPU_UNWEIGHTED"; do
  while [[ -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
    echo "$(date -Is) waiting for GPU ${gpu}"
    sleep "$POLL_SECONDS"
  done
done

evaluate_candidates() {
  local family="$1"
  local seed_root="${RUN_ROOT}/${family}/seed${SEED}"
  local candidate_root="${seed_root}/sentinel_checkpoint_selection"
  local selection="${candidate_root}/selection.json"
  mkdir -p "$candidate_root"
  local -a selector_args=()
  local -a adapters=()
  [[ -d "${seed_root}/diagnostic_adapters/checkpoint-1000" ]] && adapters+=("${seed_root}/diagnostic_adapters/checkpoint-1000")
  while IFS= read -r adapter; do adapters+=("$adapter"); done < <(find "${seed_root}/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
  adapters+=("${seed_root}/final")
  local ordinal=0
  for adapter in "${adapters[@]}"; do
    [[ -s "${adapter}/adapter_model.safetensors" ]] || continue
    ordinal=$((ordinal + 1))
    label="candidate-${ordinal}-$(basename "$adapter")"
    output="${candidate_root}/${label}"
    if [[ ! -s "${output}/summary.json" ]]; then
      CUDA_VISIBLE_DEVICES="$GPU_WEIGHTED" "$PYTHON" scripts/evaluate_active_catalog_selector.py \
        "$MODEL" "$adapter" "${SENTINEL}/val.jsonl" \
        "${SENTINEL}/val_evaluation_index.jsonl" "$output" \
        --device cuda:0 --seed "$SEED" --bootstrap-repetitions 500
    fi
    selector_args+=(--candidate "$label" "$adapter" "${output}/traces.jsonl")
  done
  (( ${#selector_args[@]} >= 4 )) || { echo "no checkpoint candidates for ${family}" >&2; exit 4; }
  if [[ ! -s "$selection" ]]; then
    "$PYTHON" scripts/select_active_catalog_sentinel_checkpoint.py "$selection" "${selector_args[@]}"
  fi
}

evaluate_candidates "$WEIGHTED_FAMILY"
evaluate_candidates "$UNWEIGHTED_FAMILY"

selected_adapter() {
  "$PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r.get("passed") and not r.get("test_assets_read"); print(r["selected"]["adapter"])' "$1"
}
weighted_adapter="$(selected_adapter "${RUN_ROOT}/${WEIGHTED_FAMILY}/seed${SEED}/sentinel_checkpoint_selection/selection.json")"
unweighted_adapter="$(selected_adapter "${RUN_ROOT}/${UNWEIGHTED_FAMILY}/seed${SEED}/sentinel_checkpoint_selection/selection.json")"
weighted_output="${RUN_ROOT}/${WEIGHTED_FAMILY}/seed${SEED}/active_catalog_val"
unweighted_output="${RUN_ROOT}/${UNWEIGHTED_FAMILY}/seed${SEED}/active_catalog_val"

full_evaluate() {
  local gpu="$1" adapter="$2" output="$3"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/evaluate_active_catalog_selector.py \
    "$MODEL" "$adapter" "${SFT_ROOT}/val.jsonl" \
    "${SFT_ROOT}/val_evaluation_index.jsonl" "$output" \
    --device cuda:0 --seed "$SEED" --bootstrap-repetitions 2000
}

[[ ! -e "$weighted_output" && ! -e "$unweighted_output" ]] || { echo "refusing existing full evaluation output" >&2; exit 5; }
full_evaluate "$GPU_WEIGHTED" "$weighted_adapter" "$weighted_output" & weighted_pid=$!
full_evaluate "$GPU_UNWEIGHTED" "$unweighted_adapter" "$unweighted_output" & unweighted_pid=$!
failed=0
wait "$weighted_pid" || failed=1
wait "$unweighted_pid" || failed=1
(( failed == 0 )) || exit 6

report="${RUN_ROOT}/seed1_sampling_ablation.json"
[[ ! -e "$report" ]] || { echo "refusing existing sampling report" >&2; exit 7; }
"$PYTHON" scripts/compare_active_catalog_sampling_ablation.py \
  "${weighted_output}/traces.jsonl" "${unweighted_output}/traces.jsonl" \
  "$report" --repetitions 2000 --seed "$SEED"
