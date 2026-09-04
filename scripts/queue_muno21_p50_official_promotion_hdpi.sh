#!/usr/bin/env bash
set -euo pipefail

# Formal validation-only promotion attempt for the existing P50 selector repair.
# The script intentionally changes no model, threshold, budget, or training data.
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-4B}"
RUN_ROOT="${RUN_ROOT:-${STORE}/runs/paper_evidence/muno21_p50_official_promotion_v2}"
LOG_ROOT="${LOG_ROOT:-${STORE}/logs/muno21_p50_official_promotion_v2}"
GPU_IDS="${GPU_IDS:-1,3,4}"
POLL_SECONDS="${POLL_SECONDS:-120}"

STATES="${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl"
EPISODES="${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl"
UPDATER="${STORE}/models/frozen_updater/muno21_v4_seed20260726/best_val_loss.pt"
P50_ROOT="${STORE}/runs/selector/muno21_p50_promotion_v1"
GENERIC_ROOT="${STORE}/runs/selector"
SEEDS=(20260811 20260812 20260813)

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
exec 9>"${RUN_ROOT}/.launcher.lock"
flock -n 9 || exit 0
[[ ! -e "${RUN_ROOT}/COMPLETE.json" ]] || exit 0

source "${PROJECT_ROOT}/scripts/server_hdpi_env.sh"
IFS=',' read -r -a GPUS <<<"${GPU_IDS}"
[[ "${#GPUS[@]}" -eq 3 ]] || { echo "GPU_IDS must contain exactly three devices" >&2; exit 2; }
for gpu in "${GPUS[@]}"; do
  case "${gpu}" in 1|3|4|5) ;; *) echo "GPU ${gpu} is not allowed" >&2; exit 2 ;; esac
done

required=("${PYTHON}" "${MODEL}/config.json" "${STATES}" "${EPISODES}" "${UPDATER}")
for seed in "${SEEDS[@]}"; do
  required+=(
    "${P50_ROOT}/p50_seed${seed: -2}/best.pt"
    "${GENERIC_ROOT}/muno21_evidence_generic_v5_seed${seed}/best.pt"
  )
done
for path in "${required[@]}"; do [[ -s "${path}" ]] || { echo "missing input: ${path}" >&2; exit 3; }; done

if [[ ! -e "${RUN_ROOT}/protocol.json" ]]; then
  cat >"${RUN_ROOT}/protocol.json" <<EOF
{
  "schema_version": "muno21-p50-official-promotion-v1",
  "split": "val",
  "test_assets_read": false,
  "selector_training_seeds": [20260811, 20260812, 20260813],
  "methods": ["generic_selector", "p50_edit_conditioned_selector"],
  "budgets": [1.5, 3.0, 4.5],
  "writeback": "frozen-updater-safe-delta-v1",
  "promotion_rule": "positive paired hierarchical utility proxy and non-inferior no-change error; report APLS and Pixel-F1 without sign filtering"
}
EOF
  sha256sum "${required[@]}" >"${RUN_ROOT}/input_sha256.txt"
fi

echo "[$(date --iso-8601=seconds)] waiting for GPUs ${GPU_IDS}" | tee -a "${LOG_ROOT}/queue.log"
while true; do
  idle=true
  for gpu in "${GPUS[@]}"; do
    pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
    [[ -z "${pids//[[:space:]]/}" ]] || idle=false
  done
  "${idle}" && break
  sleep "${POLL_SECONDS}"
done

run_seed() {
  local seed="$1" gpu="$2"
  local suffix="${seed: -2}"
  local root="${RUN_ROOT}/seed${seed}"
  local rollout="${root}/rollouts"
  local p50="${P50_ROOT}/p50_seed${suffix}/best.pt"
  local generic="${GENERIC_ROOT}/muno21_evidence_generic_v5_seed${seed}/best.pt"
  mkdir -p "${root}"

  if [[ ! -s "${rollout}/summary.json" ]]; then
    [[ ! -e "${rollout}" ]] || { echo "partial rollout root: ${rollout}" >&2; return 4; }
    "${PYTHON}" scripts/evaluate_agent_rollouts.py \
      "${MODEL}" "${STATES}" "${rollout}" \
      --checkpoint "${p50}" --generic-checkpoint "${generic}" \
      --split val --budgets 1.5,3.0,4.5 --device cpu --selector-device cpu \
      --seed "${seed}" --methods generic_selector,edit_conditioned_selector \
      >"${LOG_ROOT}/seed${seed}_rollout.log" 2>&1
  fi

  for method in generic_selector edit_conditioned_selector; do
    local writeback="${root}/writeback/${method}_safe_delta"
    local official="${root}/official/${method}_safe_delta"
    [[ -s "${rollout}/${method}.jsonl" ]] || { echo "missing rollout for ${method}" >&2; return 5; }
    if [[ ! -s "${writeback}/summary.json" ]]; then
      [[ ! -e "${writeback}" ]] || { echo "partial writeback root: ${writeback}" >&2; return 6; }
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/evaluate_agent_map_writeback.py \
        "${UPDATER}" "${EPISODES}" "${rollout}/${method}.jsonl" "${writeback}" \
        --device cuda:0 --split val --image-size 512 --threshold 0.5 \
        --add-min-delta-component-pixels 1536 --delete-min-delta-component-pixels 1024 \
        --reshape-min-delta-component-pixels 0 --preserve-largest-delta-component \
        --protocol-name "muno21-p50-seed${seed}-${method}-safe-delta-v1" \
        --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
        >"${LOG_ROOT}/seed${seed}_${method}_writeback.log" 2>&1
    fi
    if [[ ! -s "${official}/official_metrics.jsonl" ]]; then
      MUNO21_EVAL_SPLIT=val MUNO21_EPISODES="${EPISODES}" \
      MUNO21_GRAPH_SIMPLIFY_TOLERANCE=3.0 \
        bash scripts/run_muno21_official_graph_metrics.sh \
          "${writeback}/writeback.jsonl" "${official}" \
          >"${LOG_ROOT}/seed${seed}_${method}_official.log" 2>&1
    fi
  done
  printf '{"status":"complete","seed":%s,"split":"val","test_assets_read":false}\n' "${seed}" >"${root}/COMPLETE.json"
}

pids=()
for index in 0 1 2; do
  run_seed "${SEEDS[$index]}" "${GPUS[$index]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
((status == 0)) || { printf '{"status":"failed","split":"val","test_assets_read":false}\n' >"${RUN_ROOT}/FAILED.json"; exit 1; }

pairs=()
for seed in "${SEEDS[@]}"; do
  pairs+=(--pair "seed${seed}=${RUN_ROOT}/seed${seed}/official/generic_selector_safe_delta/official_metrics.jsonl=${RUN_ROOT}/seed${seed}/official/edit_conditioned_selector_safe_delta/official_metrics.jsonl")
done
"${PYTHON}" scripts/aggregate_muno21_official_paired_seeds.py \
  "${RUN_ROOT}/p50_vs_seed_matched_generic.json" "${pairs[@]}" \
  --repetitions 10000 --seed 20260808 \
  >"${LOG_ROOT}/aggregate.log" 2>&1

RUN_ROOT="${RUN_ROOT}" "${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
required = [root / "protocol.json", root / "input_sha256.txt", root / "p50_vs_seed_matched_generic.json"]
for seed in (20260811, 20260812, 20260813):
    required.append(root / f"seed{seed}" / "COMPLETE.json")
if missing := [str(path) for path in required if not path.is_file()]:
    raise FileNotFoundError(missing)
(root / "COMPLETE.json").write_text(json.dumps({
    "schema_version": "muno21-p50-official-promotion-v1",
    "status": "complete",
    "split": "val",
    "test_assets_read": False,
    "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in required},
}, indent=2) + "\n", encoding="utf-8")
PY

echo "[$(date --iso-8601=seconds)] P50 official promotion attempt complete" | tee -a "${LOG_ROOT}/queue.log"
