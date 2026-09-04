#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
GPU="${GPU:-3}"
POLL_SECONDS="${POLL_SECONDS:-120}"
WAIT_FOR_SN7_EPISODES="${WAIT_FOR_SN7_EPISODES:-0}"

SN7_RUN_ROOT="${STORAGE_ROOT}/runs/sn7_active_catalog/step0_frozen_test_v2"
SN7_EPISODES="${SN7_RUN_ROOT}/episodes_test.jsonl"
SN7_LEDGER="${STORAGE_ROOT}/artifacts/paper_results/frozen_test_access/sn7_step0_v2.json"

DATA_ROOT="${STORAGE_ROOT}/processed/muno21_v2/agent"
STATES="${DATA_ROOT}/selector_states_v1.jsonl"
EPISODES="${DATA_ROOT}/episodes_train_val_v1.jsonl"
UPDATER="${STORAGE_ROOT}/models/frozen_updater/muno21_v4_seed20260726/best_val_loss.pt"
MODEL="/home/wh/hf_models/Qwen3-4B"
TOOL_GATE_ROOT="${STORAGE_ROOT}/runs/agent/muno21_tool_need_gate_v11_seed20260831"
TOOL_GATE="${TOOL_GATE_ROOT}/gate.joblib"
TOOL_BELIEF="${STORAGE_ROOT}/runs/agent/tool_belief_anchored_v4_seed20260821/best.pt"
STOP_BASELINE="${STORAGE_ROOT}/runs/muno21_active_catalog_transfer_v1/official_always_stop/official_metrics.jsonl"
OLD_BASELINE="${STORAGE_ROOT}/runs/muno21_evidence_value_official_val_v2_20260725/old_generic/official_metrics.jsonl"

RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/paper_evidence/muno21_independent_selector_validation_v1_20260807}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/logs/muno21_independent_selector_validation_v1_20260807}"
SEEDS=(20260811 20260812 20260813)
METHODS=(generic_selector edit_conditioned_selector edit_conditioned_proactive_tools)

case "${GPU}" in
  1|3|4|5) ;;
  0|2) echo "GPU ${GPU} is excluded by the current allocation" >&2; exit 2 ;;
  *) echo "GPU ${GPU} is outside the four-GPU ActiveMap allocation" >&2; exit 2 ;;
esac

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
exec 9>"${RUN_ROOT}/.lock"
flock -n 9 || exit 0
[[ ! -s "${RUN_ROOT}/COMPLETE.json" ]] || exit 0

record_failure() {
  local rc=$?
  trap - EXIT
  if (( rc != 0 )); then
    printf '{"status":"failed","exit_code":%d,"split":"val","test_assets_read":false}\n' \
      "${rc}" >"${RUN_ROOT}/FAILED.json"
  fi
  exit "${rc}"
}
trap record_failure EXIT
rm -f "${RUN_ROOT}/FAILED.json"

required=(
  "${PYTHON}" "${STATES}" "${EPISODES}" "${UPDATER}" "${MODEL}/config.json"
  "${TOOL_GATE}" "${TOOL_GATE_ROOT}/summary.json" "${TOOL_BELIEF}"
  "${STOP_BASELINE}" "${OLD_BASELINE}"
)
for seed in "${SEEDS[@]}"; do
  required+=(
    "${STORAGE_ROOT}/runs/selector/muno21_evidence_conservative_v5_seed${seed}/best.pt"
    "${STORAGE_ROOT}/runs/selector/muno21_evidence_generic_v5_seed${seed}/best.pt"
  )
done
for path in "${required[@]}"; do
  [[ -e "${path}" ]] || { echo "missing required input: ${path}" >&2; exit 3; }
done

cat >"${RUN_ROOT}/protocol.json" <<EOF
{
  "schema_version": "muno21-independent-selector-validation-v1",
  "split": "val",
  "test_assets_read": false,
  "selector_training_seeds": [20260811, 20260812, 20260813],
  "budgets": [1.5, 3.0, 4.5],
  "tool_need_threshold": 0.09,
  "max_tool_calls": 2,
  "safe_delta": {
    "add_min_delta_component_pixels": 1536,
    "delete_min_delta_component_pixels": 1024,
    "reshape_min_delta_component_pixels": 0,
    "preserve_largest_delta_component": true
  },
  "graph_simplify_tolerance": 3.0,
  "bootstrap_repetitions": 10000,
  "purpose": "separate independent selector-training seeds from historical rollout seeds"
}
EOF
sha256sum "${required[@]}" >"${RUN_ROOT}/input_sha256.txt"

if [[ "${WAIT_FOR_SN7_EPISODES}" == "1" ]]; then
  echo "[$(date --iso-8601=seconds)] waiting for SN7 episode construction to finish" \
    | tee -a "${LOG_ROOT}/queue.log"
  while [[ ! -s "${SN7_EPISODES}" ]]; do
    status="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status","missing"))' "${SN7_LEDGER}")"
    [[ "${status}" != "failed" ]] || {
      echo "SN7 frozen test failed before MUNO21 validation launch" >&2
      exit 4
    }
    sleep "${POLL_SECONDS}"
  done
else
  echo "[$(date --iso-8601=seconds)] launching independently of SN7 construction" \
    | tee -a "${LOG_ROOT}/queue.log"
fi

echo "[$(date --iso-8601=seconds)] waiting for physical GPU ${GPU}" \
  | tee -a "${LOG_ROOT}/queue.log"
while true; do
  gpu_pids="$(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${gpu_pids//[[:space:]]/}" ]] && break
  sleep "${POLL_SECONDS}"
done

for seed in "${SEEDS[@]}"; do
  seed_root="${RUN_ROOT}/seed${seed}"
  rollout_root="${seed_root}/rollouts"
  edit_checkpoint="${STORAGE_ROOT}/runs/selector/muno21_evidence_conservative_v5_seed${seed}/best.pt"
  generic_checkpoint="${STORAGE_ROOT}/runs/selector/muno21_evidence_generic_v5_seed${seed}/best.pt"
  mkdir -p "${seed_root}"

  if [[ ! -s "${rollout_root}/summary.json" ]]; then
    [[ ! -e "${rollout_root}" ]] || {
      echo "refusing partial rollout root: ${rollout_root}" >&2
      exit 5
    }
    "${PYTHON}" scripts/evaluate_agent_rollouts.py \
      "${MODEL}" "${STATES}" "${rollout_root}" \
      --checkpoint "${edit_checkpoint}" \
      --generic-checkpoint "${generic_checkpoint}" \
      --episodes "${EPISODES}" \
      --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
      --tool-belief-checkpoint "${TOOL_BELIEF}" \
      --tool-artifact-root "${rollout_root}/tool_artifacts" \
      --tool-need-gate "${TOOL_GATE}" --tool-need-threshold 0.09 \
      --max-tool-calls 2 --split val --budgets 1.5,3.0,4.5 \
      --device cpu --selector-device cpu --seed "${seed}" \
      --methods generic_selector,edit_conditioned_selector,edit_conditioned_proactive_tools \
      >"${LOG_ROOT}/seed${seed}_rollout.log" 2>&1
  fi

  for method in "${METHODS[@]}"; do
    rollout="${rollout_root}/${method}.jsonl"
    writeback_root="${seed_root}/writeback/${method}_safe_delta"
    official_root="${seed_root}/official/${method}_safe_delta"
    [[ -s "${rollout}" ]] || { echo "missing rollout: ${rollout}" >&2; exit 6; }

    if [[ ! -s "${writeback_root}/summary.json" ]]; then
      [[ ! -e "${writeback_root}" ]] || {
        echo "refusing partial writeback root: ${writeback_root}" >&2
        exit 7
      }
      CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" scripts/evaluate_agent_map_writeback.py \
        "${UPDATER}" "${EPISODES}" "${rollout}" "${writeback_root}" \
        --device cuda:0 --split val --image-size 512 --threshold 0.5 \
        --add-min-delta-component-pixels 1536 \
        --delete-min-delta-component-pixels 1024 \
        --reshape-min-delta-component-pixels 0 \
        --preserve-largest-delta-component \
        --protocol-name "muno21-independent-seed${seed}-${method}-safe-delta-v1" \
        --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
        >"${LOG_ROOT}/seed${seed}_${method}_writeback.log" 2>&1
    fi

    if [[ ! -s "${official_root}/official_metrics.jsonl" ]]; then
      MUNO21_EVAL_SPLIT=val \
      MUNO21_EPISODES="${EPISODES}" \
      MUNO21_GRAPH_SIMPLIFY_TOLERANCE=3.0 \
        bash scripts/run_muno21_official_graph_metrics.sh \
          "${writeback_root}/writeback.jsonl" "${official_root}" \
          >"${LOG_ROOT}/seed${seed}_${method}_official.log" 2>&1
    fi
  done

  SEED="${seed}" SEED_ROOT="${seed_root}" "${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["SEED_ROOT"])
methods = (
    "generic_selector",
    "edit_conditioned_selector",
    "edit_conditioned_proactive_tools",
)
files = {}
for method in methods:
    path = root / "official" / f"{method}_safe_delta" / "official_metrics.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    files[method] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
(root / "COMPLETE.json").write_text(
    json.dumps(
        {
            "schema_version": "muno21-independent-selector-seed-v1",
            "status": "complete",
            "selector_training_seed": int(os.environ["SEED"]),
            "split": "val",
            "test_assets_read": False,
            "official_metrics": files,
        },
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
PY
done

candidate_args=()
generic_pairs=()
direct_pairs=()
for seed in "${SEEDS[@]}"; do
  active="${RUN_ROOT}/seed${seed}/official/edit_conditioned_proactive_tools_safe_delta/official_metrics.jsonl"
  generic="${RUN_ROOT}/seed${seed}/official/generic_selector_safe_delta/official_metrics.jsonl"
  direct="${RUN_ROOT}/seed${seed}/official/edit_conditioned_selector_safe_delta/official_metrics.jsonl"
  candidate_args+=(--candidate "seed${seed}=${active}")
  generic_pairs+=(--pair "seed${seed}=${generic}=${active}")
  direct_pairs+=(--pair "seed${seed}=${direct}=${active}")
done

if [[ ! -s "${RUN_ROOT}/active_vs_always_stop.json" ]]; then
  "${PYTHON}" -m scripts.aggregate_muno21_official_three_seeds \
    "${STOP_BASELINE}" "${RUN_ROOT}/active_vs_always_stop.json" \
    "${candidate_args[@]}" --repetitions 10000 --seed 20260807
fi
if [[ ! -s "${RUN_ROOT}/active_vs_old_selector.json" ]]; then
  "${PYTHON}" -m scripts.aggregate_muno21_official_three_seeds \
    "${OLD_BASELINE}" "${RUN_ROOT}/active_vs_old_selector.json" \
    "${candidate_args[@]}" --repetitions 10000 --seed 20260807
fi
if [[ ! -s "${RUN_ROOT}/active_vs_seed_matched_generic.json" ]]; then
  "${PYTHON}" scripts/aggregate_muno21_official_paired_seeds.py \
    "${RUN_ROOT}/active_vs_seed_matched_generic.json" \
    "${generic_pairs[@]}" --repetitions 10000 --seed 20260807
fi
if [[ ! -s "${RUN_ROOT}/active_vs_seed_matched_direct.json" ]]; then
  "${PYTHON}" scripts/aggregate_muno21_official_paired_seeds.py \
    "${RUN_ROOT}/active_vs_seed_matched_direct.json" \
    "${direct_pairs[@]}" --repetitions 10000 --seed 20260807
fi

RUN_ROOT="${RUN_ROOT}" GPU="${GPU}" "${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
names = (
    "protocol.json",
    "input_sha256.txt",
    "active_vs_always_stop.json",
    "active_vs_old_selector.json",
    "active_vs_seed_matched_generic.json",
    "active_vs_seed_matched_direct.json",
)
files = {}
for name in names:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(path)
    files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
(root / "COMPLETE.json").write_text(
    json.dumps(
        {
            "schema_version": "muno21-independent-selector-validation-v1",
            "status": "complete",
            "physical_gpu": int(os.environ["GPU"]),
            "selector_training_seeds": [20260811, 20260812, 20260813],
            "split": "val",
            "test_assets_read": False,
            "files": files,
        },
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
PY

trap - EXIT
echo "[$(date --iso-8601=seconds)] MUNO21 independent selector validation complete" \
  | tee -a "${LOG_ROOT}/queue.log"
