#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
SEEDS=(20260822 20260823 20260824 20260825)
SELECTOR_SEEDS=(20260812 20260813 20260811 20260812)
RUN_FAMILY="muno21_qwen3_4b_balanced_tool_sft"
ROLLOUT_FAMILY="${MUNO21_ROLLOUT_FAMILY:-agent_v10_balanced_tool}"
POSTTRAIN_CONTROL="${MUNO21_POSTTRAIN_CONTROL_NAME:-posttrain_watcher}"
WRITEBACK_SUBDIR="${MUNO21_WRITEBACK_SUBDIR:-writeback}"
OUTPUT="${MUNO21_AGGREGATE_OUTPUT:-${STORAGE_ROOT}/runs/agent/muno21_balanced_tool_four_seed_v1}"
COMBINED_ROLLOUTS="${OUTPUT}/rollouts"
LOG="${OUTPUT}/watcher.log"
LOCK="${OUTPUT}/.lock"

mkdir -p "${OUTPUT}"
exec 9>"${LOCK}"
flock -n 9 || exit 0
exec >>"${LOG}" 2>&1

echo "[$(date -Is)] waiting for four balanced-tool post-train evaluations"
for seed in "${SEEDS[@]}"; do
  marker="${STORAGE_ROOT}/runs/agent/${RUN_FAMILY}_seed${seed}/evaluation/${POSTTRAIN_CONTROL}/COMPLETE.json"
  while [[ ! -s "${marker}" ]]; do
    failed="${STORAGE_ROOT}/runs/agent/${RUN_FAMILY}_seed${seed}/evaluation/${POSTTRAIN_CONTROL}/FAILED.json"
    recovery="${STORAGE_ROOT}/runs/agent/${RUN_FAMILY}_seed${seed}/control/false_call_recovery_v1/protocol.json"
    if [[ -s "${failed}" ]]; then
      if [[ -s "${recovery}" && "${failed}" -ot "${recovery}" ]]; then
        sleep 90
        continue
      fi
      printf '{"status":"failed","seed":%s,"reason":"seed_posttrain_failed","test_assets_read":false}\n' \
        "${seed}" >"${OUTPUT}/FAILED.json"
      exit 3
    fi
    sleep 90
  done
done

mkdir -p "${COMBINED_ROLLOUTS}"
for seed in "${SEEDS[@]}"; do
  target="${STORAGE_ROOT}/artifacts/paper_rollouts/${ROLLOUT_FAMILY}_seed${seed}_val/seed${seed}"
  link="${COMBINED_ROLLOUTS}/seed${seed}"
  [[ -s "${target}/summary.json" ]] || {
    echo "missing rollout summary: ${target}/summary.json" >&2
    exit 3
  }
  if [[ -L "${link}" ]]; then
    [[ "$(readlink -f "${link}")" == "${target}" ]] || {
      echo "refusing changed rollout link: ${link}" >&2
      exit 4
    }
  elif [[ -e "${link}" ]]; then
    echo "refusing existing non-link rollout path: ${link}" >&2
    exit 4
  else
    ln -s "${target}" "${link}"
  fi
done

"${PYTHON}" - "${COMBINED_ROLLOUTS}/seed_pairing.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
record = {
    "schema_version": "activemap-agent-selector-seed-pairing-v1",
    "split": "val",
    "test_assets_read": False,
    "pairs": [
        {"agent_seed": 20260822, "selector_seed": 20260812},
        {"agent_seed": 20260823, "selector_seed": 20260813},
        {"agent_seed": 20260824, "selector_seed": 20260811},
        {"agent_seed": 20260825, "selector_seed": 20260812},
    ],
}
if path.exists() and json.loads(path.read_text(encoding="utf-8")) != record:
    raise SystemExit(f"refusing changed seed pairing: {path}")
path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
PY

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
controller_output="${OUTPUT}/controller_four_seed.json"
if [[ ! -s "${controller_output}" ]]; then
  "${PYTHON}" scripts/aggregate_agent_three_seeds.py \
    "${COMBINED_ROLLOUTS}" "${controller_output}" \
    --seeds "20260822,20260823,20260824,20260825" \
    --candidate qwen3_4b_sft_tool_to_belief \
    --baselines \
      "qwen3_4b_sft,qwen3_4b_sft_tools_no_belief,edit_conditioned_selector,generic_selector,forced_tools" \
    --bootstrap 10000 --rng-seed 20260730
fi

pair_args=()
safe_pair_args=()
filter_pair_args=()
for seed in "${SEEDS[@]}"; do
  root="${STORAGE_ROOT}/runs/agent/${RUN_FAMILY}_seed${seed}/evaluation/${WRITEBACK_SUBDIR}"
  sft="${root}/qwen3_4b_sft/writeback.jsonl"
  raw="${root}/qwen3_4b_sft_tool_to_belief/writeback.jsonl"
  safe="${root}/qwen3_4b_sft_tool_to_belief_safe_delta/writeback.jsonl"
  for path in "${sft}" "${raw}" "${safe}"; do
    [[ -s "${path}" ]] || { echo "missing writeback: ${path}" >&2; exit 3; }
  done
  pair_args+=(--pair "${seed}=${sft},${raw}")
  safe_pair_args+=(--pair "${seed}=${sft},${safe}")
  filter_pair_args+=(--pair "${seed}=${raw},${safe}")
done

if [[ ! -s "${OUTPUT}/raw_belief_vs_sft.json" ]]; then
  "${PYTHON}" scripts/aggregate_agent_writeback_pairs.py \
    "${OUTPUT}/raw_belief_vs_sft.json" "${pair_args[@]}" \
    --repetitions 10000 --seed 20260730
fi
if [[ ! -s "${OUTPUT}/safe_delta_vs_sft.json" ]]; then
  "${PYTHON}" scripts/aggregate_agent_writeback_pairs.py \
    "${OUTPUT}/safe_delta_vs_sft.json" "${safe_pair_args[@]}" \
    --repetitions 10000 --seed 20260731
fi
if [[ ! -s "${OUTPUT}/safe_delta_vs_raw_belief.json" ]]; then
  "${PYTHON}" scripts/aggregate_agent_writeback_pairs.py \
    "${OUTPUT}/safe_delta_vs_raw_belief.json" "${filter_pair_args[@]}" \
    --repetitions 10000 --seed 20260801
fi

"${PYTHON}" scripts/assess_muno21_balanced_tool_multiseed.py \
  "${OUTPUT}/controller_four_seed.json" \
  "${OUTPUT}/safe_delta_vs_sft.json" \
  "${OUTPUT}/safe_delta_vs_raw_belief.json" \
  "${OUTPUT}/promotion_decision.json" \
  --run-dir "${STORAGE_ROOT}/runs/agent/${RUN_FAMILY}_seed20260822" \
  --run-dir "${STORAGE_ROOT}/runs/agent/${RUN_FAMILY}_seed20260823" \
  --run-dir "${STORAGE_ROOT}/runs/agent/${RUN_FAMILY}_seed20260824" \
  --run-dir "${STORAGE_ROOT}/runs/agent/${RUN_FAMILY}_seed20260825"

"${PYTHON}" scripts/build_muno21_balanced_tool_multiseed_table.py \
  "${OUTPUT}/controller_four_seed.json" \
  "${OUTPUT}/raw_belief_vs_sft.json" \
  "${OUTPUT}/safe_delta_vs_sft.json" \
  "${OUTPUT}/safe_delta_vs_raw_belief.json" \
  "${OUTPUT}/paper_assets"
"${PYTHON}" scripts/plot_muno21_balanced_tool_multiseed.py \
  "${OUTPUT}/paper_assets/table.json" \
  "${OUTPUT}/paper_assets"
"${PYTHON}" scripts/plot_muno21_checkpoint_safety_trajectory.py \
  "${OUTPUT}/paper_assets" \
  "${STORAGE_ROOT}/runs/agent/${RUN_FAMILY}_seed20260822" \
  "${STORAGE_ROOT}/runs/agent/${RUN_FAMILY}_seed20260823" \
  "${STORAGE_ROOT}/runs/agent/${RUN_FAMILY}_seed20260824" \
  "${STORAGE_ROOT}/runs/agent/${RUN_FAMILY}_seed20260825" \
  --max-false-call-rate 0.02

OUTPUT="${OUTPUT}" "${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT"])
names = (
    "controller_four_seed.json",
    "raw_belief_vs_sft.json",
    "safe_delta_vs_sft.json",
    "safe_delta_vs_raw_belief.json",
    "promotion_decision.json",
    "paper_assets/table.json",
    "paper_assets/table.md",
    "paper_assets/table.tex",
    "paper_assets/controller.csv",
    "paper_assets/writeback.csv",
    "paper_assets/controller_forest.png",
    "paper_assets/controller_forest.pdf",
    "paper_assets/writeback_forest.png",
    "paper_assets/writeback_forest.pdf",
    "paper_assets/checkpoint_safety_trajectory.csv",
    "paper_assets/checkpoint_safety_trajectory.json",
    "paper_assets/checkpoint_safety_trajectory.png",
    "paper_assets/checkpoint_safety_trajectory.pdf",
)
files = {}
for name in names:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(path)
    files[name] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
record = {
    "schema_version": "muno21-balanced-tool-four-seed-aggregate-v1",
    "status": "complete",
    "model_seeds": [20260822, 20260823, 20260824, 20260825],
    "files": files,
    "split": "val",
    "test_assets_read": False,
}
(root / "COMPLETE.json").write_text(
    json.dumps(record, indent=2) + "\n", encoding="utf-8"
)
PY

echo "[$(date -Is)] four-seed aggregation complete"
