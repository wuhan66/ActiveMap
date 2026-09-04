#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
SEEDS="${SEEDS:-20260831 20260901}"
OUTPUT="${OUTPUT:-${STORAGE_ROOT}/artifacts/paper_evidence/muno21_direct_vlm_supervised_2seed_v1}"
LOG="${OUTPUT}/watcher.log"

mkdir -p "${OUTPUT}"
exec 9>"${OUTPUT}/.lock"
flock -n 9 || exit 0
[[ ! -s "${OUTPUT}/COMPLETE.json" ]] || exit 0

args=()
for seed in ${SEEDS}; do
  run="${STORAGE_ROOT}/runs/agent/muno21_direct_vlm_qwen3vl4b_sft_v1_seed${seed}"
  while [[ ! -s "${run}/COMPLETE.json" ]]; do
    if [[ -s "${run}/FAILED.json" ]]; then
      printf '{"status":"failed","reason":"upstream_failed","seed":%s}\n' \
        "${seed}" >"${OUTPUT}/FAILED.json"
      exit 3
    fi
    sleep 60
  done
  args+=(--run "${seed}=${run}")
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
"${PYTHON}" scripts/aggregate_muno21_direct_vlm_sft.py \
  "${OUTPUT}" "${args[@]}" >>"${LOG}" 2>&1

writeback_args=()
for seed in ${SEEDS}; do
  result="${STORAGE_ROOT}/runs/agent/muno21_direct_vlm_qwen3vl4b_sft_v1_seed${seed}/writeback/safe_delta/writeback.jsonl"
  writeback_args+=(--result "${seed}=${result}")
done
"${PYTHON}" scripts/aggregate_semantic_vlm_writeback_seeds.py \
  "${OUTPUT}/writeback_bootstrap.json" "${writeback_args[@]}" \
  --repetitions 10000 --bootstrap-seed 20260902 >>"${LOG}" 2>&1

comparison_args=(
  --run "Minimal zero-shot=${STORAGE_ROOT}/runs/agent/muno21_direct_vlm_qwen3vl4b_zeroshot_v1"
  --run "Operational zero-shot=${STORAGE_ROOT}/runs/agent/muno21_direct_vlm_qwen3vl4b_operational_prompt_v2"
  --run "Four-shot visual ICL=${STORAGE_ROOT}/runs/agent/muno21_direct_vlm_qwen3vl4b_fourshot_v3"
)
for seed in ${SEEDS}; do
  comparison_args+=(
    --run "Supervised Direct-VLM seed ${seed}=${STORAGE_ROOT}/runs/agent/muno21_direct_vlm_qwen3vl4b_sft_v1_seed${seed}"
  )
done
"${PYTHON}" scripts/build_muno21_direct_vlm_comparison.py \
  "${OUTPUT}/prompt_to_supervised_comparison" \
  "${comparison_args[@]}" --budget 3.0 >>"${LOG}" 2>&1

OUTPUT="${OUTPUT}" "${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT"])
paths = (
    root / "summary.json",
    root / "per_seed.csv",
    root / "table.md",
    root / "writeback_bootstrap.json",
    root / "prompt_to_supervised_comparison/table.json",
    root / "prompt_to_supervised_comparison/table.md",
)
for path in paths:
    if not path.is_file():
        raise FileNotFoundError(path)
summary = json.loads(paths[0].read_text(encoding="utf-8"))
bootstrap = json.loads(paths[3].read_text(encoding="utf-8"))
if summary["test_assets_read"] or bootstrap["test_assets_read"]:
    raise ValueError("test leakage in supervised Direct-VLM aggregate")
record = {
    "schema_version": "muno21-direct-vlm-supervised-aggregate-complete-v1",
    "status": "complete",
    "files": {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    },
    "test_assets_read": False,
}
(root / "COMPLETE.json").write_text(
    json.dumps(record, indent=2) + "\n", encoding="utf-8"
)
PY
