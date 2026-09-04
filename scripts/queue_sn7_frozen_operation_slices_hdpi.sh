#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/step0_frozen_test_v2}"
CANONICAL_ROOT="${CANONICAL_ROOT:-${STORAGE_ROOT}/artifacts/paper_results/sn7_step0_frozen_test_v2_20260807}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/artifacts/paper_results/sn7_step0_frozen_test_operation_slices_v1_20260807}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/logs/sn7_step0_frozen_test_v2}"
POLL_SECONDS="${POLL_SECONDS:-120}"
SEEDS=(20260730 20260731 20260801)

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
mkdir -p "${LOG_ROOT}" "$(dirname "${OUTPUT_ROOT}")"

exec 9>"${OUTPUT_ROOT}.lock"
flock -n 9 || exit 0
[[ ! -e "${OUTPUT_ROOT}" ]] || {
  [[ -s "${OUTPUT_ROOT}/manifest.json" ]] && exit 0
  echo "refusing incomplete operation-slice output: ${OUTPUT_ROOT}" >&2
  exit 31
}

echo "[$(date --iso-8601=seconds)] waiting for canonical frozen-test export" \
  | tee -a "${LOG_ROOT}/operation_slices_queue.log"
while [[ ! -s "${CANONICAL_ROOT}/manifest.json" ]]; do
  sleep "${POLL_SECONDS}"
done

mkdir "${OUTPUT_ROOT}"

notool_pairs=()
forced_pairs=()
for seed in "${SEEDS[@]}"; do
  benefit="${RUN_ROOT}/seed${seed}/benefit/writeback/evaluation/writeback.jsonl"
  notool="${RUN_ROOT}/seed${seed}/notool/writeback/evaluation/writeback.jsonl"
  forced="${RUN_ROOT}/seed${seed}/forced/writeback/evaluation/writeback.jsonl"
  for path in "${benefit}" "${notool}" "${forced}"; do
    [[ -s "${path}" ]] || { echo "missing writeback receipt: ${path}" >&2; exit 32; }
  done
  notool_pairs+=(--pair "${seed}=${notool},${benefit}")
  forced_pairs+=(--pair "${seed}=${forced},${benefit}")
done

"${PYTHON}" scripts/aggregate_sn7_frozen_test_operation_slices.py \
  "${OUTPUT_ROOT}/benefit_vs_notool" \
  --reference-label no-tool --candidate-label benefit-gated \
  --repetitions 10000 --seed 20260807 "${notool_pairs[@]}" \
  >"${LOG_ROOT}/operation_slices_benefit_vs_notool.log" 2>&1

"${PYTHON}" scripts/aggregate_sn7_frozen_test_operation_slices.py \
  "${OUTPUT_ROOT}/benefit_vs_forced" \
  --reference-label forced-tool --candidate-label benefit-gated \
  --repetitions 10000 --seed 20260807 "${forced_pairs[@]}" \
  >"${LOG_ROOT}/operation_slices_benefit_vs_forced.log" 2>&1

OUTPUT_ROOT="${OUTPUT_ROOT}" CANONICAL_ROOT="${CANONICAL_ROOT}" "${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT"])
canonical = Path(os.environ["CANONICAL_ROOT"]) / "manifest.json"
files = sorted(path for path in root.rglob("*") if path.is_file())

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = {
    "schema_version": "sn7-frozen-test-operation-slice-bundle-v1",
    "split": "test",
    "test_assets_read": True,
    "analysis_role": "posthoc_descriptive_only",
    "promotion_decision_used": False,
    "canonical_manifest": {
        "path": str(canonical.resolve()),
        "sha256": sha256(canonical),
    },
    "outputs": {
        str(path.relative_to(root)): sha256(path)
        for path in files
    },
}
(root / "manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
PY

echo "[$(date --iso-8601=seconds)] operation-slice export complete: ${OUTPUT_ROOT}" \
  | tee -a "${LOG_ROOT}/operation_slices_queue.log"
