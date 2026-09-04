#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
source "${PROJECT_ROOT}/scripts/server_hdpi_env.sh"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-${ACTIVEMAP_AGENT_ENV}/bin/python}"
ROOT="${MUNO21_AGENT_DATA_ROOT:-${ACTIVEMAP_PROCESSED_ROOT}/muno21_v2/agent}"
MAIN="${ROOT}/agent_data_v6_anonymized"
TOOL="${ROOT}/sparse_tool_sft_v2"
OUTPUT="${ROOT}/agent_data_v9_natural_sparse_tools"

for path in \
  "${MAIN}/train/sft.jsonl" \
  "${MAIN}/val/sft.jsonl" \
  "${TOOL}/train/sft.jsonl" \
  "${TOOL}/val/sft.jsonl"; do
  [[ -s "${path}" ]] || { echo "Required input is missing: ${path}" >&2; exit 1; }
done
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing to reuse output: ${OUTPUT}" >&2; exit 1; }

mkdir -p "${OUTPUT}/train" "${OUTPUT}/val"
cd "${PROJECT_ROOT}"
PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON}" \
  scripts/compose_agent_sft.py \
  "${MAIN}/train/sft.jsonl" "${TOOL}/train/sft.jsonl" \
  "${OUTPUT}/train/sft_composed.jsonl" \
  --training --use-tool-repeat 1 --keep-all-tool-sequences --seed 20260821
PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON}" \
  scripts/compose_agent_sft.py \
  "${MAIN}/val/sft.jsonl" "${TOOL}/val/sft.jsonl" \
  "${OUTPUT}/val/sft_composed.jsonl" --seed 20260821

"${PYTHON}" - "${OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
train = json.loads((root / "train/sft_composed.summary.json").read_text())
val = json.loads((root / "val/sft_composed.summary.json").read_text())
for action in ("ACQUIRE", "USE_TOOL"):
    train_rate = train["action_fractions"][action]
    val_rate = val["action_fractions"][action]
    ratio = train_rate / val_rate
    if not 0.5 <= ratio <= 2.0:
        raise SystemExit(f"{action} prior ratio is out of bounds: {ratio:.4f}")
assert train["use_tool_repeat"] == 1
assert train["selected_tool_negative_sequence_count"] == train["tool_negative_sequence_count"]
assert train["keep_all_tool_sequences"] is True
assert val["training_oversampling"] is False
assert train["test_assets_read"] is False and val["test_assets_read"] is False
audit = {
    "schema_version": "agent-sft-natural-prior-audit-v1",
    "train_output_count": train["output_count"],
    "val_output_count": val["output_count"],
    "action_prior_ratios_train_over_val": {
        action: train["action_fractions"][action] / val["action_fractions"][action]
        for action in ("ACQUIRE", "USE_TOOL")
    },
    "test_assets_read": False,
}
(root / "prior_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit, indent=2))
PY
