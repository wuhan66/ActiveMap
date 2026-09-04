#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
if [[ -n "${ACTIVEMAP_SERVER_ENV:-}" ]]; then
  # shellcheck source=/dev/null
  source "${ACTIVEMAP_SERVER_ENV}"
else
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/scripts/server_hdpi_env.sh"
fi
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-${ACTIVEMAP_AGENT_ENV}/bin/python}"
ROOT="${MUNO21_AGENT_ROOT:-${ACTIVEMAP_PROCESSED_ROOT}/muno21_v2/agent}"
MAIN="${ROOT}/agent_data_v6_anonymized"
TOOL="${ROOT}/sparse_tool_sft_v2"
OUTPUT="${MUNO21_AGENT_DATA_ROOT:-${ROOT}/agent_data_v10_balanced_sparse_tools}"
USE_TOOL_REPEAT="${MUNO21_USE_TOOL_REPEAT:-5}"
NO_TOOL_RATIO="${MUNO21_NO_TOOL_RATIO:-3.0}"
SEED="${MUNO21_COMPOSITION_SEED:-20260821}"

for path in \
  "${MAIN}/train/sft.jsonl" "${MAIN}/val/sft.jsonl" \
  "${TOOL}/train/sft.jsonl" "${TOOL}/val/sft.jsonl"; do
  [[ -s "${path}" ]] || { echo "Required input is missing: ${path}" >&2; exit 1; }
done
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing to reuse output: ${OUTPUT}" >&2; exit 1; }

mkdir -p "${OUTPUT}/train" "${OUTPUT}/val"
cd "${PROJECT_ROOT}"
PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON}" \
  scripts/compose_agent_sft.py \
  "${MAIN}/train/sft.jsonl" "${TOOL}/train/sft.jsonl" \
  "${OUTPUT}/train/sft_composed.jsonl" \
  --training --use-tool-repeat "${USE_TOOL_REPEAT}" \
  --no-tool-ratio "${NO_TOOL_RATIO}" --seed "${SEED}"
PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON}" \
  scripts/compose_agent_sft.py \
  "${MAIN}/val/sft.jsonl" "${TOOL}/val/sft.jsonl" \
  "${OUTPUT}/val/sft_composed.jsonl" --seed "${SEED}"

"${PYTHON}" - "${OUTPUT}" "${USE_TOOL_REPEAT}" "${NO_TOOL_RATIO}" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
repeat = int(sys.argv[2])
negative_ratio = float(sys.argv[3])
train = json.loads((root / "train/sft_composed.summary.json").read_text())
val = json.loads((root / "val/sft_composed.summary.json").read_text())
expected_negative = min(
    train["tool_negative_sequence_count"],
    math.ceil(train["tool_positive_sequence_count"] * negative_ratio),
)
checks = {
    "train_tool_positives_present": train["tool_positive_sequence_count"] > 0,
    "negative_sampling_matches_ratio": (
        train["selected_tool_negative_sequence_count"] == expected_negative
    ),
    "positive_action_repetition_matches": train["use_tool_repeat"] == repeat,
    "use_tool_training_rate_noncollapsed": (
        0.05 <= train["action_fractions"].get("USE_TOOL", 0.0) <= 0.20
    ),
    "validation_natural_prevalence": (
        val["training_oversampling"] is False
        and val["selected_tool_negative_sequence_count"]
        == val["tool_negative_sequence_count"]
    ),
    "validation_tool_positives_present": val["action_counts"].get("USE_TOOL", 0) > 0,
    "train_val_only": train["test_assets_read"] is False and val["test_assets_read"] is False,
}
if not all(checks.values()):
    raise SystemExit(f"balanced sparse-tool data gate failed: {checks}")

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

audit = {
    "schema_version": "agent-sft-balanced-tool-prior-audit-v1",
    "controlled_difference_from_v9": "training_tool_opportunity_sampling",
    "use_tool_repeat": repeat,
    "no_tool_ratio": negative_ratio,
    "train_action_fractions": train["action_fractions"],
    "validation_action_fractions": val["action_fractions"],
    "checks": {**checks, "passed": all(checks.values())},
    "files": {
        "train": sha256(root / "train/sft_composed.jsonl"),
        "val": sha256(root / "val/sft_composed.jsonl"),
    },
    "test_assets_read": False,
}
(root / "balanced_prior_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit, indent=2))
PY
