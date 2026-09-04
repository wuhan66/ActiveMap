#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/wh/projects/activemap-v1
STORE=/home/wh/ActiveMap
PYTHON="$STORE/envs/activemap-agent/bin/python"
FULL="$STORE/processed/sn7_v1/agent/sequential_selector_v1/full"
RUN="$STORE/runs/sn7_active_catalog/onpolicy_rl_reentry_stage_a_20260731"
ROLLOUTS="${ROLLOUTS:-4}"

cd "$PROJECT"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
mkdir -p "$RUN/preferences_v2"

for split in train val; do
  output="$RUN/preferences_v2/${split}.jsonl"
  test ! -e "$output" || {
    echo "refusing existing preference output: $output" >&2
    exit 4
  }
  trace_args=()
  for ((rollout=0; rollout<ROLLOUTS; rollout++)); do
    trace="$RUN/${split}_rollout${rollout}/evaluation/traces.jsonl"
    test -f "$trace" || { echo "missing rollout trace: $trace" >&2; exit 3; }
    trace_args+=(--traces "$trace")
  done
  "$PYTHON" scripts/build_active_catalog_onpolicy_preferences.py \
    "$FULL/active_catalog_sft_v4/${split}.jsonl" \
    "$FULL/active_catalog_sft_v4/${split}_evaluation_index.jsonl" \
    "$output" "${trace_args[@]}" \
    --expected-split "$split" --minimum-margin 0.001
done

"$PYTHON" - "$RUN" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
summaries = {
    split: json.loads(
        (root / "preferences_v2" / f"{split}.jsonl.summary.json").read_text()
    )
    for split in ("train", "val")
}
train = summaries["train"]
source_states = 128
preference_rate = train["preferences"] / source_states
passed = (
    train["preferences"] > 0
    and preference_rate >= 0.20
    and len(train["family_counts"]) >= 2
)
payload = {
    "schema_version": "sn7-onpolicy-rl-reentry-stage-a-v2",
    "status": "pass" if passed else "fail",
    "promotion_authorized": bool(passed),
    "minimum_preference_rate_proxy": 0.20,
    "observed_preference_rate_proxy": preference_rate,
    "train": train,
    "val": summaries["val"],
    "test_assets_read": False,
}
(root / "STAGE_A_DECISION_V2.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2))
if not passed:
    raise SystemExit(12)
PY
