#!/usr/bin/env bash
set -euo pipefail

WRITEBACK_JSONL="${1:?usage: run_muno21_official_graph_metrics.sh WRITEBACK_JSONL OUTPUT_ROOT}"
OUTPUT_ROOT="${2:?usage: run_muno21_official_graph_metrics.sh WRITEBACK_JSONL OUTPUT_ROOT}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
if [[ -f "${PROJECT_ROOT}/scripts/server_hdpi_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/scripts/server_hdpi_env.sh"
fi
GRAPH_PYTHON="${ACTIVEMAP_GRAPH_PYTHON:-${ACTIVEMAP_GRAPH_ENV}/bin/python}"
GO_BIN="${ACTIVEMAP_GO_BIN:-${ACTIVEMAP_GRAPH_ENV}/bin/go}"
OFFICIAL="${MUNO21_OFFICIAL_ROOT:-${PROJECT_ROOT}/external/muno21-official}"
DATA_ROOT="${MUNO21_DATA_ROOT:-${ACTIVEMAP_DATASET_ROOT}/muno21/extracted/mapupdate}"
PROCESSED="${MUNO21_PROCESSED_ROOT:-${ACTIVEMAP_PROCESSED_ROOT}/muno21_v2}"
SPLIT="${MUNO21_EVAL_SPLIT:-val}"
SIMPLIFY_TOLERANCE="${MUNO21_GRAPH_SIMPLIFY_TOLERANCE:-1.0}"
if [[ "${SPLIT}" == test ]]; then
  EPISODES="${MUNO21_EPISODES:-${PROCESSED}/agent/episodes_test_v1.jsonl}"
  REGIONS="${OUTPUT_ROOT}/test_regions.json"
  FROZEN_ARGS=(--split test --frozen-test)
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${GRAPH_PYTHON}" -c \
    'from activemap.frozen_test import assert_frozen_test_access; assert_frozen_test_access()'
elif [[ "${SPLIT}" == val ]]; then
  EPISODES="${MUNO21_EPISODES:-${PROCESSED}/agent/episodes_train_val_v1.jsonl}"
  REGIONS="${OUTPUT_ROOT}/validation_regions.json"
  FROZEN_ARGS=(--split val)
else
  echo "MUNO21_EVAL_SPLIT must be val or test" >&2
  exit 2
fi
DATA_SUMMARY="${PROCESSED}/updater/summary.json"
GRAPHS="${OUTPUT_ROOT}/graphs"
STATUS="${OUTPUT_ROOT}/exit_code.txt"
OFFICIAL_LOGS="${OUTPUT_ROOT}/official_metric_logs"
STRUCTURED="${OUTPUT_ROOT}/official_metrics.jsonl"

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$OFFICIAL_LOGS"
rm -f "$STATUS"
trap 'status=$?; printf "%s\n" "$status" > "$STATUS"' EXIT

[[ -x "$GRAPH_PYTHON" ]] || {
  echo "Missing graph-eval Python: $GRAPH_PYTHON" >&2
  echo "Create the isolated environment from pyproject optional dependency graph-eval." >&2
  exit 2
}
[[ -x "$GO_BIN" ]] || {
  echo "Missing Go runtime: $GO_BIN" >&2
  exit 2
}
[[ "$(git -C "$OFFICIAL" rev-parse HEAD)" == "4245b2a46485294878d3c26ccb252f5aac01acd7" ]] || {
  echo "MUNO21 official evaluator commit does not match the frozen protocol" >&2
  exit 2
}
"$GRAPH_PYTHON" -c "import skimage" || {
  echo "graph-eval Python lacks scikit-image" >&2
  exit 2
}

cd "$PROJECT_ROOT"
export PYTHONPATH="${OFFICIAL}/python:${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$(dirname "$GO_BIN"):${PATH}"
export GOPROXY=off
"$GRAPH_PYTHON" scripts/prepare_muno21_validation_regions.py \
  "$DATA_SUMMARY" "$REGIONS" "${FROZEN_ARGS[@]}"
"$GRAPH_PYTHON" scripts/export_muno21_writeback_graphs.py \
  "$WRITEBACK_JSONL" "$EPISODES" "$GRAPHS" \
  --simplify-tolerance "$SIMPLIFY_TOLERANCE" \
  "${FROZEN_ARGS[@]}"
"$GRAPH_PYTHON" scripts/audit_muno21_graph_coverage.py \
  "$DATA_ROOT/annotations.json" "$REGIONS" "$GRAPHS" \
  "$OUTPUT_ROOT/graph_coverage_audit.json" "${FROZEN_ARGS[@]}"

run_metric() {
  local output_dir="$1"
  local metric="$2"
  shift 2
  set +e
  "$@" 2>&1 | tee "${output_dir}/${metric}.log"
  local command_status="${PIPESTATUS[0]}"
  set -e
  printf '%s\n' "$command_status" >"${output_dir}/${metric}.exit_code.txt"
  if [[ "$command_status" -ne 0 ]]; then
    echo "Official metric ${metric} failed with exit code ${command_status}" >&2
    return "$command_status"
  fi
}

for inferred_dir in "$GRAPHS"/budget-*; do
  [[ -d "$inferred_dir" ]] || continue
  budget_name="$(basename "$inferred_dir")"
  budget_output="${OFFICIAL_LOGS}/${budget_name}"
  mkdir -p "$budget_output"
  (
    cd "$OFFICIAL/go"
    if [[ -s "$inferred_dir/scores.json" ]] &&
      [[ "$(cat "$budget_output/apls.exit_code.txt" 2>/dev/null)" == 0 ]]; then
      echo "Reusing completed APLS output: $inferred_dir/scores.json"
    else
      run_metric "$budget_output" apls "$GRAPH_PYTHON" metrics/apls.py \
        "$DATA_ROOT/annotations.json" "$inferred_dir" \
        "$DATA_ROOT/graphs/graphs" "$REGIONS"
    fi
    if [[ -s "$inferred_dir/geo.json" ]] &&
      [[ "$(cat "$budget_output/pixel_f1.exit_code.txt" 2>/dev/null)" == 0 ]]; then
      echo "Reusing completed Pixel-F1 output: $inferred_dir/geo.json"
    else
      run_metric "$budget_output" pixel_f1 "$GO_BIN" run metrics/geo.go \
        "$DATA_ROOT/annotations.json" "$inferred_dir" \
        "$DATA_ROOT/graphs/graphs" "$REGIONS"
    fi
    if [[ -s "$inferred_dir/error.json" ]] &&
      [[ "$(cat "$budget_output/error_rate.exit_code.txt" 2>/dev/null)" == 0 ]]; then
      echo "Reusing completed error-rate output: $inferred_dir/error.json"
    else
      run_metric "$budget_output" error_rate "$GO_BIN" run metrics/error_rate.go \
        "$DATA_ROOT/annotations.json" "$inferred_dir" \
        "$DATA_ROOT/graphs/graphs" "$REGIONS"
    fi
  )
done

"$GRAPH_PYTHON" scripts/collect_muno21_official_metrics.py \
  "$DATA_ROOT/annotations.json" "$REGIONS" "$EPISODES" "$GRAPHS" \
  "$STRUCTURED" "${FROZEN_ARGS[@]}"

"$GRAPH_PYTHON" - "$WRITEBACK_JSONL" "$REGIONS" "$OUTPUT_ROOT/graph_coverage_audit.json" \
  "$OFFICIAL_LOGS" "$OUTPUT_ROOT/official_metric_run_manifest.json" \
  "$(git -C "$OFFICIAL" rev-parse HEAD)" "$SPLIT" "$STRUCTURED" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


writeback, regions, coverage, logs_root, output = map(Path, sys.argv[1:6])
commit = sys.argv[6]
split = sys.argv[7]
structured = Path(sys.argv[8])
files = sorted(path for path in logs_root.rglob("*") if path.is_file())
budgets = sorted(path for path in logs_root.glob("budget-*") if path.is_dir())
exit_files = sorted(logs_root.rglob("*.exit_code.txt"))
manifest = {
    "schema_version": "muno21-official-graph-metric-run-v1",
    "official_commit": commit,
    "split": split,
    "inputs": {
        str(path): sha256(path) for path in (writeback, regions, coverage)
    },
    "metric_artifacts": {
        str(path.relative_to(logs_root)): sha256(path) for path in files
    },
    "budgets": [path.name for path in budgets],
    "complete": bool(budgets) and len(exit_files) == 3 * len(budgets) and all(
        path.read_text(encoding="utf-8").strip() == "0"
        for path in exit_files
    ),
    "structured_metrics": {
        "path": str(structured.resolve()),
        "sha256": sha256(structured),
    },
    "test_assets_read": split == "test",
}
output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

echo "[$(date --iso-8601=seconds)] official ${SPLIT} APLS/PixelF1/error-rate complete"
