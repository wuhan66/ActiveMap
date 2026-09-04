#!/usr/bin/env bash
set -euo pipefail

# Resume only the policy and writeback stages after the test-split schema fix.
# The expensive frozen episode/state construction is reused byte-for-byte.
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
REGISTRY="${REGISTRY:-${PROJECT_ROOT}/configs/experiments/sn7_step0_frozen_registry_v3_cap20.yaml}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/step0_frozen_test_v3_cap20}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/logs/sn7_step0_frozen_test_v3_cap20}"
FAILED_LEDGER="${FAILED_LEDGER:-${STORAGE_ROOT}/artifacts/paper_results/frozen_test_access/sn7_step0_v3_cap20.json}"
UPDATER="${UPDATER:-${STORAGE_ROOT}/models/frozen_updater/sn7_v4_hierarchical_vector_change_seed20260716/best_quality.pt}"
SEEDS=(${SEEDS:-20260730 20260731 20260801})
GPUS=(${GPUS:-1 3 4})
ARCHIVE_ROOT="${RUN_ROOT}/infrastructure_failed_split_schema_20260808"
INTERRUPTED_ARCHIVE_ROOT="${RUN_ROOT}/infrastructure_interrupted_writeback_20260808"
RECEIPT="${RUN_ROOT}/infrastructure_recovery_receipt.json"
SN7_TEST_EPISODES="${RUN_ROOT}/episodes_test.jsonl"
STATES="${RUN_ROOT}/states_test_step0.jsonl"
PROGRESS="${RUN_ROOT}/states_test_step0.progress.json"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
"${PYTHON}" -c 'from activemap.frozen_test import assert_frozen_test_access; assert_frozen_test_access()'
for path in "${REGISTRY}" "${FAILED_LEDGER}" "${UPDATER}" "${SN7_TEST_EPISODES}" "${STATES}" "${PROGRESS}"; do
  test -s "${path}"
done
(( ${#GPUS[@]} >= ${#SEEDS[@]} )) || { echo "one GPU per seed is required" >&2; exit 42; }
for gpu in "${GPUS[@]}"; do
  [[ "${gpu}" != "0" && "${gpu}" != "2" ]] || { echo "GPU ${gpu} is excluded" >&2; exit 43; }
done
"${PYTHON}" - "${FAILED_LEDGER}" "${PROGRESS}" "${STATES}" "${SN7_TEST_EPISODES}" <<'PY'
import json, sys
from pathlib import Path
ledger, progress, states, episodes = map(Path, sys.argv[1:])
l = json.loads(ledger.read_text())
p = json.loads(progress.read_text())
assert l["status"] == "command_failed" and int(l["returncode"]) != 0
assert p["status"] == "complete" and p["splits"] == ["test"]
assert p["episodes"] == 6573 and p["samples"] == 23223
assert states.stat().st_size > 0 and episodes.stat().st_size > 0
PY
for seed in "${SEEDS[@]}"; do
  log="${LOG_ROOT}/seed${seed}_notool_evaluation.log"
  test -s "${log}"
  grep -q "literal_error" "${log}"
  grep -q "split" "${log}"
done

if [[ -e "${RECEIPT}" ]]; then
  "${PYTHON}" - "${RECEIPT}" "${RUN_ROOT}" "${SEEDS[@]}" <<'PY'
import json, sys
from pathlib import Path
receipt, root = map(Path, sys.argv[1:3])
seeds = sys.argv[3:]
payload = json.loads(receipt.read_text())
assert payload["status"] == "started"
payload["launcher_retry_count"] = int(payload.get("launcher_retry_count", 0)) + 1
receipt.write_text(json.dumps(payload, indent=2) + "\n")
PY
else
  [[ ! -e "${ARCHIVE_ROOT}" ]] || { echo "failure archive exists without receipt: ${ARCHIVE_ROOT}" >&2; exit 45; }
  mkdir -p "${ARCHIVE_ROOT}" "${LOG_ROOT}"
  for seed in "${SEEDS[@]}"; do
    if [[ -e "${RUN_ROOT}/seed${seed}" ]]; then
      mv "${RUN_ROOT}/seed${seed}" "${ARCHIVE_ROOT}/seed${seed}"
    fi
  done
  "${PYTHON}" - "${FAILED_LEDGER}" "${REGISTRY}" "${STATES}" "${SN7_TEST_EPISODES}" "${PROJECT_ROOT}/src/activemap/agent/active_catalog_joint.py" "${RECEIPT}" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
failed, registry, states, episodes, schema_source, receipt = map(Path, sys.argv[1:])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
payload = {
    "schema_version": "sn7-step0-frozen-test-infrastructure-recovery-v1",
    "status": "started",
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "reason": "serialization schema rejected the already frozen test split",
    "scientific_protocol_changed": False,
    "failed_ledger": str(failed),
    "failed_ledger_sha256": sha(failed),
    "registry_sha256": sha(registry),
    "states_sha256": sha(states),
    "episodes_sha256": sha(episodes),
    "schema_source_sha256": sha(schema_source),
}
receipt.write_text(json.dumps(payload, indent=2) + "\n")
PY
fi

archive_interrupted_writeback() {
  local seed="$1" variant="$2" root="$3"
  local process_result="${root}/process_result.json"
  if [[ ! -e "${root}" || -s "${process_result}" ]]; then
    return 0
  fi

  local evaluation_root="${root}/evaluation"
  local progress="${evaluation_root}/progress.json"
  [[ -s "${progress}" ]] || {
    echo "writeback root exists without a progress receipt: ${root}" >&2
    exit 47
  }
  [[ ! -s "${evaluation_root}/writeback.jsonl" && ! -s "${evaluation_root}/summary.json" ]] || {
    echo "refusing to archive a writeback with final metric files: ${root}" >&2
    exit 48
  }

  local destination="${INTERRUPTED_ARCHIVE_ROOT}/seed${seed}/${variant}/writeback"
  [[ ! -e "${destination}" ]] || {
    echo "interrupted writeback archive already exists: ${destination}" >&2
    exit 49
  }
  mkdir -p "$(dirname "${destination}")"
  mv "${root}" "${destination}"
  "${PYTHON}" - "${destination}" "${seed}" "${variant}" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
progress_path = root / "evaluation" / "progress.json"
progress = json.loads(progress_path.read_text())
if progress.get("status") not in {"running", "inference_complete"}:
    raise SystemExit(f"unexpected interrupted progress status: {progress.get('status')}")
files = sorted(path for path in root.rglob("*") if path.is_file())
payload = {
    "schema_version": "sn7-writeback-interruption-receipt-v1",
    "status": "archived_before_exact_protocol_restart",
    "archived_at_utc": datetime.now(timezone.utc).isoformat(),
    "seed": int(sys.argv[2]),
    "variant": sys.argv[3],
    "scientific_protocol_changed": False,
    "partial_rows_processed": int(progress["rows_processed"]),
    "rows_total": int(progress["rows_total"]),
    "file_count": len(files),
    "progress_sha256": hashlib.sha256(progress_path.read_bytes()).hexdigest(),
}
(root / "INTERRUPTION_RECEIPT.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
}

run_seed() {
  local seed="$1" gpu="$2"
  local selector="${STORAGE_ROOT}/runs/selector/sn7_step0_two_stage_v2_seed${seed}/edit_utility_seed${seed}/best.pt"
  local belief="${STORAGE_ROOT}/runs/agent/sn7_step0_tool_belief_gated_seed${seed}/best_promoted.pt"
  local adapter="${STORAGE_ROOT}/runs/agent/sn7_step0_post_tool_adapter_updated_seed${seed}/best.pt"
  local gate_root="${STORAGE_ROOT}/runs/sn7_active_catalog/step0_tool_need_gate_seed${seed}_benefit_gate_v1"
  local gate="${gate_root}/gate.joblib" gate_summary="${gate_root}/summary.json"
  for path in "${selector}" "${belief}" "${adapter}" "${gate}" "${gate_summary}"; do test -f "${path}"; done

  for variant in notool forced benefit; do
    local root="${RUN_ROOT}/seed${seed}/${variant}"
    local eval_root="${root}/evaluation"
    local tool_args=()
    if [[ "${variant}" == "forced" ]]; then
      tool_args=(--tool-mode forced --belief-mode recurrent --tool-belief-checkpoint "${belief}" --post-tool-action-adapter "${adapter}" --tool-artifact-root "${root}/tool_artifacts")
    elif [[ "${variant}" == "benefit" ]]; then
      tool_args=(--tool-mode selective --belief-mode recurrent --tool-belief-checkpoint "${belief}" --post-tool-action-adapter "${adapter}" --tool-gate "${gate}" --tool-gate-summary "${gate_summary}" --tool-artifact-root "${root}/tool_artifacts")
    fi
    if [[ ! -s "${eval_root}/edit_utility.jsonl" ]]; then
      [[ ! -e "${eval_root}" ]] || { echo "incomplete evaluation root: ${eval_root}" >&2; exit 46; }
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/evaluate_active_catalog_closed_loop_baselines.py \
        "${STATES}" "${eval_root}" --episodes "${SN7_TEST_EPISODES}" \
        --learned-selector "edit_utility=${selector}" --policy edit_utility \
        --seed "${seed}" --device cuda:0 --split test --frozen-test \
        --max-candidates 16 --max-acquisitions 2 --tool-out-size 256 \
        --bootstrap-repetitions 0 --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
        "${tool_args[@]}" >"${LOG_ROOT}/recovery_seed${seed}_${variant}_evaluation.log" 2>&1
    fi
    if [[ ! -s "${root}/writeback_input.jsonl" ]]; then
      "${PYTHON}" scripts/convert_active_catalog_closed_loop_for_writeback.py \
        "${eval_root}/edit_utility.jsonl" "${root}/writeback_input.jsonl" \
        --split test --evidence-mode last --frozen-test \
        >"${LOG_ROOT}/recovery_seed${seed}_${variant}_conversion.log" 2>&1
    fi
    if [[ -s "${root}/writeback/process_result.json" ]]; then
      "${PYTHON}" - "${root}/writeback/process_result.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p["status"] == "completed" and int(p["returncode"]) == 0
PY
      continue
    fi
    archive_interrupted_writeback "${seed}" "${variant}" "${root}/writeback"
    [[ ! -e "${root}/writeback" ]] || { echo "incomplete writeback root: ${root}/writeback" >&2; exit 50; }
    "${PYTHON}" scripts/launch_active_catalog_writeback.py \
      "${UPDATER}" "${SN7_TEST_EPISODES}" "${root}/writeback_input.jsonl" \
      "${root}/writeback" --gpu "${gpu}" --python "${PYTHON}" --image-size 512 \
      --threshold 0.5 --delta-margin 0.15 --evidence-fusion confidence_weighted \
      --protocol-name sn7-step0-frozen-test-v3-cap20 --split test --frozen-test \
      --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" --monitor-interval 5 \
      >"${LOG_ROOT}/recovery_seed${seed}_${variant}_writeback.log" 2>&1
  done
}

pids=()
for index in "${!SEEDS[@]}"; do run_seed "${SEEDS[$index]}" "${GPUS[$index]}" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 )) || exit "${status}"

summary_args=(); notool_writebacks=(); forced_writebacks=(); benefit_writebacks=()
for seed in "${SEEDS[@]}"; do
  summary_args+=(--notool "${seed}=${RUN_ROOT}/seed${seed}/notool/evaluation/edit_utility.jsonl" --forced "${seed}=${RUN_ROOT}/seed${seed}/forced/evaluation/edit_utility.jsonl" --benefit "${seed}=${RUN_ROOT}/seed${seed}/benefit/evaluation/edit_utility.jsonl")
  notool_writebacks+=(--baseline "${seed}=${RUN_ROOT}/seed${seed}/notool/writeback/evaluation/writeback.jsonl")
  forced_writebacks+=(--baseline "${seed}=${RUN_ROOT}/seed${seed}/forced/writeback/evaluation/writeback.jsonl")
  benefit_writebacks+=(--candidate "${seed}=${RUN_ROOT}/seed${seed}/benefit/writeback/evaluation/writeback.jsonl")
done
if [[ ! -s "${RUN_ROOT}/three_policy_summary.json" ]]; then
  "${PYTHON}" scripts/summarize_sn7_step0_three_policy.py "${RUN_ROOT}/three_policy_summary.json" "${summary_args[@]}" --split test --frozen-test --bootstrap-repetitions 10000 --bootstrap-seed 20260729
else
  "${PYTHON}" - "${RUN_ROOT}/three_policy_summary.json" <<'PY'
import json, sys
row = json.load(open(sys.argv[1]))
assert row["schema_version"] == "sn7-selective-tool-frontier-three-seed-v1"
assert row["seeds"] == [20260730, 20260731, 20260801]
assert set(row["variants"]) == {"notool", "forced", "benefit"}
assert row["protocol"]["split"] == "test"
assert row["protocol"]["test_assets_read"] is True
PY
fi
if [[ ! -s "${RUN_ROOT}/frontier_promotion.json" ]]; then
  "${PYTHON}" scripts/assess_sn7_step0_frontier_promotion.py "${RUN_ROOT}/three_policy_summary.json" "${RUN_ROOT}/frontier_promotion.json" --min-seeds 3 --expected-split test
fi
if [[ ! -s "${RUN_ROOT}/benefit_vs_notool.json" ]]; then
  "${PYTHON}" scripts/aggregate_active_catalog_tool_writebacks.py "${RUN_ROOT}/benefit_vs_notool.json" "${notool_writebacks[@]}" "${benefit_writebacks[@]}" --split test --frozen-test --repetitions 10000 --seed 20260729
fi
if [[ ! -s "${RUN_ROOT}/benefit_vs_forced.json" ]]; then
  "${PYTHON}" scripts/aggregate_active_catalog_tool_writebacks.py "${RUN_ROOT}/benefit_vs_forced.json" "${forced_writebacks[@]}" "${benefit_writebacks[@]}" --split test --frozen-test --repetitions 10000 --seed 20260729
fi
if [[ ! -s "${RUN_ROOT}/writeback_promotion.json" ]]; then
  "${PYTHON}" scripts/assess_active_catalog_tool_writeback_promotion.py "${RUN_ROOT}/frontier_promotion.json" "${RUN_ROOT}/benefit_vs_notool.json" "${RUN_ROOT}/benefit_vs_forced.json" "${RUN_ROOT}/writeback_promotion.json" --min-seeds 3
fi
"${PYTHON}" - "${RUN_ROOT}" "${RECEIPT}" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
root, receipt = map(Path, sys.argv[1:])
promotion = json.loads((root / "writeback_promotion.json").read_text())
(root / "COMPLETE.json").write_text(json.dumps({"schema_version":"sn7-step0-frozen-test-complete-v2","promotion_passed":bool(promotion["promote"]),"test_assets_read":True}, indent=2) + "\n")
payload = json.loads(receipt.read_text()); payload.update(status="complete", finished_at_utc=datetime.now(timezone.utc).isoformat()); receipt.write_text(json.dumps(payload, indent=2) + "\n")
PY
echo "completed SN7 Step-0 frozen test v3 cap20 infrastructure recovery"
