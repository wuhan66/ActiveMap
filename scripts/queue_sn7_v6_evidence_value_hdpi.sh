#!/usr/bin/env bash
set -euo pipefail

# Validation-only V6 evidence-value recovery. The script trains only on
# source-disjoint internal V5 train partitions and leaves the sealed formal
# validation state files untouched until a separate audit phase.
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
V5_ROOT="${V5_MATCHED_RUN_ROOT:-${STORAGE_ROOT}/runs/paper_evidence/sn7_v5_matched_nonkeep_2x2_v2}"
V6_ROOT="${V6_EVIDENCE_VALUE_ROOT:-${STORAGE_ROOT}/runs/paper_evidence/sn7_v6_evidence_value_recovery_v1}"
LOG_ROOT="${STORAGE_ROOT}/logs/sn7_v6_evidence_value_recovery_v1"
SEEDS=(20260817 20260818 20260819)
GPUS=(0 1 2)

[[ -x "${PYTHON}" ]] || { echo "ActiveMap Python is missing: ${PYTHON}" >&2; exit 1; }
[[ -d "${V5_ROOT}" ]] || { echo "V5 root is missing: ${V5_ROOT}" >&2; exit 1; }
[[ ! -e "${V6_ROOT}" ]] || { echo "Refusing to overwrite V6 root: ${V6_ROOT}" >&2; exit 1; }
for gpu in "${GPUS[@]}"; do
  active="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${active//[[:space:]]/}" ]] || { echo "GPU ${gpu} is occupied: ${active}" >&2; exit 1; }
done

mkdir -p "${V6_ROOT}" "${LOG_ROOT}"
exec 9>"${V6_ROOT}/.queue.lock"
flock -n 9 || { echo "V6 evidence-value queue is already active" >&2; exit 1; }
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" - "${V6_ROOT}/protocol.json" "${V5_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "schema_version": "sn7-v6-evidence-value-recovery-v1",
            "source_v5_root": sys.argv[2],
            "seeds": [20260817, 20260818, 20260819],
            "train_only_tune": True,
            "formal_validation_used_for_selection": False,
            "test_assets_read": False,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

run_seed() {
  local seed="$1" gpu="$2"
  local states="${V5_ROOT}/selector_states/seed${seed}_internal/fit_tune.jsonl"
  local output="${V6_ROOT}/seed${seed}/run"
  [[ -f "${states}" ]] || { echo "missing internal states: ${states}" >&2; return 1; }
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" \
    "${PROJECT_ROOT}/scripts/train_evidence_value_head.py" \
    "${states}" "${output}" \
    --device cuda --seed "${seed}" --epochs 40 --patience 8 --batch-size 128 \
    --learning-rate 0.0003 --hidden-dim 128 --dropout 0.10 \
    --temperature 0.10 --utility-weight 1.0 --quality-weight 0.5 \
    --listwise-weight 1.0 --beneficial-weight 0.5 --unsafe-loss-weight 0.5 \
    --missed-loss-weight 0.5 --unsafe-penalty 0.10 --missed-penalty 0.05 \
    --maximum-false-call-rate 0.05 \
    >"${LOG_ROOT}/seed${seed}.log" 2>&1
}

pids=()
for index in "${!SEEDS[@]}"; do
  run_seed "${SEEDS[$index]}" "${GPUS[$index]}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done

"${PYTHON}" - "${V6_ROOT}/queue_status.json" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "status": "training_complete",
            "next_stage": "validation_policy_audit_required",
            "test_assets_read": False,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
echo "V6 evidence-value training complete: ${V6_ROOT}"
