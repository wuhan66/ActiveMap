#!/usr/bin/env bash
set -euo pipefail

# Validation-only V5-B replication. It cannot launch unless the first V5-B
# seed passed all operation-wise non-KEEP candidate-headroom gates. Each new
# seed is independently audited after training; no frozen test asset is read.
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
DATA_ROOT="${SN7_V5_OUTPUT:-${STORAGE_ROOT}/processed/sn7_v1/updater_v5_temporal_pair_trainval_r1}"
FIRST_HEADROOM="${DATA_ROOT}/nonkeep_candidate_headroom_val_v5b/summary.json"
BASE_CONFIG="${PROJECT_ROOT}/configs/updater/sn7_v5_temporal_explicit_change_seed20260817_server.yaml"
RUN_BASE="${STORAGE_ROOT}/runs/updater"
QUEUE_ROOT="${STORAGE_ROOT}/runs/updater/v5_temporal_explicit_change_trainval_r1_replications_20260818"
LOG_DIR="${STORAGE_ROOT}/logs"
SEEDS=(20260818 20260819)
TRAIN_GPUS=(0 1)
HEADROOM_GPU_GROUPS=("0,1" "2,3")

[[ -x "${PYTHON}" ]] || { echo "ActiveMap runtime not found: ${PYTHON}" >&2; exit 1; }
[[ -f "${BASE_CONFIG}" ]] || { echo "V5-B base configuration is missing" >&2; exit 1; }
[[ -f "${FIRST_HEADROOM}" ]] || { echo "V5-B preflight summary missing" >&2; exit 1; }
[[ ! -e "${QUEUE_ROOT}" ]] || { echo "Refusing to overwrite replication queue: ${QUEUE_ROOT}" >&2; exit 1; }

"${PYTHON}" - "${FIRST_HEADROOM}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("test_assets_read") is not False:
    raise SystemExit("V5-B preflight read test assets")
if not summary.get("gate", {}).get("passes_candidate_recovery_preflight"):
    raise SystemExit("V5-B preflight did not pass every real-edit gate")
for operation in ("ADD", "DELETE", "RESHAPE"):
    if not summary.get("gate", {}).get("passes_by_operation", {}).get(operation):
        raise SystemExit(f"V5-B preflight failed {operation}")
PY

for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$index]}"
  run_root="${RUN_BASE}/v5_temporal_explicit_change_trainval_r1_seed${seed}"
  [[ ! -e "${run_root}" ]] || { echo "Run root already exists: ${run_root}" >&2; exit 1; }
  gpu="${TRAIN_GPUS[$index]}"
  active="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${active//[[:space:]]/}" ]] || { echo "GPU ${gpu} is occupied: ${active}" >&2; exit 1; }
done

mkdir -p "${QUEUE_ROOT}/configs" "${LOG_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON}" - "${BASE_CONFIG}" "${QUEUE_ROOT}" "${FIRST_HEADROOM}" "${SEEDS[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import yaml

base_path = Path(sys.argv[1])
queue_root = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
configs = {}
for raw_seed in sys.argv[4:]:
    seed = int(raw_seed)
    payload = dict(base)
    payload["seed"] = seed
    payload["output_dir"] = f"/home/wh/ActiveMap/runs/updater/v5_temporal_explicit_change_trainval_r1_seed{seed}"
    payload["archive_dir"] = f"/home/wh/projects/activemap-v1/outputs/updater/v5_temporal_explicit_change_trainval_r1_seed{seed}"
    path = queue_root / "configs" / f"seed{seed}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    configs[str(seed)] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
(queue_root / "launch_receipt.json").write_text(json.dumps({
    "schema_version": "sn7-v5b-replication-queue-v1",
    "stage": "validation_only_updater_replications",
    "base_config": str(base_path.resolve()),
    "base_config_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
    "authorized_by_headroom": str(summary_path.resolve()),
    "authorized_by_headroom_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    "rendered_configs": configs,
    "test_assets_read": False,
}, indent=2) + "\n", encoding="utf-8")
PY

train_seed() {
  local seed="$1" gpu="$2" config="$3" run_root="$4" group="$5"
  local shard_dir="${DATA_ROOT}/headroom_val_v5b_seed${seed}_shards"
  local states="${DATA_ROOT}/selector_states_headroom_val_v5b_seed${seed}.jsonl"
  local headroom_dir="${DATA_ROOT}/nonkeep_candidate_headroom_val_v5b_seed${seed}"
  local train_log="${LOG_DIR}/train_sn7_v5_temporal_explicit_change_seed${seed}.log"
  local headroom_log="${LOG_DIR}/queue_sn7_v5b_headroom_seed${seed}.log"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m activemap.cli train-updater "${config}" \
    >"${train_log}" 2>&1
  [[ -f "${run_root}/metrics.json" && -f "${run_root}/best_quality.pt" ]] || {
    echo "Seed ${seed} ended without its required checkpoint" >&2
    return 1
  }
  SN7_V5_RUN="${run_root}" \
  SN7_V5_HEADROOM_CHECKPOINT="${run_root}/best_quality.pt" \
  SN7_V5_HEADROOM_SHARD_DIR="${shard_dir}" \
  SN7_V5_HEADROOM_STATES="${states}" \
  SN7_V5_HEADROOM_DIR="${headroom_dir}" \
  SN7_V5_HEADROOM_LOG_PREFIX="v5b_seed${seed}" \
  SN7_V5_HEADROOM_BOOTSTRAP_SEED="${seed}" \
  SN7_V5_HEADROOM_GPUS="${group}" \
  bash "${PROJECT_ROOT}/scripts/queue_sn7_v5_validation_headroom_parallel_hdpi.sh" \
    >"${headroom_log}" 2>&1
  [[ -f "${headroom_dir}/summary.json" ]] || {
    echo "Seed ${seed} ended without a headroom summary" >&2
    return 1
  }
}

pids=()
for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$index]}"
  config="${QUEUE_ROOT}/configs/seed${seed}.yaml"
  run_root="${RUN_BASE}/v5_temporal_explicit_change_trainval_r1_seed${seed}"
  train_seed "${seed}" "${TRAIN_GPUS[$index]}" "${config}" "${run_root}" "${HEADROOM_GPU_GROUPS[$index]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
(( status == 0 )) || exit "${status}"

"${PYTHON}" - "${QUEUE_ROOT}/completion_receipt.json" "${DATA_ROOT}" "${SEEDS[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
data_root = Path(sys.argv[2])
summaries = {}
for seed in sys.argv[3:]:
    path = data_root / f"nonkeep_candidate_headroom_val_v5b_seed{seed}" / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summaries[seed] = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "passes_candidate_recovery_preflight": bool(summary.get("gate", {}).get("passes_candidate_recovery_preflight")),
        "test_assets_read": summary.get("test_assets_read"),
    }
if not all(value["passes_candidate_recovery_preflight"] and value["test_assets_read"] is False for value in summaries.values()):
    raise SystemExit("at least one V5-B replication failed its validation-only headroom gate")
output.write_text(json.dumps({
    "schema_version": "sn7-v5b-replication-complete-v1",
    "stage": "validation_only_updater_replications",
    "seeds": summaries,
    "test_assets_read": False,
}, indent=2) + "\n", encoding="utf-8")
PY

echo "V5-B replications completed: ${QUEUE_ROOT}"
