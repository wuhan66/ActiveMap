#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
MODEL="/home/wh/hf_models/Qwen3-VL-4B-Instruct"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"
MUNO_RUN="${STORAGE_ROOT}/runs/muno21_active_catalog_transfer_v1"
DATA="${STORAGE_ROOT}/processed/muno21_v2/agent"
VISUAL="${DATA}/active_catalog_validation_visual_v1"
STATES="${DATA}/selector_states_v1.jsonl"
EPISODES="${DATA}/episodes_train_val_v1.jsonl"
SFT="${RUN}/qwen3vl4b_weighted_replication_seed2/seed20260718/final"
UPDATER="${STORAGE_ROOT}/models/frozen_updater/muno21_v4_seed20260726/best_val_loss.pt"
SELECTION="${RUN}/conservative_rl_selection_n512_seed20260718/selection.json"
GPU="${GPU:-3}"
POLL_SECONDS="${POLL_SECONDS:-60}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
mkdir -p "${MUNO_RUN}"

until [[ -s "${SELECTION}" && -s "${VISUAL}/summary.json" ]]; do
  sleep "${POLL_SECONDS}"
done
[[ -s "${UPDATER}" && -s "${SFT}/adapter_config.json" ]] || {
  echo "missing frozen MUNO updater or SN7 SFT adapter" >&2
  exit 2
}

gpu_is_idle() {
  local pids
  pids="$(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${pids//[[:space:]]/}" ]]
}

run_controller() {
  local label="$1"
  local adapter="$2"
  local policy_mode="$3"
  local output="${MUNO_RUN}/closed_loop_${label}_n414"
  if [[ -s "${output}/evaluation/traces.jsonl" ]]; then
    return
  fi
  [[ ! -e "${output}" ]] || {
    echo "refusing partial MUNO controller output: ${output}" >&2
    return 3
  }
  until gpu_is_idle; do
    sleep "${POLL_SECONDS}"
  done
  "${PYTHON}" scripts/launch_active_catalog_closed_loop.py \
    "${MODEL}" "${adapter}" "${STATES}" "${EPISODES}" \
    "${VISUAL}/val_visual_prompts.jsonl" \
    "${VISUAL}/val_visual_index.jsonl" \
    "${output}" --gpu "${GPU}" --seed 20260718 \
    --policy-mode "${policy_mode}" --max-candidates 16 \
    --max-acquisitions 2 --max-new-tokens 64 \
    --bootstrap-repetitions 500 --monitor-interval 5
}

run_controller always_stop "${SFT}" always_stop
run_controller sft_seed20260718 "${SFT}" full_action

decision="$("${PYTHON}" -c \
  "import json; print(json.load(open('${SELECTION}'))['decision'])")"
selected="$("${PYTHON}" -c \
  "import json; print(json.load(open('${SELECTION}'))['selected_candidate'])")"
final_label="sft_seed20260718"
final_adapter="${SFT}"
if [[ "${decision}" == "promote_one" ]]; then
  assessment="${RUN}/promoted_${selected}_final_writeback_assessment.json"
  until [[ -s "${assessment}" ]]; do
    sleep "${POLL_SECONDS}"
  done
  final_decision="$("${PYTHON}" -c \
    "import json; print(json.load(open('${assessment}'))['decision'])")"
  if [[ "${final_decision}" == "promote_rl" ]]; then
    case "${selected}" in
      rl_lr2p5_kl005_bal50)
        final_adapter="${RUN}/contextual_rl_conservative_lr2p5e6_kl005_bal50_n4096/seed20260718/final"
        ;;
      rl_lr2p5_kl010_bal50)
        final_adapter="${RUN}/contextual_rl_conservative_lr2p5e6_kl010_bal50_n4096/seed20260718/final"
        ;;
      rl_lr5_kl010_bal50)
        final_adapter="${RUN}/contextual_rl_conservative_lr5e6_kl010_bal50_n4096/seed20260718/final"
        ;;
      rl_lr5_kl005_bal25)
        final_adapter="${RUN}/contextual_rl_conservative_lr5e6_kl005_bal25_n4096/seed20260718/final"
        ;;
      rl_lr2p5_kl010_bal25)
        final_adapter="${RUN}/contextual_rl_conservative_lr2p5e6_kl010_bal25_n4096/seed20260718/final"
        ;;
      *)
        echo "unrecognized promoted RL candidate: ${selected}" >&2
        exit 4
        ;;
    esac
    final_label="rl_${selected}_seed20260718"
    run_controller "${final_label}" "${final_adapter}" full_action
  elif [[ "${final_decision}" != "retain_sft" ]]; then
    echo "invalid final writeback decision: ${final_decision}" >&2
    exit 5
  fi
elif [[ "${decision}" != "stop_rl" ]]; then
  echo "invalid controller selection decision: ${decision}" >&2
  exit 6
fi

stop_trace="${MUNO_RUN}/closed_loop_always_stop_n414/evaluation/traces.jsonl"
final_trace="${MUNO_RUN}/closed_loop_${final_label}_n414/evaluation/traces.jsonl"
controller_comparison="${MUNO_RUN}/final_${final_label}_paired_vs_always_stop.json"
if [[ ! -s "${controller_comparison}" ]]; then
  "${PYTHON}" scripts/compare_active_catalog_closed_loop.py \
    "${controller_comparison}" --reference always_stop \
    --repetitions 5000 --seed 20260729 \
    --records "always_stop=${stop_trace}" \
    --records "final=${final_trace}"
fi

stop_input="${MUNO_RUN}/always_stop_writeback_input.jsonl"
final_input="${MUNO_RUN}/final_${final_label}_writeback_input.jsonl"
[[ -s "${stop_input}" ]] || \
  "${PYTHON}" scripts/convert_active_catalog_closed_loop_for_writeback.py \
    "${stop_trace}" "${stop_input}"
[[ -s "${final_input}" ]] || \
  "${PYTHON}" scripts/convert_active_catalog_closed_loop_for_writeback.py \
    "${final_trace}" "${final_input}"

run_writeback() {
  local label="$1"
  local input="$2"
  local output="${MUNO_RUN}/writeback_${label}"
  if [[ -s "${output}/evaluation/summary.json" ]]; then
    return
  fi
  [[ ! -e "${output}" ]] || {
    echo "refusing partial MUNO writeback output: ${output}" >&2
    return 7
  }
  until gpu_is_idle; do
    sleep "${POLL_SECONDS}"
  done
  "${PYTHON}" scripts/launch_active_catalog_writeback.py \
    "${UPDATER}" "${EPISODES}" "${input}" "${output}" \
    --gpu "${GPU}" --image-size 512 --threshold 0.5 \
    --protocol-name "muno21-${label}-active-catalog-writeback-v1" \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
    --monitor-interval 5
}

run_writeback always_stop "${stop_input}"
run_writeback "final_${final_label}" "${final_input}"

writeback_comparison="${MUNO_RUN}/final_${final_label}_writeback_paired_vs_always_stop.json"
if [[ ! -s "${writeback_comparison}" ]]; then
  "${PYTHON}" scripts/compare_agent_writebacks.py \
    "${MUNO_RUN}/writeback_always_stop/evaluation/writeback.jsonl" \
    "${MUNO_RUN}/writeback_final_${final_label}/evaluation/writeback.jsonl" \
    "${writeback_comparison}" --bootstrap 5000 --seed 20260729 \
    --group-key aoi_id
fi

for label in always_stop "final_${final_label}"; do
  official="${MUNO_RUN}/official_${label}"
  if [[ ! -s "${official}/official_metric_run_manifest.json" ]]; then
    bash scripts/run_muno21_official_graph_metrics.sh \
      "${MUNO_RUN}/writeback_${label}/evaluation/writeback.jsonl" \
      "${official}"
  fi
done

official_comparison="${MUNO_RUN}/final_${final_label}_official_paired_vs_always_stop.json"
if [[ ! -s "${official_comparison}" ]]; then
  "${PYTHON}" scripts/compare_muno21_official_metrics.py \
    "${MUNO_RUN}/official_always_stop/official_metrics.jsonl" \
    "${MUNO_RUN}/official_final_${final_label}/official_metrics.jsonl" \
    "${official_comparison}" --repetitions 5000 --seed 20260729
fi

"${PYTHON}" - "${MUNO_RUN}" "${final_label}" "${decision}" "${selected}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
final_label, stage1_decision, selected = sys.argv[2:]
artifacts = {
    "controller_comparison": root / f"final_{final_label}_paired_vs_always_stop.json",
    "writeback_comparison": root / f"final_{final_label}_writeback_paired_vs_always_stop.json",
    "official_comparison": root / f"final_{final_label}_official_paired_vs_always_stop.json",
}
manifest = {
    "schema_version": "muno21-frozen-sn7-policy-transfer-v1",
    "final_policy": final_label,
    "stage1_rl_decision": stage1_decision,
    "selected_rl_candidate": None if selected == "None" else selected,
    "split": "val",
    "artifacts": {
        name: {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in artifacts.items()
    },
    "test_assets_read": False,
}
(root / "final_transfer_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
PY
