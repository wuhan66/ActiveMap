#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"
GPU="${GPU:-4}"
POLL_SECONDS="${POLL_SECONDS:-60}"
FIFTH_RESULT="${RUN}/closed_loop_rl_conservative_lr2p5e6_kl010_bal25_n4096_seed20260718_n512_paired.json"
FIFTH_SESSION="sn7_vla_rl_conservative_fifth_n4096"
SMOKE="${RUN}/react_style_qwen_seed20260718_smoke_v2"
FULL="${RUN}/react_style_qwen_seed20260718_n512_v2"
SFT="${RUN}/closed_loop_sft_seed20260718_n512"
DATA="${STORAGE_ROOT}/processed/sn7_v1/agent/sequential_selector_v1/full"
EPISODES="${DATA}/closed_loop_v1/episodes_val.jsonl"
UPDATER="${STORAGE_ROOT}/runs/updater/v4_hierarchical_vector_change_scratch_seed20260716/best_quality.pt"
SFT_WRITEBACK_INPUT="${RUN}/closed_loop_sft_seed20260718_n512_writeback_input.jsonl"
REACT_WRITEBACK_INPUT="${RUN}/react_style_qwen_seed20260718_n512_v2_writeback_input.jsonl"
SFT_WRITEBACK="${RUN}/closed_loop_sft_seed20260718_n512_writeback"
REACT_WRITEBACK="${RUN}/react_style_qwen_seed20260718_n512_v2_writeback"
REACT_VISUALS="${STORAGE_ROOT}/runs/paper_visuals/react_style_qwen_seed20260718_n512_v2"
PAIRED_VISUALS="${STORAGE_ROOT}/runs/paper_visuals/sft_vs_react_seed20260718_n512_v2"
CONTROLLER_TABLE="${RUN}/controller_validation_diagnostic_n512_seed20260718/controller_table.json"
RL_SELECTION="${RUN}/conservative_rl_selection_n512_seed20260718/selection.json"
REACT_COMPARISON="${RUN}/react_style_qwen_seed20260718_n512_v2_paired_vs_sft.json"
EVIDENCE_MANIFEST="${RUN}/paper_evidence_manifest_seed20260718_n512"

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"

until [[ -s "${FIFTH_RESULT}" ]] && ! tmux has-session -t "${FIFTH_SESSION}" 2>/dev/null; do
  sleep "${POLL_SECONDS}"
done

gpu_is_idle() {
  local pids
  pids="$(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${pids//[[:space:]]/}" ]]
}

until gpu_is_idle; do
  sleep "${POLL_SECONDS}"
done

if [[ ! -s "${SMOKE}/evaluation/summary.json" ]]; then
  GPU="${GPU}" bash scripts/run_sn7_react_baseline.sh smoke
fi
if [[ ! -s "${SMOKE}/smoke_assessment.json" ]]; then
  "${PYTHON}" scripts/assess_sn7_react_smoke.py \
    "${SMOKE}/evaluation/summary.json" \
    "${SMOKE}/evaluation/traces.jsonl" \
    "${SMOKE}/smoke_assessment.json"
fi
smoke_passed="$("${PYTHON}" -c \
  "import json; print(str(json.load(open('${SMOKE}/smoke_assessment.json'))['passed']).lower())")"
[[ "${smoke_passed}" == "true" ]] || {
  echo "ReAct v2 smoke assessment did not pass" >&2
  exit 8
}

until gpu_is_idle; do
  sleep "${POLL_SECONDS}"
done

if [[ ! -s "${FULL}/evaluation/summary.json" ]]; then
  GPU="${GPU}" bash scripts/run_sn7_react_baseline.sh full
fi

[[ -s "${SFT}/evaluation/traces.jsonl" ]] || {
  echo "missing matched SFT trace: ${SFT}" >&2
  exit 5
}
if [[ ! -s "${REACT_COMPARISON}" ]]; then
  "${PYTHON}" scripts/compare_active_catalog_closed_loop.py \
    "${REACT_COMPARISON}" \
    --reference sft --repetitions 2000 --seed 20260718 \
    --records "sft=${SFT}/evaluation/traces.jsonl" \
    --records "react=${FULL}/evaluation/traces.jsonl"
fi

if [[ ! -s "${SFT_WRITEBACK_INPUT}" ]]; then
  "${PYTHON}" scripts/convert_active_catalog_closed_loop_for_writeback.py \
    "${SFT}/evaluation/traces.jsonl" "${SFT_WRITEBACK_INPUT}"
fi
if [[ ! -s "${REACT_WRITEBACK_INPUT}" ]]; then
  "${PYTHON}" scripts/convert_active_catalog_closed_loop_for_writeback.py \
    "${FULL}/evaluation/traces.jsonl" "${REACT_WRITEBACK_INPUT}"
fi

until gpu_is_idle; do
  sleep "${POLL_SECONDS}"
done
if [[ ! -s "${SFT_WRITEBACK}/evaluation/writeback.jsonl" ]]; then
  [[ ! -e "${SFT_WRITEBACK}" ]] || {
    echo "refusing partial SFT writeback: ${SFT_WRITEBACK}" >&2
    exit 6
  }
  "${PYTHON}" scripts/launch_active_catalog_writeback.py \
    "${UPDATER}" "${EPISODES}" "${SFT_WRITEBACK_INPUT}" "${SFT_WRITEBACK}" \
    --gpu "${GPU}" --image-size 512 --threshold 0.5 \
    --protocol-name sn7-sft-n512-vector-writeback-v1 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
    --monitor-interval 5
fi

until gpu_is_idle; do
  sleep "${POLL_SECONDS}"
done
if [[ ! -s "${REACT_WRITEBACK}/evaluation/writeback.jsonl" ]]; then
  [[ ! -e "${REACT_WRITEBACK}" ]] || {
    echo "refusing partial ReAct writeback: ${REACT_WRITEBACK}" >&2
    exit 7
  }
  "${PYTHON}" scripts/launch_active_catalog_writeback.py \
    "${UPDATER}" "${EPISODES}" "${REACT_WRITEBACK_INPUT}" "${REACT_WRITEBACK}" \
    --gpu "${GPU}" --image-size 512 --threshold 0.5 \
    --protocol-name sn7-react-n512-vector-writeback-v1 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
    --monitor-interval 5
fi

if [[ ! -s "${REACT_VISUALS}/summary.json" ]]; then
  "${PYTHON}" scripts/render_active_catalog_closed_loop_examples.py \
    "${EPISODES}" "${FULL}/evaluation/traces.jsonl" \
    "${REACT_WRITEBACK}/evaluation/writeback.jsonl" "${REACT_VISUALS}" \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
    --count 8
fi

if [[ ! -s "${PAIRED_VISUALS}/summary.json" ]]; then
  "${PYTHON}" scripts/render_active_catalog_paired_examples.py \
    "${EPISODES}" "${PAIRED_VISUALS}" --candidate react \
    --method "sft=${SFT}/evaluation/traces.jsonl,${SFT_WRITEBACK}/evaluation/writeback.jsonl" \
    --method "react=${FULL}/evaluation/traces.jsonl,${REACT_WRITEBACK}/evaluation/writeback.jsonl" \
    --per-edit-success 2 --per-edit-failure 1 --per-edit-disagreement 1 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
    --min-visual-fraction 0.01
fi

until [[ -s "${CONTROLLER_TABLE}" && -s "${RL_SELECTION}" ]]; do
  sleep "${POLL_SECONDS}"
done

if [[ ! -s "${EVIDENCE_MANIFEST}/manifest.json" ]]; then
  "${PYTHON}" scripts/build_sn7_paper_evidence_manifest.py \
    "${EVIDENCE_MANIFEST}" --expected-records 512 --minimum-images 8 \
    --json "controller_table=${CONTROLLER_TABLE}" \
    --json "rl_selection=${RL_SELECTION}" \
    --json "react_vs_sft=${REACT_COMPARISON}" \
    --json "sft_writeback_summary=${SFT_WRITEBACK}/evaluation/summary.json" \
    --json "react_writeback_summary=${REACT_WRITEBACK}/evaluation/summary.json" \
    --json "paired_visual_summary=${PAIRED_VISUALS}/summary.json" \
    --jsonl "sft_trace=${SFT}/evaluation/traces.jsonl" \
    --jsonl "react_trace=${FULL}/evaluation/traces.jsonl" \
    --jsonl "sft_writeback=${SFT_WRITEBACK}/evaluation/writeback.jsonl" \
    --jsonl "react_writeback=${REACT_WRITEBACK}/evaluation/writeback.jsonl" \
    --image-dir "paired_visuals=${PAIRED_VISUALS}"
fi
