#!/usr/bin/env bash
# Shared paths and GPU policy for hdpi-sys1. Source this file before server jobs.
export ACTIVEMAP_PROJECT_ROOT="${ACTIVEMAP_PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
export ACTIVEMAP_STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
# Historical launchers use ACTIVEMAP_DATA_ROOT as the experiment storage root.
export ACTIVEMAP_DATA_ROOT="${ACTIVEMAP_DATA_ROOT:-${ACTIVEMAP_STORAGE_ROOT}}"
export ACTIVEMAP_DATASET_ROOT="${ACTIVEMAP_DATASET_ROOT:-${ACTIVEMAP_STORAGE_ROOT}/datasets}"
export ACTIVEMAP_PROCESSED_ROOT="${ACTIVEMAP_PROCESSED_ROOT:-${ACTIVEMAP_STORAGE_ROOT}/processed}"
export ACTIVEMAP_RUN_ROOT="${ACTIVEMAP_RUN_ROOT:-${ACTIVEMAP_STORAGE_ROOT}/runs}"
export ACTIVEMAP_LOG_ROOT="${ACTIVEMAP_LOG_ROOT:-${ACTIVEMAP_STORAGE_ROOT}/logs}"
export ACTIVEMAP_MODEL_ROOT="${ACTIVEMAP_MODEL_ROOT:-/home/wh/hf_models}"
export ACTIVEMAP_GIS_ENV="${ACTIVEMAP_GIS_ENV:-${ACTIVEMAP_STORAGE_ROOT}/envs/activemap-gis}"
export ACTIVEMAP_AGENT_ENV="${ACTIVEMAP_AGENT_ENV:-${ACTIVEMAP_STORAGE_ROOT}/envs/activemap-agent}"
export ACTIVEMAP_GRAPH_ENV="${ACTIVEMAP_GRAPH_ENV:-${ACTIVEMAP_STORAGE_ROOT}/envs/activemap-graph}"
export ACTIVEMAP_GO_BIN="${ACTIVEMAP_GO_BIN:-${ACTIVEMAP_GRAPH_ENV}/bin/go}"

# ActiveMap may use up to five free GPUs. Every launcher still checks current
# process ownership and availability before starting work.
export ACTIVEMAP_GPU_IDS="${ACTIVEMAP_GPU_IDS:-0,1,2,3,4,5,6,7}"
export ACTIVEMAP_MAX_GPUS="${ACTIVEMAP_MAX_GPUS:-5}"
