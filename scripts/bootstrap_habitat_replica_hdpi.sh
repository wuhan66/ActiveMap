#!/usr/bin/env bash
set -euo pipefail

# Creates a self-contained, headless Habitat-Sim environment and downloads only
# the public scenes needed for the indoor RGB-D smoke and episode materializer.
CONDA_BIN="${CONDA_BIN:-/home/wh/miniconda3/bin/conda}"
ENV_PREFIX="${HABITAT_ENV_PREFIX:-/home/wh/ActiveMap/envs/habitat-sim}"
DATA_ROOT="${HABITAT_DATA_ROOT:-/home/wh/ActiveMap/datasets/habitat}"
SCENE_UIDS="${HABITAT_SCENE_UIDS:-habitat_test_scenes replica_cad_dataset}"

[[ -x "${CONDA_BIN}" ]] || {
  echo "Conda executable not found: ${CONDA_BIN}" >&2
  exit 1
}

if [[ ! -f "${ENV_PREFIX}/conda-meta/history" ]]; then
  # The current aihabitat Conda builds are published for Python 3.9.
  "${CONDA_BIN}" create --yes --prefix "${ENV_PREFIX}" -c conda-forge python=3.9 cmake=3.27 pip
  "${CONDA_BIN}" install --yes --prefix "${ENV_PREFIX}" \
    -c conda-forge -c aihabitat habitat-sim withbullet
fi

"${CONDA_BIN}" run --prefix "${ENV_PREFIX}" python - <<'PY'
import habitat_sim

print(f"habitat-sim import passed: {habitat_sim.__version__}")
PY

mkdir -p "${DATA_ROOT}"
"${CONDA_BIN}" run --prefix "${ENV_PREFIX}" \
  python -m habitat_sim.utils.datasets_download \
  --uids ${SCENE_UIDS} \
  --data-path "${DATA_ROOT}"

cat >"${DATA_ROOT}/ACTIVEMAP_BOOTSTRAP_COMPLETE.json" <<EOF
{
  "schema_version": "activemap-habitat-bootstrap-v1",
  "environment_prefix": "${ENV_PREFIX}",
  "scene_uids": "${SCENE_UIDS}",
  "role": "indoor_rgbd_navigation_development",
  "test_assets_read": false
}
EOF

echo "Habitat bootstrap complete: ${ENV_PREFIX}"
