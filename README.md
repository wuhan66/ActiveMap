# ActiveMap

Research code for **policy-relative downstream verification before safe
editable vector-map writeback**.

ActiveMap treats a map update as an executable decision, rather than a raster
change prediction alone. Starting from an editable prior and a shared
candidate-perception catalog, it produces a typed KEEP, ADD, DELETE, or
RESHAPE proposal, selectively admits optional downstream evidence, and uses a
separate Safe Commit rule to write or defer the terminal map transaction.

The current research scope is deliberately bounded to **single-episode
writeback after a shared candidate-perception front end**. It does not claim
segmentation state of the art, end-to-end visual-compute savings, long-horizon
persistent deployment, or VLM/RL superiority.

## Included

- Typed editable-map operations, vector transactions, and audit records.
- Prior-conditioned updater and evidence-selector reference implementations.
- Counterfactual utility construction, Safe Commit, and executable evaluation.
- Dataset construction protocols for SpaceNet 7, MUNO21, and Inria.
- Synthetic smoke data, schemas, reference configs, and unit tests.

Raw datasets, pretrained weights, paper checkpoints, private logs, and sealed
benchmark artifacts are intentionally excluded. Obtain third-party data and
models from their original providers under the applicable terms.

## Quick Start

Create the base environment, then install a PyTorch build compatible with the
local CUDA driver when neural training is required:

~~~bash
conda env create -f environment.yml
conda activate activemap
python -m pip install torch torchvision --index-url <matching-pytorch-index>
python -m pip install -e . --no-deps
bash scripts/run_smoke.sh
~~~

The smoke workflow creates synthetic updater and selector data, trains small
models, evaluates executable map outputs, and writes rollout traces under
ignored outputs paths. It does not download external data.

## External Data

~~~bash
bash scripts/download_sn7.sh /path/to/sn7 false
bash scripts/prepare_sn7.sh /path/to/sn7/train
bash scripts/download_muno21.sh /path/to/muno21
bash scripts/download_inria.sh /path/to/inria
~~~

See [docs/dataset_protocol.md](docs/dataset_protocol.md) for geographic/task
splits, typed-edit construction, candidate-evidence scope, and writeback
evaluation. [docs/metrics.md](docs/metrics.md) defines final-map quality,
false/missed edits, cost, and paired comparison metrics.

## Verification

~~~bash
python -m pytest -q \
  tests/test_cli.py \
  tests/test_counterfactual_builder.py \
  tests/test_episode_utility.py \
  tests/test_geometry_edits.py \
  tests/test_safe_commit.py \
  tests/test_rollout.py \
  tests/test_vector_map.py \
  tests/test_schema_export.py \
  tests/test_qc_split_safety.py
~~~

Detailed reproducibility and release boundaries are in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) and
[MODEL_DATA_AVAILABILITY.md](MODEL_DATA_AVAILABILITY.md).

## Release Status

This repository is a private pre-release while the associated manuscript is
under anonymous review. A public release requires a final third-party
data/model license audit, author metadata, citation metadata, and a checkpoint
availability matrix. Until then, the code is provided without a public software
license.
