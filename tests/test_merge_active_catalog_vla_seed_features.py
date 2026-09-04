import json
from pathlib import Path

import numpy as np

from scripts.merge_active_catalog_vla_seed_features import merge_sources


def _features(root: Path, offset: float) -> None:
    root.mkdir()
    np.save(root / "features.npy", np.asarray([[offset, 1.0]], dtype=np.float16))
    (root / "records.jsonl").write_text(
        json.dumps(
            {
                "example_id": "same",
                "task_id": "task",
                "split": "train",
                "oracle_use_tool": True,
                "consensus_mean_utility_gain": 0.2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "active-catalog-vla-frozen-features-v1",
                "feature_dim": 2,
                "pooling": "last",
                "utility_metadata": "best_acquire_minus_stop_utility",
                "source": {"sha256": "source"},
                "model": {"identifier": "model"},
                "adapter": {"weights_sha256": str(offset)},
                "test_assets_read": False,
            }
        ),
        encoding="utf-8",
    )


def test_merge_preserves_task_groups_and_separates_example_ids(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "output"
    _features(first, 1.0)
    _features(second, 2.0)

    summary = merge_sources(output, [(1, first), (2, second)])
    rows = [
        json.loads(line)
        for line in (output / "records.jsonl").read_text().splitlines()
    ]

    assert summary["model_seeds"] == [1, 2]
    assert {row["task_id"] for row in rows} == {"task"}
    assert {row["example_id"] for row in rows} == {"1:same", "2:same"}
