import json

import pytest

from scripts.aggregate_agent_writeback_pairs import aggregate
from scripts.compare_agent_writebacks import HIGHER_IS_BETTER, LOWER_IS_BETTER


def _row(task, aoi, budget, value):
    row = {
        "task_id": task,
        "aoi_id": aoi,
        "budget": budget,
        "target": "COMMIT:ADD",
        "split": "val",
        "test_assets_read": False,
    }
    row.update({name: value for name in HIGHER_IS_BETTER})
    row.update({name: 1.0 - value for name in LOWER_IS_BETTER})
    return row


def _trace(tmp_path, name, value):
    path = tmp_path / f"{name}.jsonl"
    rows = [
        _row(f"{aoi}-{budget}", aoi, budget, value)
        for aoi in ("a", "b")
        for budget in (1.5, 3.0, 4.5)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def test_writeback_pairs_use_seed_matched_sft_and_shared_aoi_draws(tmp_path):
    pairs = [
        (
            model_seed,
            _trace(tmp_path, f"sft-{model_seed}", 0.2),
            _trace(tmp_path, f"rl-{model_seed}", 0.4),
        )
        for model_seed in (1, 2, 3)
    ]

    result = aggregate(pairs, repetitions=100, seed=8)

    assert result["seed_count"] == 3
    assert result["seed_matched_references"] is True
    assert result["shared_aoi_resampling_across_model_seeds"] is True
    assert result["candidate_minus_seed_matched_sft"]["raster_iou_auc"][
        "observed_delta"
    ] == pytest.approx(0.2)
    assert result["seed_variation"]["raster_iou_auc"]["sample_std_delta"] == 0.0


def test_writeback_pairs_reject_cross_seed_protocol_drift(tmp_path):
    sft1 = _trace(tmp_path, "sft1", 0.2)
    rl1 = _trace(tmp_path, "rl1", 0.4)
    sft2 = _trace(tmp_path, "sft2", 0.2)
    rl2 = _trace(tmp_path, "rl2", 0.4)
    for path in (sft2, rl2):
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        rows[0]["aoi_id"] = "changed"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    with pytest.raises(ValueError, match="cross-seed writeback protocol mismatch"):
        aggregate(
            [(1, sft1, rl1), (2, sft2, rl2)],
            repetitions=10,
            seed=1,
        )
