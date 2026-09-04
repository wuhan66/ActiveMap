import json
from pathlib import Path

import pytest

from activemap.agent.identifiers import public_task_id
from activemap.models import (
    CandidateHypothesis,
    EditOperation,
    EditRecord,
    EpisodeRecord,
    GeoJSONGeometry,
)
from scripts.collect_muno21_official_metrics import collect


def _episode(index: int, operation: EditOperation) -> EpisodeRecord:
    geometry = None
    if operation is EditOperation.ADD:
        geometry = GeoJSONGeometry(
            type="LineString",
            coordinates=[[0.0, 0.0], [1.0, 1.0]],
        )
    return EpisodeRecord(
        episode_id=f"muno21-{index}-p0-{operation.value.lower()}",
        split="val",
        source_dataset="MUNO21",
        map_before="before.graph",
        target_map="target.graph",
        hypothesis=CandidateHypothesis(
            op=EditOperation.KEEP,
            object_id=f"road-{index}",
            source="editable_prior",
        ),
        evidence_catalog=[],
        gt_edit=EditRecord(
            op=operation,
            object_id=f"road-{index}",
            geometry=geometry,
        ),
        is_synthetic=False,
        derivation_version="test",
    )


def test_collects_per_change_scores_and_honest_aggregate_error(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            [
                {"Tags": ["nochange"], "Cluster": {"Region": "city"}},
                {"Tags": ["constructed"], "Cluster": {"Region": "city"}},
            ]
        ),
        encoding="utf-8",
    )
    regions = tmp_path / "regions.json"
    regions.write_text('["city"]', encoding="utf-8")
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(
        "\n".join(
            [
                _episode(0, EditOperation.KEEP).model_dump_json(),
                _episode(1, EditOperation.ADD).model_dump_json(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    budget = tmp_path / "graphs" / "budget-3p0"
    budget.mkdir(parents=True)
    (budget / "scores.json").write_text("[[1, 0.4]]", encoding="utf-8")
    (budget / "geo.json").write_text("[[1, 0.6]]", encoding="utf-8")
    (budget / "error.json").write_text("0.25", encoding="utf-8")

    rows, summary = collect(
        annotations, regions, episodes, budget.parent, split="val"
    )

    assert rows[0]["task_id"] == public_task_id("muno21-1-p0-add")
    assert rows[0]["apls_improvement"] == pytest.approx(0.4)
    assert rows[0]["pixel_f1_improvement"] == pytest.approx(0.6)
    assert rows[1] == {
        "task_id": "__aggregate__",
        "budget": 3.0,
        "no_change_error_rate": 0.25,
        "support_count": 1,
        "official_unit": "official_nochange_aggregate",
    }
    assert summary["error_rate_unit"] == "official_aggregate_per_model_seed"
    assert summary["test_assets_read"] is False


def test_collect_test_metrics_requires_frozen_authorization(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="run_frozen_paper_test"):
        collect(
            tmp_path / "annotations.json",
            tmp_path / "regions.json",
            tmp_path / "episodes.jsonl",
            tmp_path / "graphs",
            split="test",
        )
