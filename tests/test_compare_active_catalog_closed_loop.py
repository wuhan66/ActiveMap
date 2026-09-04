import json
import sys

import pytest

from scripts import compare_active_catalog_closed_loop
from scripts.compare_active_catalog_closed_loop import paired_aoi_bootstrap


def _row(aoi, correct, utility, false_edit=False, cost=1.0):
    return {
        "aoi_id": aoi,
        "terminal_correct": correct,
        "false_edit": false_edit,
        "missed_edit": False,
        "acquisitions": int(cost > 0),
        "steps": 2,
        "spent_cost": cost,
        "quality_gain": utility + cost,
        "quality_cost_utility": utility,
        "episode_utility_v2_proxy_balanced": utility,
        "episode_utility_v2_proxy_safety": utility - 0.1 * float(false_edit),
        "episode_utility_v2_proxy_cost_aware": utility - 0.1 * cost,
        "model_action_count": 0,
        "valid_action_count": 0,
        "fallback_count": 0,
    }


def test_paired_aoi_bootstrap_reports_directional_gain():
    keys = [("e1", 1.0), ("e2", 1.0)]
    reference = {
        keys[0]: _row("a", False, 0.0, cost=0.0),
        keys[1]: _row("b", False, 0.0, cost=0.0),
    }
    candidate = {
        keys[0]: _row("a", True, 0.2),
        keys[1]: _row("b", True, 0.3),
    }
    result = paired_aoi_bootstrap(candidate, reference, repetitions=50, seed=7)
    assert result["intervals"]["terminal_accuracy"]["observed_delta"] == 1.0
    assert result["intervals"]["mean_quality_cost_utility"]["ci95_low"] > 0
    assert (
        result["intervals"]["mean_episode_utility_v2_proxy_balanced"]["ci95_low"]
        > 0
    )
    assert result["non_dominated_observed"] is True


def test_paired_aoi_bootstrap_rejects_unpaired_inputs():
    with pytest.raises(ValueError, match="identical"):
        paired_aoi_bootstrap(
            {("e1", 1.0): _row("a", True, 0.1)},
            {("e2", 1.0): _row("a", True, 0.1)},
            repetitions=20,
            seed=7,
        )


def _trace_row(episode, aoi, utility):
    row = _row(aoi, True, utility)
    row.update(
        {
            "split": "val",
            "test_assets_read": False,
            "source_episode": episode,
            "budget": 3.0,
        }
    )
    return row


def test_manifest_comparison_requires_exact_validation_support(tmp_path, monkeypatch):
    rows = [_trace_row("episode-a", "a", 0.1), _trace_row("episode-b", "b", 0.2)]
    candidate = tmp_path / "candidate.jsonl"
    reference = tmp_path / "reference.jsonl"
    serialized = "".join(json.dumps(row) + "\n" for row in rows)
    candidate.write_text(serialized, encoding="utf-8")
    reference.write_text(serialized, encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "split": "val",
                "test_assets_read": False,
                "selection_contract": {"support": "fixture"},
                "records": [
                    {"source_episode": "episode-a", "budget": 3.0},
                    {"source_episode": "episode-b", "budget": 3.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_active_catalog_closed_loop.py",
            str(output),
            "--records",
            f"candidate={candidate}",
            "--records",
            f"reference={reference}",
            "--candidate",
            "candidate",
            "--manifest",
            str(manifest),
            "--repetitions",
            "10",
        ],
    )

    compare_active_catalog_closed_loop.main()

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["comparison_manifest"]["record_count"] == 2
    assert result["traces"]["candidate"]["sha256"]

    manifest.write_text(
        json.dumps(
            {
                "split": "val",
                "test_assets_read": False,
                "records": [{"source_episode": "episode-a", "budget": 3.0}],
            }
        ),
        encoding="utf-8",
    )
    output.unlink()
    with pytest.raises(ValueError, match="does not exactly cover"):
        compare_active_catalog_closed_loop.main()
