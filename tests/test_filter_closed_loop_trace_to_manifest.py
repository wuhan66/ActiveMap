import json

import pytest

from scripts.filter_closed_loop_trace_to_manifest import filter_trace


def _row(episode: str, budget: float, *, aoi: str = "a") -> dict:
    return {
        "source_episode": episode,
        "budget": budget,
        "aoi_id": aoi,
        "split": "val",
        "test_assets_read": False,
        "acquisitions": 0,
    }


def test_filter_requires_exact_manifest_support(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [_row("one", 1.5), _row("two", 3.0), _row("extra", 4.5)]
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "split": "val",
                "test_assets_read": False,
                "records": [
                    {"source_episode": "one", "budget": 1.5},
                    {"source_episode": "two", "budget": 3.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    rows, receipt = filter_trace(trace, manifest)
    assert [row["source_episode"] for row in rows] == ["one", "two"]
    assert receipt["record_count"] == 2


def test_filter_rejects_missing_manifest_record(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps(_row("one", 1.5)) + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "split": "val",
                "test_assets_read": False,
                "records": [
                    {"source_episode": "one", "budget": 1.5},
                    {"source_episode": "two", "budget": 3.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not exactly cover"):
        filter_trace(trace, manifest)
