import json
from pathlib import Path

import pytest

from scripts.summarize_muno21_graph_metric_self_check import summarize


def test_self_check_summary_is_explicitly_not_a_model_result(tmp_path: Path) -> None:
    (tmp_path / "self_check_manifest.json").write_text(
        json.dumps(
            {
                "regions": ["val-city"],
                "changed_graph_count": 2,
                "nochange_graph_count": 3,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "scores.json").write_text(
        json.dumps([[1, 1.0], [2, 0.95]]), encoding="utf-8"
    )
    (tmp_path / "geo.json").write_text(
        json.dumps([[1, 0.98], [2, 0.96]]), encoding="utf-8"
    )
    (tmp_path / "error.json").write_text("0.1", encoding="utf-8")

    report = summarize(tmp_path)

    assert report["passed"] is True
    assert report["model_result"] is False
    assert report["test_assets_read"] is False
    assert report["apls"]["mean"] == pytest.approx(0.975)
