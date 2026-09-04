import json

import pytest

from scripts.audit_muno21_graph_coverage import audit


def _annotations(path):
    path.write_text(
        json.dumps(
            [
                {"Cluster": {"Region": "chicago"}},
                {"Cluster": {"Region": "boston"}},
                {"Cluster": {"Region": "chicago"}},
            ]
        ),
        encoding="utf-8",
    )


def test_graph_coverage_requires_every_validation_annotation(tmp_path):
    annotations = tmp_path / "annotations.json"
    regions = tmp_path / "regions.json"
    graphs = tmp_path / "graphs" / "budget-1p5"
    graphs.mkdir(parents=True)
    _annotations(annotations)
    regions.write_text('["chicago"]', encoding="utf-8")
    (graphs / "0.graph").write_text("\n", encoding="utf-8")
    (graphs / "2.graph").write_text("\n", encoding="utf-8")

    result = audit(annotations, regions, graphs.parent)

    assert result["expected_annotation_count"] == 2
    assert result["coverage_complete"] is True


def test_graph_coverage_rejects_missing_annotation(tmp_path):
    annotations = tmp_path / "annotations.json"
    regions = tmp_path / "regions.json"
    graphs = tmp_path / "graphs" / "budget-1p5"
    graphs.mkdir(parents=True)
    _annotations(annotations)
    regions.write_text('["chicago"]', encoding="utf-8")
    (graphs / "0.graph").write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="coverage mismatch"):
        audit(annotations, regions, graphs.parent)
