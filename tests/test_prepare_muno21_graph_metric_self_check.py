import json
from pathlib import Path

from scripts.prepare_muno21_graph_metric_self_check import prepare_self_check


def test_prepare_self_check_uses_all_annotations_from_selected_regions(
    tmp_path: Path,
) -> None:
    annotations = [
        {"Cluster": {"Region": "chicago", "Tile": [1, 2]}, "Tags": []},
        {"Cluster": {"Region": "chicago", "Tile": [2, 3]}, "Tags": ["nochange"]},
        {"Cluster": {"Region": "test-city", "Tile": [4, 5]}, "Tags": []},
    ]
    annotations_path = tmp_path / "annotations.json"
    regions_path = tmp_path / "regions.json"
    graph_dir = tmp_path / "graphs"
    output_dir = tmp_path / "out"
    graph_dir.mkdir()
    annotations_path.write_text(json.dumps(annotations), encoding="utf-8")
    regions_path.write_text(json.dumps(["chicago"]), encoding="utf-8")
    source = graph_dir / "chicago_1_2_2020-07-01.graph"
    source.write_text("graph", encoding="utf-8")
    nochange_source = graph_dir / "chicago_2_3_2020-07-01.graph"
    nochange_source.write_text("nochange graph", encoding="utf-8")

    summary = prepare_self_check(
        annotations_path, regions_path, graph_dir, output_dir
    )

    assert summary["graph_count"] == 2
    assert summary["changed_graph_count"] == 1
    assert summary["nochange_graph_count"] == 1
    assert summary["model_result"] is False
    assert summary["test_assets_read"] is False
    assert (output_dir / "0.graph").read_text(encoding="utf-8") == "graph"
    assert (output_dir / "1.graph").read_text(encoding="utf-8") == "nochange graph"
    assert not (output_dir / "2.graph").exists()
