import json

from PIL import Image

from scripts.render_chronological_map_maintenance import render


def polygon(x0, x1):
    return {
        "type": "Polygon",
        "coordinates": [[[x0, 0], [x1, 0], [x1, 1], [x0, 1], [x0, 0]]],
    }


def test_renderer_exports_separate_unlabeled_masks(tmp_path):
    trace = tmp_path / "trace.jsonl"
    row = {
        "chain_id": "chain-0",
        "step": 0,
        "task_id": "task",
        "aoi_id": "aoi",
        "object_id": "object",
        "timestamp": "2018_02",
        "target_edit": "ADD",
        "independent_iou": 1.0,
        "carry_iou": 0.8,
        "risk_gated_iou": 0.9,
        "risk_gate_intervened": True,
        "geometries": {
            "target": polygon(0, 1),
            "independent": polygon(0, 1),
            "carry": polygon(0, 0.8),
            "risk_gated": polygon(0, 0.9),
        },
    }
    trace.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output = tmp_path / "visuals"
    summary = render(trace, output, size=64, maximum_chains=1)

    step = next(path for path in output.rglob("00_2018_02"))
    assert summary["text_embedded_in_png"] is False
    assert {path.name for path in step.glob("*.png")} == {
        "target.png",
        "independent.png",
        "carry.png",
        "risk_gated.png",
    }
    assert Image.open(step / "target.png").size == (64, 64)
