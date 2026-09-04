import json
from pathlib import Path

import pytest

from scripts.compare_sn7_changemamba_modalities import compare_modalities


def _aggregate(path: Path, mode: str, values: list[float]) -> Path:
    protocol = {
        "manifest_sha256": "manifest",
        "train_count": 10,
        "validation_count": 4,
        "test_assets_read": False,
        "input_contract": mode,
    }
    runs = []
    for seed, value in zip((1, 2, 3), values, strict=True):
        runs.append(
            {
                "seed": seed,
                "metrics": {
                    "committed_map_iou": value + 0.5,
                    "map_iou_delta": value,
                    "change_iou": value + 0.1,
                    "operation_accuracy": value + 0.2,
                    "keep_false_change_fraction": 0.1 - value,
                },
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": "sn7-changemamba-three-seed-aggregate-v1",
                "run_count": 3,
                "protocol": protocol,
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_compare_modalities_reports_paired_contributions(tmp_path: Path) -> None:
    paths = {
        "image_prior": _aggregate(tmp_path / "full.json", "full", [0.3, 0.4, 0.5]),
        "image_only": _aggregate(
            tmp_path / "image.json", "image", [0.2, 0.3, 0.4]
        ),
        "prior_only": _aggregate(
            tmp_path / "prior.json", "prior", [0.1, 0.2, 0.3]
        ),
    }
    result = compare_modalities(paths)
    assert result["test_assets_read"] is False
    assert result["joint_input_gain_over_best_single_modality"] == {
        "best_single_modality": "image_only",
        "map_iou_delta": pytest.approx(0.1),
    }
    assert (
        result["paired_comparisons"]["editable_prior_contribution"]["mean_delta"][
            "map_iou_delta"
        ]
        == pytest.approx(0.1)
    )
    assert (
        result["paired_comparisons"]["new_image_contribution"]["mean_delta"][
            "map_iou_delta"
        ]
        == pytest.approx(0.2)
    )


def test_compare_modalities_rejects_protocol_mismatch(tmp_path: Path) -> None:
    paths = {
        "image_prior": _aggregate(tmp_path / "full.json", "full", [0.3, 0.4, 0.5]),
        "image_only": _aggregate(
            tmp_path / "image.json", "image", [0.2, 0.3, 0.4]
        ),
        "prior_only": _aggregate(
            tmp_path / "prior.json", "prior", [0.1, 0.2, 0.3]
        ),
    }
    value = json.loads(paths["prior_only"].read_text(encoding="utf-8"))
    value["protocol"]["manifest_sha256"] = "different"
    paths["prior_only"].write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol mismatch"):
        compare_modalities(paths)
