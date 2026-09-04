import json
from pathlib import Path

import pytest

from scripts.bootstrap_sn7_changemamba_modalities import paired_bootstrap


def _audit(path: Path, offset: float) -> Path:
    path.mkdir()
    (path / "summary.json").write_text(
        json.dumps({"test_assets_read": False}), encoding="utf-8"
    )
    rows = []
    for aoi_index in range(2):
        for sample_index, target in enumerate(("KEEP", "ADD")):
            delta = offset + 0.01 * aoi_index
            rows.append(
                {
                    "sample_id": f"a{aoi_index}-s{sample_index}",
                    "aoi_id": f"a{aoi_index}",
                    "target_edit": target,
                    "committed_map_iou": 0.5 + delta,
                    "map_iou_delta": delta,
                    "change_iou": 0.4 + delta,
                    "operation_correct": 1.0,
                    "false_edit": float(target == "KEEP" and offset < 0.15),
                    "missed_edit": 0.0,
                    "wrong_edit": 0.0,
                }
            )
    (path / "per_sample.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def test_paired_bootstrap_uses_shared_aoi_draws(tmp_path: Path) -> None:
    audits = {}
    for mode, offset in (
        ("image_prior", 0.3),
        ("image_only", 0.2),
        ("prior_only", 0.1),
    ):
        audits[mode] = [
            _audit(tmp_path / f"{mode}-{seed}", offset) for seed in range(3)
        ]
    result = paired_bootstrap(audits, draws=100, seed=7)
    prior_gain = result["results"]["full_minus_image_only"]["map_iou_delta"]
    image_gain = result["results"]["full_minus_prior_only"]["map_iou_delta"]
    assert prior_gain["observed"] == pytest.approx(0.1)
    assert prior_gain["ci_low"] == pytest.approx(0.1)
    assert image_gain["observed"] == pytest.approx(0.2)
    assert result["sample_count"] == 4
    assert result["aoi_count"] == 2


def test_paired_bootstrap_rejects_identity_mismatch(tmp_path: Path) -> None:
    audits = {
        mode: [_audit(tmp_path / f"{mode}-{seed}", 0.1) for seed in range(3)]
        for mode in ("image_prior", "image_only", "prior_only")
    }
    rows = (audits["prior_only"][2] / "per_sample.jsonl").read_text(
        encoding="utf-8"
    )
    (audits["prior_only"][2] / "per_sample.jsonl").write_text(
        rows.replace('"a0-s0"', '"different"', 1), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="sample identity mismatch"):
        paired_bootstrap(audits, draws=100, seed=7)
