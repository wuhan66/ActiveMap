import json
from pathlib import Path

import pytest

from scripts.audit_sn7_controller_writeback_job import audit


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _job(tmp_path: Path) -> Path:
    root = tmp_path / "job"
    closed_rows = []
    writeback_rows = []
    for index, budget in enumerate((1.5, 3.0)):
        sample = f"sample-{index}"
        closed_rows.append(
            {
                "sample_id": sample,
                "budget": budget,
                "split": "val",
                "policy": "edit_utility",
                "test_assets_read": False,
            }
        )
        writeback_rows.append(
            {
                "source_example_id": sample,
                "budget": budget,
                "split": "val",
                "mask_artifact": f"mask-{index}.npz",
                "test_assets_read": False,
            }
        )
    _write_json(
        root / "closed_loop" / "summary.json",
        {
            "schema_version": "active-catalog-closed-loop-baselines-v1",
            "split": "val",
            "sample_count": 2,
            "protocol": {"test_assets_read": False},
        },
    )
    (root / "closed_loop" / "edit_utility.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in closed_rows),
        encoding="utf-8",
    )
    _write_json(
        root / "writeback" / "summary.json",
        {
            "sample_count": 2,
            "protocol": {
                "name": "protocol",
                "test_assets_read": False,
                "prior_input_corruption": {
                    "translation_pixels": 4,
                    "morphology": "none",
                    "morphology_pixels": 0,
                    "corruption_seed": 17,
                    "scope": "model_input_only",
                },
            },
            "budgets": [{"sample_count": 1}, {"sample_count": 1}],
        },
    )
    (root / "writeback" / "writeback.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in writeback_rows),
        encoding="utf-8",
    )
    return root


def test_audit_accepts_complete_matched_job(tmp_path: Path) -> None:
    report = audit(
        _job(tmp_path),
        expected_rows=2,
        protocol_name="protocol",
        translation_pixels=4,
        morphology="none",
        morphology_pixels=0,
        corruption_seed=17,
    )

    assert report["passed"] is True
    assert report["identity_count"] == 2


def test_audit_rejects_identity_mismatch(tmp_path: Path) -> None:
    root = _job(tmp_path)
    rows = [
        json.loads(line)
        for line in (root / "writeback" / "writeback.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    rows[1]["source_example_id"] = "other"
    (root / "writeback" / "writeback.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="identities differ"):
        audit(
            root,
            expected_rows=2,
            protocol_name="protocol",
            translation_pixels=4,
            morphology="none",
            morphology_pixels=0,
            corruption_seed=17,
        )
