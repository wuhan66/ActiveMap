import copy
import json
from pathlib import Path

import pytest

from scripts.audit_sn7_frozen_anchor_parity import audit


def _row() -> dict:
    return {
        "edit_type": "KEEP",
        "hypothesis_features": [float(index) for index in range(16)],
        "state_features": [float(index) for index in range(8)],
        "evidence_features": [[float(index) for index in range(13)]],
        "oracle_utilities": [0.1],
        "metadata": {
            "source_episode": "episode",
            "budget": 1.5,
            "executable_outcomes": {"evidence": {"quality": 0.1}},
            "evidence_predictions": {"evidence": {"confidence": 0.5}},
        },
    }


def _write(path: Path, row: dict) -> None:
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_parity_audit_accepts_only_whitelisted_changes(tmp_path: Path) -> None:
    frozen = _row()
    anchored = copy.deepcopy(frozen)
    anchored["edit_type"] = "DELETE"
    anchored["hypothesis_features"][0] = 0.9
    anchored["state_features"][2] = 0.8
    anchored["evidence_features"][0][11] = 0.7
    anchored["oracle_utilities"] = [0.6]
    anchored["metadata"]["evidence_predictions"] = {"evidence": {"confidence": 0.9}}
    anchored["metadata"]["prior_input_corruption"] = {
        "translation_pixels": 0,
        "morphology": "erode",
        "morphology_pixels": 4,
        "corruption_seed": 0,
        "scope": "model_input_only",
    }
    frozen_path = tmp_path / "frozen.jsonl"
    anchored_path = tmp_path / "anchored.jsonl"
    _write(frozen_path, frozen)
    _write(anchored_path, anchored)

    report = audit(frozen_path, anchored_path)

    assert report["passed"] is True
    assert report["changed_state_count"] == 1
    assert report["prior_input_corruption"]["morphology"] == "erode"


def test_parity_audit_accepts_per_row_translation_realizations(
    tmp_path: Path,
) -> None:
    frozen_rows = []
    anchored_rows = []
    for index, shift in enumerate((-4, 3)):
        frozen = _row()
        frozen["metadata"]["source_episode"] = f"episode-{index}"
        anchored = copy.deepcopy(frozen)
        anchored["state_features"][2] += 0.1
        anchored["metadata"]["prior_input_corruption"] = {
            "translation_pixels": 4,
            "shift_x": shift,
            "shift_y": -shift,
            "morphology": "none",
            "morphology_pixels": 0,
            "corruption_seed": 17,
            "scope": "model_input_only",
        }
        frozen_rows.append(frozen)
        anchored_rows.append(anchored)
    frozen_path = tmp_path / "frozen.jsonl"
    anchored_path = tmp_path / "anchored.jsonl"
    _write_rows(frozen_path, frozen_rows)
    _write_rows(anchored_path, anchored_rows)

    report = audit(frozen_path, anchored_path)

    assert report["corruption_realization_count"] == 2
    assert report["prior_input_corruption"]["translation_pixels"] == 4


def test_parity_audit_rejects_out_of_radius_translation(
    tmp_path: Path,
) -> None:
    frozen = _row()
    anchored = copy.deepcopy(frozen)
    anchored["state_features"][2] += 0.1
    anchored["metadata"]["prior_input_corruption"] = {
        "translation_pixels": 4,
        "shift_x": 5,
        "shift_y": 0,
        "corruption_seed": 17,
        "scope": "model_input_only",
    }
    frozen_path = tmp_path / "frozen.jsonl"
    anchored_path = tmp_path / "anchored.jsonl"
    _write(frozen_path, frozen)
    _write(anchored_path, anchored)

    with pytest.raises(ValueError, match="invalid shift_x"):
        audit(frozen_path, anchored_path)


def test_parity_audit_rejects_non_whitelisted_change(tmp_path: Path) -> None:
    frozen = _row()
    anchored = copy.deepcopy(frozen)
    anchored["metadata"]["budget"] = 3.0
    anchored["metadata"]["prior_input_corruption"] = {
        "translation_pixels": 4,
        "corruption_seed": 17,
    }
    frozen_path = tmp_path / "frozen.jsonl"
    anchored_path = tmp_path / "anchored.jsonl"
    _write(frozen_path, frozen)
    _write(anchored_path, anchored)

    with pytest.raises(ValueError):
        audit(frozen_path, anchored_path)
