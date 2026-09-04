import copy
import json
from pathlib import Path

import pytest

from scripts.anchor_sn7_corrupted_states_to_frozen_step0 import anchor_file, anchor_row


def _row() -> dict:
    return {
        "sample_id": "episode__b1p5__s0",
        "split": "val",
        "edit_type": "KEEP",
        "hypothesis_features": [float(index) for index in range(16)],
        "state_features": [float(index) for index in range(8)],
        "evidence_ids": ["evidence"],
        "evidence_features": [[float(index) for index in range(13)]],
        "evidence_costs": [1.0],
        "false_edit_risks": [0.0],
        "oracle_utilities": [0.1],
        "metadata": {
            "source_episode": "episode",
            "budget": 1.5,
            "oracle_step": 0,
            "aoi_id": "aoi",
            "gt_edit": "ADD",
            "initial_evidence_id": "evidence",
            "executable_outcomes": {"evidence": {"quality_gain": 0.1}},
            "evidence_predictions": {"evidence": {"confidence": 0.6}},
        },
    }


def test_anchor_transplants_only_perception_dependent_fields() -> None:
    frozen = _row()
    regenerated = copy.deepcopy(frozen)
    regenerated["edit_type"] = "ADD"
    regenerated["hypothesis_features"] = [100.0 + index for index in range(16)]
    regenerated["state_features"] = [200.0 + index for index in range(8)]
    regenerated["evidence_features"][0][11] = 311.0
    regenerated["false_edit_risks"] = [0.9]
    regenerated["oracle_utilities"] = [0.8]
    regenerated["metadata"]["executable_outcomes"] = {
        "evidence": {"quality_gain": 0.8}
    }
    regenerated["metadata"]["evidence_predictions"] = {
        "evidence": {"confidence": 0.9}
    }
    regenerated["metadata"]["prior_input_corruption"] = {
        "translation_pixels": 8,
        "corruption_seed": 17,
    }

    anchored = anchor_row(frozen, regenerated)

    assert anchored["edit_type"] == "ADD"
    assert anchored["false_edit_risks"] == [0.0]
    assert anchored["hypothesis_features"][14] == 14.0
    assert anchored["hypothesis_features"][13] == 113.0
    assert anchored["state_features"][7] == 207.0
    assert anchored["evidence_features"][0][11] == 311.0
    assert anchored["oracle_utilities"] == [0.8]
    assert "prior_input_corruption" not in frozen["metadata"]


def test_anchor_rejects_changed_frozen_identity() -> None:
    frozen = _row()
    regenerated = copy.deepcopy(frozen)
    regenerated["metadata"]["budget"] = 3.0

    with pytest.raises(ValueError, match="identities differ"):
        anchor_row(frozen, regenerated)


def test_anchor_summary_retains_morphology_protocol(tmp_path: Path) -> None:
    frozen = _row()
    regenerated = copy.deepcopy(frozen)
    regenerated["metadata"]["prior_input_corruption"] = {
        "translation_pixels": 0,
        "morphology": "erode",
        "morphology_pixels": 4,
        "corruption_seed": 0,
        "scope": "model_input_only",
    }
    frozen_path = tmp_path / "frozen.jsonl"
    regenerated_path = tmp_path / "regenerated.jsonl"
    output_path = tmp_path / "anchored.jsonl"
    frozen_path.write_text(json.dumps(frozen) + "\n", encoding="utf-8")
    regenerated_path.write_text(json.dumps(regenerated) + "\n", encoding="utf-8")

    summary = anchor_file(frozen_path, regenerated_path, output_path)

    assert summary["schema_version"].endswith("-v2")
    assert summary["morphology"] == "erode"
    assert summary["morphology_pixels"] == 4
    assert summary["prior_input_corruption"]["translation_pixels"] == 0
