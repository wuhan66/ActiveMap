from __future__ import annotations

import json
from pathlib import Path

from activemap.features import EVIDENCE_DIM, HYPOTHESIS_DIM, STATE_DIM
from activemap.selector_records import SelectorSample
from scripts.select_sn7_v5_qualitative_cases import select_cases


def _sample(operation: str, source: str) -> SelectorSample:
    return SelectorSample.model_validate(
        {
            "sample_id": f"{source}__b1p5__s0",
            "split": "val",
            "edit_type": operation,
            "hypothesis_features": [0.0] * HYPOTHESIS_DIM,
            "state_features": [0.0] * STATE_DIM,
            "evidence_ids": ["candidate"],
            "evidence_features": [[0.0] * EVIDENCE_DIM],
            "evidence_costs": [0.2],
            "false_edit_risks": [0.1],
            "oracle_utilities": [0.2],
            "metadata": {
                "source_episode": source,
                "aoi_id": "aoi-a",
                "budget": 1.5,
                "oracle_step": 0,
                "gt_edit": operation,
                "selected_evidence_ids": ["anchor"],
            },
        }
    )


def test_qualitative_selection_is_deterministic_and_outcome_free(tmp_path: Path) -> None:
    states = tmp_path / "states.jsonl"
    samples = [
        _sample(operation, f"{operation.lower()}-{index}")
        for operation in ("ADD", "DELETE", "RESHAPE")
        for index in range(4)
    ]
    states.write_text("\n".join(sample.model_dump_json() for sample in samples) + "\n")

    first = select_cases(states, per_operation=2, seed=5)
    second = select_cases(states, per_operation=2, seed=5)

    assert first == second
    assert len(first["cases"]) == 6
    assert first["controller_or_writeback_outputs_read"] is False
    assert first["test_assets_read"] is False
    assert all("prediction" not in row for row in first["cases"])
    json.dumps(first)
