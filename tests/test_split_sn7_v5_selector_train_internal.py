from __future__ import annotations

from pathlib import Path

from activemap.features import EVIDENCE_DIM, HYPOTHESIS_DIM, STATE_DIM
from activemap.selector_records import SelectorSample
from scripts.split_sn7_v5_selector_train_internal import create_internal_split


def _sample(source: str, index: int) -> SelectorSample:
    return SelectorSample.model_validate(
        {
            "sample_id": f"{source}__b1p5__s{index}",
            "split": "train",
            "edit_type": "ADD",
            "hypothesis_features": [0.0] * HYPOTHESIS_DIM,
            "state_features": [0.0] * STATE_DIM,
            "evidence_ids": ["candidate"],
            "evidence_features": [[0.0] * EVIDENCE_DIM],
            "evidence_costs": [0.2],
            "false_edit_risks": [0.1],
            "oracle_utilities": [0.2],
            "metadata": {
                "source_episode": source,
                "aoi_id": source.split("-")[0],
                "budget": 1.5,
                "oracle_step": index,
                "gt_edit": "ADD",
                "selected_evidence_ids": ["anchor"],
            },
        }
    )


def test_internal_split_keeps_formal_validation_out_of_selector_training(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    samples = [_sample(f"aoi-{index}", step) for index in range(6) for step in (0, 1)]
    source.write_text("\n".join(sample.model_dump_json() for sample in samples) + "\n")

    result = create_internal_split(source, tmp_path / "internal", tune_fraction=0.2, seed=7)

    assert result["formal_validation_assets_read"] is False
    assert result["test_assets_read"] is False
    assert result["split"]["source_episode_overlap"] == 0
    assert result["outputs"]["fit"]["state_count"] + result["outputs"]["tune"]["state_count"] == len(samples)
    tune = Path(result["outputs"]["tune"]["path"]).read_text().splitlines()
    assert tune and all('"split":"val"' in line for line in tune)
