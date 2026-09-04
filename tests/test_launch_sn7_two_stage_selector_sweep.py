from pathlib import Path

import pytest
import yaml

from scripts.launch_sn7_two_stage_selector_sweep import jobs, prepare_job


def test_jobs_assigns_two_stage_variants() -> None:
    assert jobs(5, [0, 2, 4, 5]) == [
        {"variant": "generic_gate_rank", "seed": 5, "gpu": 0},
        {"variant": "generic_utility", "seed": 5, "gpu": 2},
        {"variant": "edit_gate_rank", "seed": 5, "gpu": 4},
        {"variant": "edit_utility", "seed": 5, "gpu": 5},
    ]


def test_jobs_requires_four_gpus() -> None:
    with pytest.raises(ValueError, match="four distinct GPUs"):
        jobs(5, [0, 1])


def test_prepare_job_overrides_samples(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    samples = tmp_path / "states.jsonl"
    row = {"variant": "edit_utility", "seed": 5, "gpu": 0}
    prepared = prepare_job(repo, tmp_path / "runs", row, samples)
    payload = yaml.safe_load(Path(prepared["config"]).read_text(encoding="utf-8"))
    assert payload["data"]["samples"] == str(samples.resolve())
    assert payload["model"]["candidate_value_head"] is True
    assert payload["model"]["candidate_decision_mode"] == "value"
    assert payload["model"]["terminal_gate_mode"] == "value"
    assert payload["training"]["candidate_value_sign_weight"] == 1.0
