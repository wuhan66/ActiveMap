from pathlib import Path

import json

import pytest

from scripts.watch_and_evaluate_active_catalog_candidates import (
    candidate_adapters,
    completed_process_result,
)


def test_candidate_adapters_discovers_retained_and_final(tmp_path: Path) -> None:
    for relative in (
        "retained_candidates/checkpoint-1000",
        "retained_candidates/checkpoint-500",
        "final",
    ):
        path = tmp_path / relative
        path.mkdir(parents=True)
        (path / "adapter_config.json").write_text("{}\n", encoding="utf-8")

    assert [label for label, _ in candidate_adapters(tmp_path)] == [
        "checkpoint-1000",
        "checkpoint-500",
        "final",
    ]


def test_candidate_adapters_ignores_incomplete_directories(tmp_path: Path) -> None:
    (tmp_path / "retained_candidates/checkpoint-500").mkdir(parents=True)
    assert candidate_adapters(tmp_path) == []


def test_completed_process_result_waits_for_success(tmp_path: Path) -> None:
    assert completed_process_result(tmp_path) is None
    path = tmp_path / "process_result.json"
    path.write_text("", encoding="utf-8")
    assert completed_process_result(tmp_path) is None
    path.write_text(json.dumps({"returncode": 0, "pid": 7}), encoding="utf-8")
    assert completed_process_result(tmp_path) == {"returncode": 0, "pid": 7}


def test_completed_process_result_rejects_failed_training(tmp_path: Path) -> None:
    (tmp_path / "process_result.json").write_text(
        json.dumps({"returncode": 1}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="training failed"):
        completed_process_result(tmp_path)
