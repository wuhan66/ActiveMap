import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_updater_conditioned_selector_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_updater_conditioned", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_group_split_is_deterministic_and_episode_disjoint() -> None:
    rows = [
        {"sample_id": f"a-{index}", "metadata": {"source_episode": "episode-a"}}
        for index in range(3)
    ] + [
        {"sample_id": f"b-{index}", "metadata": {"source_episode": "episode-b"}}
        for index in range(3)
    ]

    assignments = {
        row["metadata"]["source_episode"]: MODULE._group_is_dev(
            row["metadata"]["source_episode"], 20260809, 0.2
        )
        for row in rows
    }
    assert len(assignments) == 2
    assert all(
        MODULE._group_is_dev(row["metadata"]["source_episode"], 20260809, 0.2)
        == assignments[row["metadata"]["source_episode"]]
        for row in rows
    )


def test_loader_rejects_nontrain_or_missing_source_episode(tmp_path: Path) -> None:
    path = tmp_path / "states.jsonl"
    path.write_text(json.dumps({"sample_id": "x", "split": "val", "metadata": {}}) + "\n")
    try:
        MODULE._load(path)
    except ValueError as error:
        assert "train-only" in str(error)
    else:
        raise AssertionError("expected train-only validation")
