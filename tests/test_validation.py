from pathlib import Path

from activemap.validation import validate_jsonl


def test_example_episode_matches_schema() -> None:
    project_root = Path(__file__).resolve().parents[1]
    valid_count, errors = validate_jsonl(
        project_root / "examples" / "episodes.jsonl",
        project_root / "schemas" / "episode.schema.json",
    )
    assert valid_count == 1
    assert errors == []


def test_invalid_jsonl_reports_line(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    jsonl = tmp_path / "invalid.jsonl"
    jsonl.write_text('{"episode_id": "missing-fields"}\n', encoding="utf-8")
    valid_count, errors = validate_jsonl(
        jsonl,
        project_root / "schemas" / "episode.schema.json",
    )
    assert valid_count == 0
    assert errors
    assert errors[0].startswith("line 1")
