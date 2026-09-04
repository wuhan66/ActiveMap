import json
from pathlib import Path

import pytest

from scripts.build_sn7_paper_evidence_manifest import build_manifest


def _write_jsonl(path: Path, count: int, *, test_read: bool = False) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": f"task-{index}",
                    "split": "val",
                    "test_assets_read": test_read,
                }
            )
            + "\n"
            for index in range(count)
        ),
        encoding="utf-8",
    )


def test_build_manifest_validates_and_fingerprints_bundle(tmp_path: Path) -> None:
    table = tmp_path / "table.json"
    table.write_text(
        json.dumps({"schema_version": "table-v1", "test_assets_read": False}),
        encoding="utf-8",
    )
    traces = tmp_path / "traces.jsonl"
    _write_jsonl(traces, 2)
    images = tmp_path / "images"
    images.mkdir()
    (images / "example.png").write_bytes(b"png")

    result = build_manifest(
        {"table": table},
        {"traces": traces},
        {"qualitative": images},
        expected_records=2,
        minimum_images=1,
    )

    assert result["status"] == "qc_approved"
    assert result["test_assets_read"] is False
    assert result["artifacts"]["traces"]["record_count"] == 2
    assert result["artifacts"]["qualitative"]["image_count"] == 1


def test_build_manifest_rejects_test_access(tmp_path: Path) -> None:
    traces = tmp_path / "traces.jsonl"
    _write_jsonl(traces, 1, test_read=True)

    with pytest.raises(ValueError, match="test_assets_read=false"):
        build_manifest(
            {},
            {"traces": traces},
            {},
            expected_records=1,
            minimum_images=0,
        )


def test_build_manifest_rejects_wrong_record_count(tmp_path: Path) -> None:
    traces = tmp_path / "traces.jsonl"
    _write_jsonl(traces, 1)

    with pytest.raises(ValueError, match="expected 2"):
        build_manifest(
            {},
            {"traces": traces},
            {},
            expected_records=2,
            minimum_images=0,
        )


def test_build_manifest_v2_hashes_submission_files(tmp_path: Path) -> None:
    artifact = tmp_path / "table.tex"
    artifact.write_text("\\begin{tabular}{lr}\\end{tabular}\n", encoding="utf-8")
    manifest = build_manifest(
        {},
        {},
        {},
        expected_records=1,
        minimum_images=1,
        file_paths={"latex_table": artifact},
    )
    row = manifest["artifacts"]["latex_table"]
    assert manifest["schema_version"] == "sn7-paper-evidence-manifest-v2"
    assert row["kind"] == "file"
    assert row["suffix"] == ".tex"
    assert len(row["sha256"]) == 64
