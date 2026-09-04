import json
from pathlib import Path

from scripts.audit_sn7_paper_evidence_manifest import audit
from scripts.build_sn7_paper_evidence_manifest import build_manifest


def test_audit_detects_file_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "table.tex"
    artifact.write_text("frozen\n", encoding="utf-8")
    manifest = build_manifest(
        {},
        {},
        {},
        expected_records=1,
        minimum_images=0,
        file_paths={"table": artifact},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert audit(manifest_path)["ready"] is True

    artifact.write_text("changed\n", encoding="utf-8")
    result = audit(manifest_path)
    assert result["ready"] is False
    assert result["artifacts"][0]["hash_matches"] is False
