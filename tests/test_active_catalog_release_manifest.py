from scripts.active_catalog_release_manifest import build_manifest, verify_manifest


def test_release_manifest_detects_code_drift(tmp_path):
    for root in ("src", "scripts", "configs", "tests", "docs"):
        (tmp_path / root).mkdir()
    source = tmp_path / "src" / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    manifest = build_manifest(
        tmp_path,
        declared_artifacts={"states": "a" * 64},
    )
    assert verify_manifest(tmp_path, manifest)["passed"] is True
    assert manifest["declared_artifact_sha256"]["states"] == "a" * 64

    source.write_text("VALUE = 2\n", encoding="utf-8")
    result = verify_manifest(tmp_path, manifest)
    assert result["passed"] is False
    assert result["hash_mismatches"] == ["src/module.py"]


def test_release_manifest_detects_missing_file(tmp_path):
    for root in ("src", "scripts", "configs", "tests", "docs"):
        (tmp_path / root).mkdir()
    source = tmp_path / "scripts" / "run.sh"
    source.write_text("#!/bin/sh\n", encoding="utf-8")
    manifest = build_manifest(tmp_path)
    source.unlink()
    result = verify_manifest(tmp_path, manifest)
    assert result["passed"] is False
    assert result["missing"] == ["scripts/run.sh"]
