from __future__ import annotations

import importlib.util
from pathlib import Path


def load_artifact_builder():
    script = Path(__file__).resolve().parents[1] / "tools" / "create_anonymous_review_artifact.py"
    spec = importlib.util.spec_from_file_location("anonymous_review_artifact", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_anonymity_pattern_catches_generic_provenance_without_baked_in_host() -> None:
    builder = load_artifact_builder()

    assert builder.BANNED_TEXT.search(r"C:\Users\author\workspace")
    assert builder.BANNED_TEXT.search("/home/researcher/project")
    assert builder.BANNED_TEXT.search("compute-server")
    assert builder.BANNED_TEXT.search("host.example.icu")
    assert builder.BANNED_TEXT.search("author@example.edu")
    assert not builder.BANNED_TEXT.search("harmonic regularization")
    assert not builder.BANNED_TEXT.search("--max-connection-per-server=16")


def test_curated_reviewer_package_includes_qc_approval_dependency() -> None:
    builder = load_artifact_builder()

    assert "scripts/approve_dataset_qc.py" in builder.ROOT_FILES
    assert "examples/episodes.jsonl" in builder.ROOT_FILES
