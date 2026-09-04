#!/usr/bin/env python3
"""Build a small, anonymous, fail-closed ActiveMap reviewer-code package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml", ".json"}
# Keep this pattern generic: a reviewer-facing source bundle must not encode a
# particular workstation, account, or compute host merely to detect it later.
BANNED_TEXT = re.compile(
    r"""(?ix)
    (?:
        \b[a-z]:[\\/]
        | /(?:home|users)/
        | \b(?![a-z0-9-]*per-server\b)[a-z0-9][a-z0-9-]*-server\b
        | \b[a-z0-9-]+\.(?:internal|local|lan|icu)\b
        | \b[a-z0-9._%+-]+@[a-z0-9.-]+\.(?:com|org|edu|net|cn|io|icu)\b
    )
    """
)
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}

ROOT_FILES = (
    "README_REVIEW.md",
    "REPRODUCIBILITY.md",
    "MODEL_DATA_AVAILABILITY.md",
    "RELEASE_NOTICE.md",
    "environment.yml",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-ml.txt",
    "requirements-agent.txt",
    "requirements-optional.txt",
    "requirements-serving.txt",
    "Makefile",
    "tools/create_anonymous_review_artifact.py",
    "examples/episodes.jsonl",
    "configs/smoke/selector.yaml",
    "configs/smoke/updater.yaml",
    "configs/selector/base.yaml",
    "configs/updater/base.yaml",
    "scripts/bootstrap_server.sh",
    "scripts/run_smoke.sh",
    "scripts/approve_dataset_qc.py",
    "scripts/download_sn7.sh",
    "scripts/prepare_sn7.sh",
    "scripts/download_muno21.sh",
    "scripts/download_inria.sh",
    "docs/dataset_protocol.md",
    "docs/metrics.md",
    "docs/DATA.md",
    "docs/SN7_V1_DATA_CARD.md",
)
TEST_FILES = (
    "tests/test_cli.py",
    "tests/test_counterfactual_builder.py",
    "tests/test_episode_utility.py",
    "tests/test_geometry_edits.py",
    "tests/test_safe_commit.py",
    "tests/test_rollout.py",
    "tests/test_vector_map.py",
    "tests/test_schema_export.py",
    "tests/test_qc_split_safety.py",
    "tests/test_models.py",
    "tests/test_nn_selector.py",
    "tests/test_nn_updater.py",
    "tests/test_sn7.py",
    "tests/test_splits.py",
)
TREE_ROOTS = ("src", "schemas")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_packagable(path: Path) -> bool:
    return path.is_file() and not any(part in EXCLUDED_PARTS for part in path.parts)


def collect_files() -> list[Path]:
    selected: set[Path] = set()
    for relative in (*ROOT_FILES, *TEST_FILES):
        path = ROOT / relative
        if not is_packagable(path):
            raise FileNotFoundError(f"Missing curated reviewer-artifact input: {relative}")
        selected.add(path)
    for relative in TREE_ROOTS:
        directory = ROOT / relative
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing reviewer-artifact tree: {relative}")
        selected.update(path for path in directory.rglob("*") if is_packagable(path))
    return sorted(selected, key=lambda path: path.relative_to(ROOT).as_posix())


def scan_anonymity(files: list[Path]) -> None:
    findings: list[str] = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if BANNED_TEXT.search(relative):
            findings.append(f"path:{relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if BANNED_TEXT.search(line):
                findings.append(f"{relative}:{line_number}: {line.strip()}")
    if findings:
        message = "Reviewer artifact contains identifying provenance:\n" + "\n".join(findings)
        raise RuntimeError(message)


def verify_python_sources(files: list[Path]) -> None:
    for path in files:
        if path.suffix == ".py":
            compile(path.read_text(encoding="utf-8"), path.relative_to(ROOT).as_posix(), "exec")


def create_package(output: Path, archive: bool) -> Path | None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    files = collect_files()
    scan_anonymity(files)
    verify_python_sources(files)
    output.mkdir(parents=True)
    records: list[dict[str, str]] = []
    for source in files:
        relative = source.relative_to(ROOT)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append({"path": relative.as_posix(), "sha256": sha256(target)})
    manifest = {
        "schema_version": "activemap/anonymous-review-artifact-v1",
        "scope": "minimal executable code and documentation for anonymous peer review",
        "smoke_command": "bash scripts/run_smoke.sh",
        "files": records,
        "anonymity_scan": "passed",
        "source_compile_check": "passed",
    }
    manifest_path = output / "ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if archive:
        return Path(shutil.make_archive(str(output), "zip", root_dir=output))
    return None


def verify_archive(archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        names = {item.filename for item in archive.infolist() if not item.is_dir()}
        if "ARTIFACT_MANIFEST.json" not in names:
            raise RuntimeError("Archive is missing ARTIFACT_MANIFEST.json")
        manifest = json.loads(archive.read("ARTIFACT_MANIFEST.json").decode("utf-8"))
        expected = {"ARTIFACT_MANIFEST.json"}
        for record in manifest.get("files", []):
            relative = record.get("path")
            digest = record.get("sha256")
            if not isinstance(relative, str) or not isinstance(digest, str):
                raise RuntimeError("Archive manifest has an invalid file record")
            expected.add(relative)
            if relative not in names:
                raise RuntimeError(f"Archive is missing {relative}")
            if hashlib.sha256(archive.read(relative)).hexdigest() != digest:
                raise RuntimeError(f"Archive hash mismatch for {relative}")
        unexpected = sorted(names - expected)
        if unexpected:
            raise RuntimeError("Archive contains non-allowlisted files: " + ", ".join(unexpected))
        for name in sorted(names):
            suffix = Path(name).suffix.lower()
            if suffix not in TEXT_SUFFIXES:
                continue
            text = archive.read(name).decode("utf-8")
            if BANNED_TEXT.search(text):
                raise RuntimeError(f"Archive anonymity scan failed: {name}")
    print(f"Verified anonymous reviewer artifact: {archive_path.resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New staging directory")
    parser.add_argument("--archive", action="store_true", help="Create a sibling ZIP")
    parser.add_argument("--verify-archive", type=Path, help="Verify an existing ZIP")
    args = parser.parse_args()
    if args.verify_archive is not None:
        if args.output is not None or args.archive:
            parser.error("--verify-archive cannot be combined with package creation")
        verify_archive(args.verify_archive.resolve())
        return 0
    if args.output is None:
        parser.error("--output is required when creating a package")
    archive = create_package(args.output.resolve(), args.archive)
    print(f"Created anonymous reviewer artifact: {args.output.resolve()}")
    if archive is not None:
        print(f"Created archive: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
