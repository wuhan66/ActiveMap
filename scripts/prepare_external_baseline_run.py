#!/usr/bin/env python3
"""Create an immutable run manifest for one registered external baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_baseline(
    registry: dict[str, Any], suite_id: str, baseline_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    suites = [suite for suite in registry.get("suites", []) if suite.get("id") == suite_id]
    if len(suites) != 1:
        raise ValueError(f"expected one suite {suite_id!r}, found {len(suites)}")
    suite = suites[0]
    matches = [
        item for item in suite.get("baselines", []) if item.get("id") == baseline_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one baseline {suite_id}/{baseline_id}, found {len(matches)}"
        )
    return suite, matches[0]


def build_manifest(
    registry_path: Path,
    suite_id: str,
    baseline_id: str,
    stage: str,
    seed: int,
    split: str,
    dataset_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    suite, baseline = resolve_baseline(registry, suite_id, baseline_id)
    source_commit = baseline.get("source_commit")
    if not source_commit:
        raise ValueError(
            f"{suite_id}/{baseline_id} has no immutable source_commit; complete source audit first"
        )
    if baseline.get("license_status") in {
        "review_required",
        "research_only_review",
        "no_license_detected",
    }:
        raise ValueError(
            f"{suite_id}/{baseline_id} has unresolved license_status="
            f"{baseline.get('license_status')!r}"
        )
    allowed_stages = {"prepare", "train", "validate", "test"}
    if stage not in allowed_stages:
        raise ValueError(f"invalid stage {stage!r}; expected one of {sorted(allowed_stages)}")
    if split == "test" and stage != "test":
        raise ValueError("test split may only be used by the test stage")
    if stage == "test" and split != "test":
        raise ValueError("test stage requires split=test")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "schema_version": "activemap-external-baseline-run-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite_id": suite_id,
        "baseline_id": baseline_id,
        "display_name": baseline.get("display_name"),
        "dataset": suite.get("dataset"),
        "comparison_scope": suite.get("comparison_scope"),
        "stage": stage,
        "split": split,
        "seed": seed,
        "adapter": baseline.get("adapter"),
        "source": {
            "url": baseline.get("source_url"),
            "commit": source_commit,
            "license_status": baseline.get("license_status"),
            "origin": baseline.get("origin"),
            "paper_title": baseline.get("paper_title"),
            "venue": baseline.get("venue"),
            "publication_year": baseline.get("publication_year"),
            "publication_date": str(baseline.get("publication_date")),
            "paper_url": baseline.get("paper_url"),
            "reproducibility": baseline.get("reproducibility"),
        },
        "inputs": {
            "registry": str(registry_path.resolve()),
            "registry_sha256": sha256_file(registry_path),
            "dataset_manifest": str(dataset_manifest.resolve()),
            "dataset_manifest_sha256": sha256_file(dataset_manifest),
        },
        "metrics": suite.get("metrics", []),
        "output_contract": registry.get("protocol", {}).get("output_contract"),
        "output_dir": str(output_dir.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--stage", choices=["prepare", "train", "validate", "test"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(
        registry_path=args.registry,
        suite_id=args.suite,
        baseline_id=args.baseline,
        stage=args.stage,
        seed=args.seed,
        split=args.split,
        dataset_manifest=args.dataset_manifest,
        output_dir=args.output_dir,
    )
    output = args.output_dir / "run_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
