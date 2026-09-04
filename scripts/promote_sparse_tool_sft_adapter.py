#!/usr/bin/env python3
"""Create an immutable validation-selected SFT adapter promotion record."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def promote_adapter(
    run_dir: Path,
    decision_path: Path,
    output: Path,
    *,
    seed: int,
) -> dict[str, object]:
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("protocol", {}).get("test_assets_read") is not False:
        raise ValueError("checkpoint decision does not prove test isolation")
    if decision.get("selection_passed") is not True:
        raise ValueError("no checkpoint passed the frozen validation gates")
    label = str(decision.get("selected_checkpoint") or "")
    if not label:
        raise ValueError("selection decision lacks selected_checkpoint")
    adapter = run_dir / "checkpoints" / label
    if not adapter.is_dir() and label == "final":
        adapter = run_dir / "final"
    config = adapter / "adapter_config.json"
    weights = sorted(adapter.glob("adapter_model.*"))
    if not config.is_file() or not weights:
        raise ValueError(f"selected adapter is incomplete: {adapter}")
    artifacts = [config, *weights]
    promotion: dict[str, object] = {
        "schema_version": "activemap-sft-adapter-promotion-v1",
        "approved": True,
        "seed": seed,
        "selection_split": "val",
        "test_assets_read": False,
        "selected_checkpoint": label,
        "adapter_path": str(adapter.resolve()),
        "selection_decision": str(decision_path.resolve()),
        "selection_decision_sha256": _sha256(decision_path),
        "adapter_artifacts": [
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in artifacts
        ],
    }
    if output.exists():
        raise FileExistsError(f"refusing to overwrite promotion record: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(promotion, indent=2) + "\n", encoding="utf-8")
    return promotion


def verify_adapter_promotion(path: Path) -> dict[str, object]:
    promotion = json.loads(path.read_text(encoding="utf-8"))
    if promotion.get("schema_version") != "activemap-sft-adapter-promotion-v1":
        raise ValueError("invalid SFT adapter promotion schema")
    if promotion.get("approved") is not True or promotion.get("test_assets_read") is not False:
        raise ValueError("SFT adapter promotion is not approved and test-isolated")
    artifacts = promotion.get("adapter_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("SFT adapter promotion has no artifacts")
    for artifact in artifacts:
        artifact_path = Path(str(artifact["path"]))
        if not artifact_path.is_file():
            raise ValueError(f"promoted adapter artifact is missing: {artifact_path}")
        if artifact_path.stat().st_size != int(artifact["size_bytes"]):
            raise ValueError(f"promoted adapter artifact size changed: {artifact_path}")
        if _sha256(artifact_path) != artifact["sha256"]:
            raise ValueError(f"promoted adapter artifact hash changed: {artifact_path}")
    return promotion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("decision", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            promote_adapter(
                args.run_dir,
                args.decision,
                args.output,
                seed=args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
