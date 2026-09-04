#!/usr/bin/env python3
"""Verify that anchored corruption states differ only on approved fields."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


HYPOTHESIS_INDICES = (0, 1, 2, 3, 12, 13)
STATE_INDICES = (2, 3, 4, 5, 7)
REALIZATION_FIELDS = ("shift_x", "shift_y")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _identity(row: dict[str, Any]) -> tuple[str, float]:
    metadata = row["metadata"]
    return str(metadata["source_episode"]), float(metadata["budget"])


def _corruption_config(corruption: Any) -> dict[str, Any]:
    if not isinstance(corruption, dict):
        raise ValueError("missing corruption provenance")
    config = copy.deepcopy(corruption)
    radius = config.get("translation_pixels", 0)
    if not isinstance(radius, int) or isinstance(radius, bool) or radius < 0:
        raise ValueError("invalid translation_pixels")
    for key in REALIZATION_FIELDS:
        shift = config.pop(key, 0)
        if not isinstance(shift, int) or isinstance(shift, bool) or abs(shift) > radius:
            raise ValueError(f"invalid {key} for translation radius {radius}")
    if config.get("scope") != "model_input_only":
        raise ValueError("corruption scope must be model_input_only")
    morphology = config.get("morphology", "none")
    morphology_pixels = config.get("morphology_pixels", 0)
    if morphology not in {"none", "erode", "dilate"}:
        raise ValueError("invalid morphology")
    if (
        not isinstance(morphology_pixels, int)
        or isinstance(morphology_pixels, bool)
        or morphology_pixels < 0
        or (morphology == "none" and morphology_pixels != 0)
        or (morphology != "none" and morphology_pixels == 0)
    ):
        raise ValueError("invalid morphology_pixels")
    config["morphology"] = morphology
    config["morphology_pixels"] = morphology_pixels
    return config


def _normalize_allowed(
    anchored: dict[str, Any],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    normalized = copy.deepcopy(anchored)
    normalized["edit_type"] = frozen["edit_type"]
    for index in HYPOTHESIS_INDICES:
        normalized["hypothesis_features"][index] = frozen["hypothesis_features"][index]
    for index in STATE_INDICES:
        normalized["state_features"][index] = frozen["state_features"][index]
    if len(normalized["evidence_features"]) != len(frozen["evidence_features"]):
        raise ValueError("evidence feature counts differ")
    for current, reference in zip(
        normalized["evidence_features"],
        frozen["evidence_features"],
        strict=True,
    ):
        current[11] = reference[11]
    normalized["oracle_utilities"] = frozen["oracle_utilities"]
    for key in (
        "executable_outcomes",
        "evidence_predictions",
        "prior_input_corruption",
    ):
        if key == "prior_input_corruption":
            normalized["metadata"].pop(key, None)
        elif key in frozen["metadata"]:
            normalized["metadata"][key] = frozen["metadata"][key]
        else:
            normalized["metadata"].pop(key, None)
    return normalized


def audit(frozen_path: Path, anchored_path: Path) -> dict[str, Any]:
    frozen_rows = _read(frozen_path)
    anchored_rows = _read(anchored_path)
    if len(frozen_rows) != len(anchored_rows):
        raise ValueError("frozen and anchored row counts differ")
    frozen = {_identity(row): row for row in frozen_rows}
    anchored = {_identity(row): row for row in anchored_rows}
    if len(frozen) != len(frozen_rows) or len(anchored) != len(anchored_rows):
        raise ValueError("state identities must be unique")
    if set(frozen) != set(anchored):
        raise ValueError("frozen and anchored identities differ")

    changed = 0
    protocols = set()
    realizations = set()
    for identity, reference in frozen.items():
        current = anchored[identity]
        if current != reference:
            changed += 1
        corruption = current["metadata"].get("prior_input_corruption")
        try:
            config = _corruption_config(corruption)
        except ValueError as error:
            raise ValueError(f"{error}: {identity}") from error
        protocols.add(json.dumps(config, sort_keys=True))
        realizations.add(
            (
                int(corruption.get("shift_x", 0)),
                int(corruption.get("shift_y", 0)),
            )
        )
        normalized = _normalize_allowed(current, reference)
        if normalized != reference:
            raise ValueError(f"non-whitelisted anchored difference: {identity}")
    if len(protocols) != 1:
        raise ValueError("anchored rows mix corruption protocols")
    if changed == 0:
        raise ValueError("corruption intervention changed no anchored rows")

    return {
        "schema_version": "sn7-frozen-anchor-parity-audit-v1",
        "passed": True,
        "state_count": len(anchored_rows),
        "episode_count": len({identity[0] for identity in anchored}),
        "changed_state_count": changed,
        "prior_input_corruption": json.loads(next(iter(protocols))),
        "corruption_realization_count": len(realizations),
        "frozen_sha256": _sha256(frozen_path),
        "anchored_sha256": _sha256(anchored_path),
        "allowed_differences": {
            "edit_type": True,
            "hypothesis_features": list(HYPOTHESIS_INDICES),
            "state_features": list(STATE_INDICES),
            "evidence_feature_index": 11,
            "oracle_utilities": True,
            "metadata": [
                "executable_outcomes",
                "evidence_predictions",
                "prior_input_corruption",
            ],
        },
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen", type=Path)
    parser.add_argument("anchored", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(args.frozen, args.anchored)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
