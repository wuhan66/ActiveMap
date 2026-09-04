#!/usr/bin/env python3
"""Anchor regenerated corrupted states to the frozen Step-0 observable schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(row: dict[str, Any]) -> tuple[str, float]:
    metadata = row["metadata"]
    return str(metadata["source_episode"]), float(metadata["budget"])


def anchor_row(
    frozen: dict[str, Any],
    regenerated: dict[str, Any],
) -> dict[str, Any]:
    """Move perception-dependent fields into an otherwise frozen state row."""

    if _identity(frozen) != _identity(regenerated):
        raise ValueError("frozen and regenerated state identities differ")
    for field in ("split", "sample_id", "evidence_ids", "evidence_costs"):
        if frozen[field] != regenerated[field]:
            raise ValueError(f"frozen state invariant changed: {field}")
    for field in ("aoi_id", "gt_edit", "initial_evidence_id"):
        if frozen["metadata"][field] != regenerated["metadata"][field]:
            raise ValueError(f"frozen metadata invariant changed: {field}")

    anchored = json.loads(json.dumps(frozen))
    anchored["edit_type"] = regenerated["edit_type"]
    for index in (0, 1, 2, 3, 12, 13):
        anchored["hypothesis_features"][index] = regenerated[
            "hypothesis_features"
        ][index]
    for index in (2, 3, 4, 5, 7):
        anchored["state_features"][index] = regenerated["state_features"][index]
    if len(anchored["evidence_features"]) != len(
        regenerated["evidence_features"]
    ):
        raise ValueError("evidence feature counts differ")
    for target, source in zip(
        anchored["evidence_features"],
        regenerated["evidence_features"],
        strict=True,
    ):
        target[11] = source[11]
    anchored["oracle_utilities"] = regenerated["oracle_utilities"]
    anchored["metadata"]["executable_outcomes"] = regenerated["metadata"][
        "executable_outcomes"
    ]
    anchored["metadata"]["evidence_predictions"] = regenerated["metadata"][
        "evidence_predictions"
    ]
    anchored["metadata"]["prior_input_corruption"] = regenerated["metadata"][
        "prior_input_corruption"
    ]
    return anchored


def anchor_file(
    frozen_path: Path,
    regenerated_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(output_path)
    frozen_rows = [
        json.loads(line)
        for line in frozen_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frozen = {_identity(row): row for row in frozen_rows}
    if len(frozen) != len(frozen_rows):
        raise ValueError("duplicate frozen Step-0 identities")

    regenerated_rows = []
    for line in regenerated_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row["metadata"].get("oracle_step", -1)) == 0:
            regenerated_rows.append(row)
    regenerated = {_identity(row): row for row in regenerated_rows}
    if len(regenerated) != len(regenerated_rows):
        raise ValueError("duplicate regenerated Step-0 identities")
    if set(regenerated) != set(frozen):
        missing = len(set(frozen) - set(regenerated))
        extra = len(set(regenerated) - set(frozen))
        raise ValueError(
            f"regenerated/frozen identity mismatch: missing={missing}, extra={extra}"
        )

    anchored = [anchor_row(row, regenerated[_identity(row)]) for row in frozen_rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in anchored:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    protocols = {
        json.dumps(
            {
                "translation_pixels": int(
                    row["metadata"]["prior_input_corruption"].get(
                        "translation_pixels", 0
                    )
                ),
                "morphology": str(
                    row["metadata"]["prior_input_corruption"].get(
                        "morphology", "none"
                    )
                ),
                "morphology_pixels": int(
                    row["metadata"]["prior_input_corruption"].get(
                        "morphology_pixels", 0
                    )
                ),
                "corruption_seed": int(
                    row["metadata"]["prior_input_corruption"].get(
                        "corruption_seed", 0
                    )
                ),
                "scope": str(
                    row["metadata"]["prior_input_corruption"].get(
                        "scope", "model_input_only"
                    )
                ),
            },
            sort_keys=True,
        )
        for row in anchored
    }
    if len(protocols) != 1:
        raise ValueError("anchored rows mix corruption protocols")
    protocol = json.loads(next(iter(protocols)))
    summary = {
        "schema_version": "sn7-frozen-step0-corruption-anchor-v2",
        "frozen_path": str(frozen_path.resolve()),
        "frozen_sha256": _sha256(frozen_path),
        "regenerated_path": str(regenerated_path.resolve()),
        "regenerated_sha256": _sha256(regenerated_path),
        "output_path": str(output_path.resolve()),
        "output_sha256": _sha256(output_path),
        "state_count": len(anchored),
        "episode_count": len({identity[0] for identity in frozen}),
        "prior_input_corruption": protocol,
        "translation_pixels": protocol["translation_pixels"],
        "morphology": protocol["morphology"],
        "morphology_pixels": protocol["morphology_pixels"],
        "corruption_seed": protocol["corruption_seed"],
        "frozen_fields": [
            "false_edit_risks",
            "hypothesis_features[14]",
            "all non-perception observables",
        ],
        "transplanted_fields": [
            "edit_type",
            "initial edit probabilities and confidence",
            "uncertainty feature",
            "oracle utilities",
            "evidence predictions",
            "executable outcomes",
        ],
        "split": "val",
        "test_assets_read": False,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen", type=Path)
    parser.add_argument("regenerated", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(anchor_file(args.frozen, args.regenerated, args.output), indent=2))


if __name__ == "__main__":
    main()
