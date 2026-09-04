#!/usr/bin/env python3
"""Freeze an EDIT-only validation support from initial controller states.

The sealed SN7 test cannot be repartitioned. This utility creates a separate,
validation-only support for ADD/DELETE/RESHAPE analysis. It filters only the
registered target operation and initial state; it never reads predictions,
policy outcomes, utilities, or test assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from activemap.models import EditOperation
from activemap.selector_records import SelectorSample


EDIT_OPERATIONS = (EditOperation.ADD, EditOperation.DELETE, EditOperation.RESHAPE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_operation(sample: SelectorSample) -> EditOperation:
    return EditOperation(str(sample.metadata["gt_edit"]))


def build_manifest(states_path: Path) -> dict[str, Any]:
    """Create a deterministic complete support over validation edit states."""

    selected: list[SelectorSample] = []
    identities: set[tuple[str, float]] = set()
    for line_number, line in enumerate(
        states_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        sample = SelectorSample.model_validate_json(line)
        if sample.split != "val" or int(sample.metadata.get("oracle_step", -1)) != 0:
            continue
        target = _target_operation(sample)
        if target not in EDIT_OPERATIONS:
            continue
        identity = (str(sample.metadata["source_episode"]), float(sample.metadata["budget"]))
        if identity in identities:
            raise ValueError(f"duplicate initial edit state at line {line_number}: {identity}")
        identities.add(identity)
        selected.append(sample)
    if not selected:
        raise ValueError("no validation ADD/DELETE/RESHAPE initial states")

    selected.sort(
        key=lambda sample: (
            str(sample.metadata["aoi_id"]),
            _target_operation(sample).value,
            str(sample.metadata["source_episode"]),
            float(sample.metadata["budget"]),
            sample.sample_id,
        )
    )
    by_operation = Counter(_target_operation(sample).value for sample in selected)
    by_aoi_operation: dict[str, Counter[str]] = defaultdict(Counter)
    records = []
    for sample in selected:
        operation = _target_operation(sample).value
        aoi_id = str(sample.metadata["aoi_id"])
        by_aoi_operation[aoi_id][operation] += 1
        records.append(
            {
                "sample_id": sample.sample_id,
                "source_episode": str(sample.metadata["source_episode"]),
                "aoi_id": aoi_id,
                "budget": float(sample.metadata["budget"]),
                "target_edit": operation,
            }
        )
    missing = [operation.value for operation in EDIT_OPERATIONS if not by_operation[operation.value]]
    if missing:
        raise ValueError(f"EDIT-only manifest lacks operation support: {missing}")
    return {
        "schema_version": "activemap-edit-only-validation-manifest-v1",
        "split": "val",
        "test_assets_read": False,
        "selection_contract": {
            "state": "registered initial controller state (oracle_step=0)",
            "target_operations": [operation.value for operation in EDIT_OPERATIONS],
            "selection_uses_policy_outcome": False,
            "selection_uses_oracle_utility": False,
            "selection_uses_prediction": False,
            "support": "complete eligible validation support; no outcome-based subsampling",
        },
        "source": {"path": str(states_path.resolve()), "sha256": _sha256(states_path)},
        "record_count": len(records),
        "aoi_count": len(by_aoi_operation),
        "operation_counts": dict(sorted(by_operation.items())),
        "aoi_operation_counts": {
            aoi: dict(sorted(counts.items())) for aoi, counts in sorted(by_aoi_operation.items())
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("val_states", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = build_manifest(args.val_states)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
