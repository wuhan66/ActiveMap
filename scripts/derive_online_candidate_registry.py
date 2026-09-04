"""Derive a train-internal online registry with one candidate updater checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_registry", type=Path)
    parser.add_argument("updater_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--controller-seed", type=int, required=True)
    parser.add_argument(
        "--method", default="carried_prior_residual_updater_headroom"
    )
    parser.add_argument(
        "--purpose", default="train_internal_carried_prior_raw_headroom_gate"
    )
    return parser.parse_args()


def derive(
    base_registry: Path,
    updater_checkpoint: Path,
    *,
    controller_seed: int,
    method: str = "carried_prior_residual_updater_headroom",
    purpose: str = "train_internal_carried_prior_raw_headroom_gate",
) -> dict[str, Any]:
    base = yaml.safe_load(base_registry.read_text(encoding="utf-8"))
    if not isinstance(base, dict) or base.get("test_assets_read") is not False:
        raise ValueError("base registry must be validation-only")
    seed_key = str(controller_seed)
    seed_assets = base.get("seed_artifacts", {}).get(seed_key)
    protocol = base.get("protocol")
    if not isinstance(seed_assets, dict) or not isinstance(protocol, dict):
        raise ValueError("base registry lacks controller seed or writeback protocol")
    if not updater_checkpoint.is_file():
        raise FileNotFoundError(updater_checkpoint)
    if not method or not purpose:
        raise ValueError("method and purpose must be non-empty")
    return {
        "schema_version": "sn7-online-candidate-updater-registry-v1",
        "status": "train_internal_candidate_gate_only",
        "dataset": base.get("dataset", "sn7"),
        "method": method,
        "seeds": [controller_seed],
        "test_policy": "not_authorized",
        "test_assets_read": False,
        "protocol": protocol,
        # The evaluator consumes this stable field name. Its provenance below
        # makes explicit that it is a candidate, not a promoted frozen updater.
        "shared_artifacts": [
            {
                "id": "frozen_updater",
                "path": str(updater_checkpoint.resolve()),
                "sha256": _sha256(updater_checkpoint),
            }
        ],
        "seed_artifacts": {seed_key: seed_assets},
        "online_state_contract": base.get("online_state_contract", {}),
        "provenance": {
            "base_registry": {
                "path": str(base_registry.resolve()),
                "sha256": _sha256(base_registry),
            },
            "candidate_updater": {
                "path": str(updater_checkpoint.resolve()),
                "sha256": _sha256(updater_checkpoint),
                "purpose": purpose,
            },
        },
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    registry = derive(
        args.base_registry,
        args.updater_checkpoint,
        controller_seed=args.controller_seed,
        method=args.method,
        purpose=args.purpose,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
