#!/usr/bin/env python3
"""Aggregate ActiveMap versus learned defer with seed-then-AOI bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.aggregate_sn7_r1_r2_validation import (
    _hierarchical_bootstrap,
    _paired_deltas,
    _read_trace,
    _sha256,
)

SEEDS = (20260730, 20260731, 20260801)
POLICY = "learned_defer"


def _load_receipt(root: Path, seed: int) -> tuple[dict[str, Any], Path]:
    path = root / f"seed{seed}" / "COMPLETE.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "activemap-sn7-learned-defer-extension-seed-v1":
        raise ValueError(f"unexpected learned-defer receipt schema for seed {seed}")
    if payload.get("split") != "val" or payload.get("test_assets_read") is not False:
        raise ValueError(f"seed {seed} is not validation-only")
    if int(payload.get("controller_seed", -1)) != seed:
        raise ValueError(f"receipt seed mismatch at {path}")
    contract = payload.get("comparison_contract", {})
    if not all(
        contract.get(key) is True
        for key in (
            "only_selector_conditioning_differs",
            "same_architecture_training_budget_and_seed",
            "same_recurrent_belief_tool_gate_safe_commit",
        )
    ):
        raise ValueError(f"seed {seed} lacks the matched learned-defer contract")
    selector = payload["selectors"][POLICY]
    if selector.get("condition_on_hypothesis") is not False:
        raise ValueError(f"seed {seed} learned defer is hypothesis-conditioned")
    if selector.get("stop_margin_source") != "checkpoint":
        raise ValueError(f"seed {seed} learned defer uses a STOP-margin override")
    return payload, path


def aggregate(root: Path, *, repetitions: int, seed: int) -> dict[str, Any]:
    receipts = {value: _load_receipt(root, value) for value in SEEDS}
    for key in ("val_states", "val_episodes", "edit_manifest"):
        hashes = {payload["inputs"][key]["sha256"] for payload, _ in receipts.values()}
        if len(hashes) != 1:
            raise ValueError(f"controller seeds disagree on {key} hash")
    shared_components = set(next(iter(receipts.values()))[0]["shared_controller_components"])
    for component in shared_components:
        for controller_seed, (payload, _) in receipts.items():
            item = payload["shared_controller_components"].get(component)
            if item is None or _sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError(
                    f"shared component hash mismatch for seed {controller_seed}: {component}"
                )

    results = {}
    for analysis, key in (
        ("r1_natural", "natural_validation_traces"),
        ("r2_edit_only", "r2_filtered_traces"),
    ):
        paired = {}
        supports = {}
        for controller_seed, (payload, _) in receipts.items():
            active_receipt = payload[key]["activemap"]
            defer_receipt = payload[key][POLICY]
            active_path = Path(active_receipt["path"])
            defer_path = Path(defer_receipt["path"])
            if _sha256(active_path) != active_receipt["sha256"]:
                raise ValueError(f"ActiveMap trace hash mismatch for seed {controller_seed}")
            if _sha256(defer_path) != defer_receipt["sha256"]:
                raise ValueError(f"learned-defer trace hash mismatch for seed {controller_seed}")
            active_rows = _read_trace(active_path)
            defer_rows = _read_trace(defer_path)
            paired[controller_seed] = _paired_deltas(active_rows, defer_rows)
            supports[controller_seed] = len(active_rows)
        results[analysis] = {
            "activemap_minus_learned_defer": _hierarchical_bootstrap(
                paired, repetitions=repetitions, seed=seed
            ),
            "records_per_seed": supports,
        }

    return {
        "schema_version": "activemap-sn7-learned-defer-extension-three-seed-v1",
        "split": "val",
        "test_assets_read": False,
        "seeds": list(SEEDS),
        "comparison_contract": (
            "edit-conditioned ActiveMap minus architecture-matched hypothesis-agnostic "
            "learned defer; shared recurrent belief, selective tool gate, Safe Commit, "
            "one-acquisition budget, states, episodes, and controller seeds"
        ),
        "results": results,
        "receipts": {
            str(controller_seed): {"path": str(path.resolve()), "sha256": _sha256(path)}
            for controller_seed, (_, path) in receipts.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260830)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.bootstrap_repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    result = aggregate(
        args.run_root,
        repetitions=args.bootstrap_repetitions,
        seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["results"], indent=2))


if __name__ == "__main__":
    main()
