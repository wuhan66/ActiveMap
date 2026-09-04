#!/usr/bin/env python3
"""Audit whether static tool competence remains reachable in recurrent rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _record(value: str) -> tuple[int, Path]:
    seed, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("record must use SEED=/path/to/file.json")
    return int(seed), Path(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(
    static: dict[int, dict[str, Any]],
    recurrent: dict[int, dict[str, Any]],
    *,
    method: str = "qwen3_4b_sft_tool_to_belief",
    minimum_seeds: int = 2,
) -> dict[str, Any]:
    if set(static) != set(recurrent) or len(static) < minimum_seeds:
        raise ValueError(
            f"reachability audit requires the same {minimum_seeds} or more seeds"
        )
    rows = []
    for seed in sorted(static):
        decision = static[seed]
        if decision.get("protocol", {}).get("test_assets_read") is not False:
            raise ValueError(f"seed {seed} static decision violates test isolation")
        selected = decision.get("selected_checkpoint")
        checkpoint = next(
            (row for row in decision.get("checkpoints", []) if row.get("label") == selected),
            None,
        )
        if checkpoint is None:
            raise ValueError(f"seed {seed} lacks a selected static checkpoint")
        rollout = recurrent[seed]
        if rollout.get("protocol", {}).get("test_assets_read") is not False:
            raise ValueError(f"seed {seed} rollout violates test isolation")
        counts = rollout.get("action_counts", {}).get(method)
        if not isinstance(counts, dict):
            raise ValueError(f"seed {seed} rollout lacks method={method}")
        static_calls = int(checkpoint.get("predicted_tool_calls", 0))
        recurrent_calls = int(counts.get("USE_TOOL", 0))
        acquisitions = int(counts.get("ACQUIRE", 0))
        rows.append(
            {
                "seed": seed,
                "selected_checkpoint": selected,
                "static_predicted_tool_calls": static_calls,
                "static_grounded_call_recall": float(
                    checkpoint.get("grounded_call_recall", 0.0)
                ),
                "recurrent_acquisitions": acquisitions,
                "recurrent_tool_calls": recurrent_calls,
                "recurrent_call_per_acquisition": (
                    recurrent_calls / acquisitions if acquisitions else 0.0
                ),
                "static_competence_recurrently_unreachable": (
                    static_calls > 0 and acquisitions > 0 and recurrent_calls == 0
                ),
            }
        )
    gap_count = sum(row["static_competence_recurrently_unreachable"] for row in rows)
    replicated = len(rows) >= 2 and gap_count >= len(rows) // 2 + 1
    return {
        "schema_version": "agent-tool-call-reachability-audit-v1",
        "method": method,
        "seeds": sorted(static),
        "per_seed": rows,
        "summary": {
            "single_seed_diagnostic": len(rows) == 1,
            "gap_observed": gap_count > 0,
            "replicated_gap_seed_count": gap_count,
            "static_to_recurrent_reachability_gap_replicated": replicated,
            "flat_policy_agentic_claim_supported": not replicated,
            "required_intervention": (
                "explicit_hierarchical_tool_need_gate" if replicated else None
            ),
        },
        "claim_boundary": (
            "Static action competence is not evidence of recurrent tool use."
        ),
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--static", action="append", type=_record, required=True)
    parser.add_argument("--recurrent", action="append", type=_record, required=True)
    parser.add_argument("--method", default="qwen3_4b_sft_tool_to_belief")
    parser.add_argument("--allow-single-seed-diagnostic", action="store_true")
    args = parser.parse_args()
    static_paths, recurrent_paths = dict(args.static), dict(args.recurrent)
    if len(static_paths) != len(args.static) or len(recurrent_paths) != len(args.recurrent):
        raise ValueError("duplicate seed record")
    report = audit(
        {seed: json.loads(path.read_text()) for seed, path in static_paths.items()},
        {seed: json.loads(path.read_text()) for seed, path in recurrent_paths.items()},
        method=args.method,
        minimum_seeds=1 if args.allow_single_seed_diagnostic else 2,
    )
    report["sources"] = {
        "static": {
            str(seed): {"path": str(path.resolve()), "sha256": _sha256(path)}
            for seed, path in static_paths.items()
        },
        "recurrent": {
            str(seed): {"path": str(path.resolve()), "sha256": _sha256(path)}
            for seed, path in recurrent_paths.items()
        },
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
