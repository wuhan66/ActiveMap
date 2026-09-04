#!/usr/bin/env python3
"""Audit whether a carried-prior trace is a true re-perceive/re-control rollout.

Chronological replay is not enough for a persistent-control claim. This audit
requires every policy-specific next state to receive the exact executed prior
from the preceding step and records fresh candidate-front-end and direct-draft
receipts generated against that carried prior. It deliberately does not assess
quality, so it cannot promote a result by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "policy",
    "chain_id",
    "step",
    "split",
    "test_assets_read",
    "prior_input_sha256",
    "candidate_frontend_prior_sha256",
    "direct_hypothesis_prior_sha256",
    "executed_prior_sha256",
    "candidate_frontend_receipt_sha256",
    "direct_hypothesis_receipt_sha256",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(trace_path: Path, *, expected_split: str = "val") -> dict[str, Any]:
    if expected_split not in {"train", "val"}:
        raise ValueError("true sequential audit only accepts train/validation traces")
    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("sequential trace is empty")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row_number, row in enumerate(rows, 1):
        missing = sorted(REQUIRED_FIELDS - row.keys())
        if missing:
            raise ValueError(f"trace row {row_number} lacks fields: {missing}")
        if row["split"] != expected_split or row["test_assets_read"] is not False:
            raise ValueError(
                f"true sequential audit expected {expected_split}-only traces"
            )
        if not isinstance(row["step"], int) or row["step"] < 0:
            raise ValueError(f"trace row {row_number} has invalid step")
        if any(not isinstance(row[field], str) or len(row[field]) != 64 for field in REQUIRED_FIELDS - {"policy", "chain_id", "step", "split", "test_assets_read"}):
            raise ValueError(f"trace row {row_number} has malformed SHA-256 receipt")
        groups[(str(row["policy"]), str(row["chain_id"]))].append(row)

    policy_chains: dict[str, int] = defaultdict(int)
    transition_count = 0
    changed_prior_transitions = 0
    for (policy, chain_id), chain in sorted(groups.items()):
        chain.sort(key=lambda row: row["step"])
        expected_steps = list(range(len(chain)))
        observed_steps = [row["step"] for row in chain]
        if observed_steps != expected_steps:
            raise ValueError(
                f"policy={policy} chain={chain_id} does not have contiguous steps"
            )
        policy_chains[policy] += 1
        for index, row in enumerate(chain):
            if row["candidate_frontend_prior_sha256"] != row["prior_input_sha256"]:
                raise ValueError(
                    f"policy={policy} chain={chain_id} step={index} reused a frontend from another prior"
                )
            if row["direct_hypothesis_prior_sha256"] != row["prior_input_sha256"]:
                raise ValueError(
                    f"policy={policy} chain={chain_id} step={index} reused a direct hypothesis from another prior"
                )
            if index:
                previous = chain[index - 1]
                if row["prior_input_sha256"] != previous["executed_prior_sha256"]:
                    raise ValueError(
                        f"policy={policy} chain={chain_id} step={index} is not fed the preceding executed prior"
                    )
                transition_count += 1
                changed_prior_transitions += int(
                    row["prior_input_sha256"] != previous["prior_input_sha256"]
                )
    if not transition_count:
        raise ValueError("true sequential audit requires at least one carried transition")
    if len(policy_chains) < 2:
        raise ValueError("true sequential comparison requires at least two policies")
    return {
        "schema_version": "activemap-true-sequential-rollout-audit-v1",
        "passed": True,
        "split": expected_split,
        "test_assets_read": False,
        "trace": {"path": str(trace_path.resolve()), "sha256": _sha256(trace_path)},
        "policy_count": len(policy_chains),
        "chain_count": len(groups),
        "transition_count": transition_count,
        "changed_prior_transition_count": changed_prior_transitions,
        "policy_chains": dict(sorted(policy_chains.items())),
        "claim_boundary": (
            "input-contract audit only; report map quality, safety, cost, and AOI/chain "
            "bootstrap separately before making a sequential efficacy claim"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = audit(args.trace, expected_split=args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
