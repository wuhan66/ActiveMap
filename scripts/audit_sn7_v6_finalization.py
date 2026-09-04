#!/usr/bin/env python3
"""Audit a completed V6 validation finalization without changing its result.

The script treats the finalizer's aggregate ``promotion`` object as the sole
decision source. It validates registered provenance and writes an optional
human-readable receipt; it never recalibrates, recomputes intervals, or opens
test assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SEEDS = (20260817, 20260818, 20260819)
FINALIZATION_SCHEMA = "sn7-v6-evidence-value-finalization-v1"
AGGREGATE_SCHEMA = "sn7-v6-evidence-value-matched-factorial-v1"
PROMOTION_CHECKS = {
    "selection_final_map_quality_lower_positive",
    "selection_false_edit_upper_nonpositive",
    "selection_missed_edit_upper_nonpositive",
    "safe_commit_false_edit_upper_nonpositive",
    "safe_commit_final_map_quality_lower_nonnegative",
    "selected_cost_upper_strictly_below_forced",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit(v6_root: Path, finalization_dir: Path) -> dict[str, Any]:
    protocol_path = v6_root / "protocol.json"
    queue_path = v6_root / "queue_status.json"
    receipt_path = finalization_dir / "finalization_receipt.json"
    aggregate_path = finalization_dir / "three_seed_factorial_with_forced_summary.json"

    protocol = load_json(protocol_path)
    queue = load_json(queue_path)
    receipt = load_json(receipt_path)
    aggregate = load_json(aggregate_path)

    require(queue.get("status") == "complete", "V6 queue is not complete")
    require(queue.get("test_assets_read") is False, "V6 queue accessed test assets")
    require(protocol.get("test_assets_read") is False, "V6 protocol permits test access")
    require(receipt.get("schema_version") == FINALIZATION_SCHEMA, "unexpected finalization schema")
    require(receipt.get("test_assets_read") is False, "finalizer accessed test assets")
    require(
        receipt.get("forced_safe_commit_only_uses_train_fitted_gates") is True,
        "forced Safe Commit was not declared train-fitted",
    )
    require(aggregate.get("schema_version") == AGGREGATE_SCHEMA, "unexpected aggregate schema")
    require(aggregate.get("test_assets_read") is False, "aggregate accessed test assets")
    require(
        tuple(aggregate.get("model_seeds", ())) == SEEDS,
        "aggregate seeds differ from registration",
    )
    require(aggregate.get("policies") == [
        "direct_commit",
        "direct_safe_commit",
        "selected_commit",
        "selected_safe_commit",
        "forced_safe_commit",
    ], "aggregate lacks the registered five policy cells")
    require(
        aggregate.get("v6_protocol_sha256") == sha256(protocol_path),
        "aggregate protocol hash does not match registered protocol",
    )
    require(
        receipt.get("v6_protocol_sha256") == sha256(protocol_path),
        "finalization receipt protocol hash does not match registered protocol",
    )

    promotion = aggregate.get("promotion")
    require(isinstance(promotion, dict), "aggregate has no promotion decision")
    checks = promotion.get("checks")
    require(isinstance(checks, dict), "promotion has no gate checks")
    require(
        set(checks) == PROMOTION_CHECKS,
        "promotion checks differ from the registered six-gate decision rule",
    )
    require(
        all(isinstance(value, bool) for value in checks.values()),
        "promotion checks must be boolean",
    )
    eligible = promotion.get("eligible_for_extension_claim")
    require(isinstance(eligible, bool), "promotion eligibility must be boolean")
    require(
        eligible is all(checks.values()),
        "promotion eligibility is inconsistent with its gate checks",
    )
    forced = promotion.get("forced_acquisition_cost_control")
    require(isinstance(forced, dict), "forced-acquisition control was not evaluated")
    require("paired_delta" in forced, "forced-acquisition contrast is incomplete")

    return {
        "integrity_passed": True,
        "promotion_eligible": bool(promotion.get("eligible_for_extension_claim")),
        "promotion_checks": checks,
        "promotion_reason": str(promotion.get("reason", "")),
        "protocol_sha256": sha256(protocol_path),
        "model_seeds": list(SEEDS),
        "test_assets_read": False,
        "aggregate": str(aggregate_path.resolve()),
        "finalization_receipt": str(receipt_path.resolve()),
    }


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# SN7 V6 Finalization Audit",
        "",
        f"- Integrity: `{'PASS' if result['integrity_passed'] else 'FAIL'}`",
        f"- Promotion eligibility: `{result['promotion_eligible']}`",
        f"- Test assets read by V6: `{result['test_assets_read']}`",
        f"- Protocol SHA-256: `{result['protocol_sha256']}`",
        f"- Seeds: `{', '.join(map(str, result['model_seeds']))}`",
        "",
        "## Predeclared Promotion Checks",
        "",
        "| Check | Pass |",
        "| --- | --- |",
    ]
    lines.extend(f"| `{name}` | `{value}` |" for name, value in result["promotion_checks"].items())
    lines.extend(
        [
            "",
            "## Finalizer Reason",
            "",
            result["promotion_reason"],
            "",
            "## Authoritative Artifacts",
            "",
            f"- Aggregate: `{result['aggregate']}`",
            f"- Finalization receipt: `{result['finalization_receipt']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("v6_writeback_root", type=Path)
    parser.add_argument("finalization_dir", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    result = audit(args.v6_writeback_root, args.finalization_dir)
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
