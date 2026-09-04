#!/usr/bin/env python3
"""Validate the paper claim-to-baseline-to-metric protocol."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ALLOWED_STATUSES = {
    "protocol_ready",
    "incomplete",
    "partial_validation_support",
    "final_candidate_evaluation_pending",
    "validation_supported",
    "validation_supported_negative_boundary",
    "planned",
}
SUPPORTED_STATUSES = {
    "validation_supported",
    "validation_supported_negative_boundary",
}


def duplicates(values: list[str]) -> list[str]:
    return sorted(key for key, count in Counter(values).items() if count > 1)


def audit(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []

    anchors = {str(item["id"]): item for item in payload.get("literature_anchors", [])}
    baselines = {str(item["id"]): item for item in payload.get("baselines", [])}
    figures = {str(item["id"]): item for item in payload.get("figures", [])}
    claims = payload.get("claims", [])

    for label, items in (
        ("literature anchor", list(anchors)),
        ("baseline", list(baselines)),
        ("figure", list(figures)),
        ("claim", [str(item.get("id", "")) for item in claims]),
    ):
        found = duplicates(items)
        if found:
            errors.append(f"duplicate {label} ids: {found}")

    origin_anchors: set[str] = set()
    for section in payload.get("problem_origin", {}).values():
        if isinstance(section, dict):
            origin_anchors.update(map(str, section.get("anchors", [])))
    unknown_problem_anchors = sorted(origin_anchors - set(anchors))
    if unknown_problem_anchors:
        errors.append(f"problem origin uses unknown anchors: {unknown_problem_anchors}")

    required_metric_order = payload.get("protocol_invariants", {}).get(
        "primary_metric_order", []
    )
    if required_metric_order[:3] != [
        "executable_map_quality",
        "false_edit_safety",
        "quality_cost_frontier",
    ]:
        errors.append("primary metric order must start with quality, safety, and cost frontier")

    claim_rows: list[dict[str, Any]] = []
    for claim in claims:
        claim_id = str(claim.get("id", "<missing>"))
        status = str(claim.get("status", ""))
        if status not in ALLOWED_STATUSES:
            errors.append(f"{claim_id}: invalid status {status!r}")

        unknown_origins = sorted(set(map(str, claim.get("origin", []))) - set(anchors))
        unknown_baselines = sorted(
            set(map(str, claim.get("comparisons", []))) - set(baselines)
        )
        unknown_figures = sorted(set(map(str, claim.get("figures", []))) - set(figures))
        if unknown_origins:
            errors.append(f"{claim_id}: unknown literature anchors {unknown_origins}")
        if unknown_baselines:
            errors.append(f"{claim_id}: unknown baselines {unknown_baselines}")
        if unknown_figures:
            errors.append(f"{claim_id}: unknown figures {unknown_figures}")
        if not claim.get("statement"):
            errors.append(f"{claim_id}: missing statement")
        if not claim.get("metrics"):
            errors.append(f"{claim_id}: missing metrics")
        if not claim.get("comparisons"):
            errors.append(f"{claim_id}: missing comparisons")

        missing_evidence: list[str] = []
        for evidence in map(str, claim.get("evidence", [])):
            evidence_path = repo_root / evidence
            if not evidence_path.is_file() or evidence_path.stat().st_size == 0:
                missing_evidence.append(evidence)
        if status in SUPPORTED_STATUSES and not claim.get("evidence"):
            warnings.append(f"{claim_id}: supported status has no local evidence selector")
        if missing_evidence:
            errors.append(f"{claim_id}: missing evidence files {missing_evidence}")

        claim_rows.append(
            {
                "id": claim_id,
                "status": status,
                "comparison_count": len(claim.get("comparisons", [])),
                "metric_count": len(claim.get("metrics", [])),
                "figure_count": len(claim.get("figures", [])),
                "evidence_files": len(claim.get("evidence", [])),
            }
        )

    gates = payload.get("promotion_gates", {}).get("main_controller", {})
    unknown_gate_baselines = sorted(
        set(map(str, gates.get("required_baselines", []))) - set(baselines)
    )
    if unknown_gate_baselines:
        errors.append(f"main promotion gate uses unknown baselines: {unknown_gate_baselines}")
    if str(gates.get("candidate", "")) not in baselines:
        errors.append("main promotion gate candidate is not registered")

    ready_baselines = sum(
        item.get("readiness") in {"ready", "checkpoint_ready"}
        for item in baselines.values()
    )
    return {
        "schema_version": "activemap-claim-evidence-audit-v1",
        "protocol": str(path),
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "literature_anchors": len(anchors),
            "baselines": len(baselines),
            "ready_or_checkpoint_ready_baselines": ready_baselines,
            "claims": len(claims),
            "figures": len(figures),
            "fully_supported_claims": sum(
                row["status"] in SUPPORTED_STATUSES for row in claim_rows
            ),
        },
        "claims": claim_rows,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ActiveMap Claim-Evidence Protocol Audit",
        "",
        f"- Passed: `{str(report['passed']).lower()}`",
        f"- Literature anchors: `{report['summary']['literature_anchors']}`",
        f"- Baselines: `{report['summary']['baselines']}`",
        "- Ready/checkpoint-ready baselines: "
        f"`{report['summary']['ready_or_checkpoint_ready_baselines']}`",
        f"- Claims: `{report['summary']['claims']}`",
        f"- Fully supported claims: `{report['summary']['fully_supported_claims']}`",
        f"- Planned figures: `{report['summary']['figures']}`",
        "",
        "| Claim | Status | Baselines | Metrics | Figures | Evidence files |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report["claims"]:
        lines.append(
            f"| {row['id']} | {row['status']} | {row['comparison_count']} | "
            f"{row['metric_count']} | {row['figure_count']} | {row['evidence_files']} |"
        )
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {item}" for item in report["errors"])
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in report["warnings"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/experiments/claim_evidence_protocol_v2.yaml"),
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = args.protocol.resolve()
    repo_root = protocol.parents[2]
    report = audit(protocol, repo_root)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
