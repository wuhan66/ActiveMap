#!/usr/bin/env python3
"""Render paired rollout reports into an auditable claim matrix."""

from __future__ import annotations

import argparse
import csv
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


def _format_interval(summary: dict[str, Any]) -> str:
    low, high = map(float, summary["ci95"])
    return f"{float(summary['mean']):.6f} [{low:.6f}, {high:.6f}]"


def render(report_paths: list[Path], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse paired table directory: {output_dir}")
    reports = []
    for path in sorted(report_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "activemap-paired-rollout-report-v1":
            raise ValueError(f"invalid paired report schema: {path}")
        reports.append((path, payload))
    if not reports:
        raise ValueError("no paired rollout reports")
    ids = [payload["comparison_id"] for _, payload in reports]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate paired comparison ids")

    rows = []
    for _, payload in reports:
        primary_budget = f"{float(payload['budgets'][len(payload['budgets']) // 2]):g}"
        primary = payload["by_budget"][primary_budget]
        rows.append(
            {
                "comparison_id": payload["comparison_id"],
                "claim": payload["claim"],
                "passed": str(bool(payload["all_required_gates_passed"])).lower(),
                "utility_delta_ci95": _format_interval(
                    primary["quality_cost_utility"]["delta_candidate_minus_baseline"]
                ),
                "quality_cost_auc_delta_ci95": _format_interval(
                    payload["quality_cost_auc"]["delta_candidate_minus_baseline"]
                ),
                "false_edit_delta_ci95": _format_interval(
                    primary["false_edit_rate"]["delta_candidate_minus_baseline"]
                ),
                "candidate_false_edit_ci95": _format_interval(
                    primary["false_edit_rate"]["candidate"]
                ),
                "cost_delta_ci95": _format_interval(
                    primary["mean_cost"]["delta_candidate_minus_baseline"]
                ),
                **{
                    f"gate_{name}": str(bool(value)).lower()
                    for name, value in sorted(payload["gates"].items())
                },
            }
        )
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError("paired reports expose inconsistent gate sets")

    output_dir.mkdir(parents=True)
    csv_path = output_dir / "paired_claim_matrix.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    markdown_path = output_dir / "paired_claim_matrix.md"
    compact = [
        "comparison_id",
        "passed",
        "utility_delta_ci95",
        "quality_cost_auc_delta_ci95",
        "false_edit_delta_ci95",
        "cost_delta_ci95",
    ]
    lines = [
        "# Paired Claim Matrix",
        "",
        "| " + " | ".join(compact) + " |",
        "|" + "|".join("---" for _ in compact) + "|",
    ]
    lines.extend(
        "| " + " | ".join(str(row[column]) for column in compact) + " |"
        for row in rows
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "activemap-paired-table-manifest-v1",
        "report_count": len(reports),
        "reports": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path, _ in reports
        ],
        "tables": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in (csv_path, markdown_path)
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    paths = sorted(args.reports_root.glob("*.json"))
    print(json.dumps(render(paths, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
