#!/usr/bin/env python3
"""Validate the external baseline registry and render a progress report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ALLOWED_SCOPES = {"direct", "adapted", "protocol"}
ALLOWED_PRIORITIES = {"critical", "important", "optional"}
ALLOWED_STATUSES = {
    "source_audit",
    "implementation_pending",
    "implementation_ready",
    "validation_running",
    "validation_complete",
    "frozen_test_complete",
}
READY_STATUSES = {
    "implementation_ready",
    "validation_running",
    "validation_complete",
    "frozen_test_complete",
}
FROZEN_TEST_READY_STATUSES = {"validation_complete", "frozen_test_complete"}


def audit_registry(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []

    if payload.get("schema_version") != "activemap-external-baselines-v1":
        errors.append("unsupported schema_version")
    protocol = payload.get("protocol", {})
    for field in (
        "splits_from_activemap_manifests_only",
        "validation_selects_checkpoint_and_thresholds",
        "frozen_test_once",
        "common_preprocessing_required",
        "immutable_source_commit_required_before_run",
        "license_review_required_before_distribution",
    ):
        if protocol.get(field) is not True:
            errors.append(f"protocol.{field} must be true")

    suites = payload.get("suites", [])
    suite_ids = [str(suite.get("id", "")) for suite in suites]
    duplicates = sorted(key for key, count in Counter(suite_ids).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate suite ids: {duplicates}")

    global_ids: list[str] = []
    for suite in suites:
        suite_id = str(suite.get("id", "<missing>"))
        scope = str(suite.get("comparison_scope", ""))
        priority = str(suite.get("priority", ""))
        metrics = list(map(str, suite.get("metrics", [])))
        if scope not in ALLOWED_SCOPES:
            errors.append(f"{suite_id}: invalid comparison_scope={scope!r}")
        if priority not in ALLOWED_PRIORITIES:
            errors.append(f"{suite_id}: invalid priority={priority!r}")
        if not metrics:
            errors.append(f"{suite_id}: metrics must be non-empty")
        baselines = suite.get("baselines", [])
        if not baselines:
            errors.append(f"{suite_id}: baselines must be non-empty")
        local_ids = [str(item.get("id", "")) for item in baselines]
        local_duplicates = sorted(
            key for key, count in Counter(local_ids).items() if count > 1
        )
        if local_duplicates:
            errors.append(f"{suite_id}: duplicate baseline ids: {local_duplicates}")
        global_ids.extend(f"{suite_id}/{baseline_id}" for baseline_id in local_ids)

        for item in baselines:
            baseline_id = str(item.get("id", "<missing>"))
            label = f"{suite_id}/{baseline_id}"
            status = str(item.get("status", ""))
            source_url = item.get("source_url")
            source_commit = item.get("source_commit")
            license_status = str(item.get("license_status", ""))
            reproducibility = item.get("reproducibility", {})
            adapter = str(item.get("adapter", ""))
            item_priority = str(item.get("priority", priority))
            origin = str(item.get("origin", ""))
            paper_title = str(item.get("paper_title", ""))
            venue = str(item.get("venue", ""))
            publication_year = item.get("publication_year")
            publication_date = str(item.get("publication_date", ""))
            paper_url = item.get("paper_url")
            validation_evidence = item.get("validation_evidence")
            if status not in ALLOWED_STATUSES:
                errors.append(f"{label}: invalid status={status!r}")
            if item_priority not in ALLOWED_PRIORITIES:
                errors.append(f"{label}: invalid priority={item_priority!r}")
            if not source_url:
                errors.append(f"{label}: source_url is required")
            if not license_status:
                errors.append(f"{label}: license_status is required")
            if not adapter:
                errors.append(f"{label}: adapter is required")
            if not origin:
                errors.append(f"{label}: origin is required")
            if not paper_title:
                errors.append(f"{label}: paper_title is required")
            if not venue:
                errors.append(f"{label}: venue is required")
            if not isinstance(publication_year, int) or not 1900 <= publication_year <= 2100:
                errors.append(f"{label}: publication_year must be an integer year")
            if not publication_date:
                errors.append(f"{label}: publication_date is required")
            if not paper_url:
                errors.append(f"{label}: paper_url is required")
            if status in READY_STATUSES and not source_commit:
                errors.append(f"{label}: ready baseline needs source_commit")
            if status in {"validation_complete", "frozen_test_complete"} and license_status in {
                "review_required",
                "research_only_review",
                "no_license_detected",
            }:
                warnings.append(f"{label}: result exists but distribution license is unresolved")
            if validation_evidence:
                evidence_path = Path(str(validation_evidence))
                if not evidence_path.is_absolute():
                    evidence_path = Path.cwd() / evidence_path
                if not evidence_path.is_file():
                    errors.append(
                        f"{label}: validation_evidence does not exist: "
                        f"{validation_evidence}"
                    )
            rows.append(
                {
                    "suite": suite_id,
                    "dataset": suite.get("dataset"),
                    "scope": scope,
                    "priority": item_priority,
                    "id": baseline_id,
                    "display_name": item.get("display_name"),
                    "status": status,
                    "adapter": adapter,
                    "origin": origin,
                    "paper_title": paper_title,
                    "venue": venue,
                    "publication_year": publication_year,
                    "publication_date": publication_date,
                    "paper_url": paper_url,
                    "source_commit": source_commit,
                    "license_status": license_status,
                    "code_availability": reproducibility.get("code", "unrecorded"),
                    "weights_availability": reproducibility.get("weights", "unrecorded"),
                    "execution_mode": reproducibility.get("execution", "unrecorded"),
                    "validation_evidence": validation_evidence,
                }
            )

    duplicated_global = sorted(
        key for key, count in Counter(global_ids).items() if count > 1
    )
    if duplicated_global:
        errors.append(f"duplicate fully qualified baseline ids: {duplicated_global}")

    counts = Counter(row["status"] for row in rows)
    critical = [row for row in rows if row["priority"] == "critical"]
    validation_ready = [
        row for row in critical if row["status"] in FROZEN_TEST_READY_STATUSES
    ]
    frozen = [row for row in critical if row["status"] == "frozen_test_complete"]
    return {
        "schema_version": "activemap-external-baseline-audit-v1",
        "registry": str(path),
        "protocol_valid": not errors,
        "ready_for_frozen_test": (
            bool(critical) and len(validation_ready) == len(critical) and not errors
        ),
        "frozen_test_complete": (
            bool(critical) and len(frozen) == len(critical) and not errors
        ),
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "suites": len(suites),
            "baselines": len(rows),
            "critical": len(critical),
            "critical_validation_ready": len(validation_ready),
            "critical_frozen_test_complete": len(frozen),
            "status_counts": dict(sorted(counts.items())),
        },
        "baselines": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# External Baseline Progress",
        "",
        f"Protocol valid: `{str(report['protocol_valid']).lower()}`  ",
        f"Ready for frozen test: `{str(report['ready_for_frozen_test']).lower()}`  ",
        f"Frozen test complete: `{str(report['frozen_test_complete']).lower()}`",
        "",
        "| Suite | Baseline | Year | Scope | Code | Weights | Execution | Status | License |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["baselines"]:
        lines.append(
            f"| {row['suite']} | {row['display_name']} | {row['publication_year']} | "
            f"{row['scope']} | {row['code_availability']} | "
            f"{row['weights_availability']} | {row['execution_mode']} | "
            f"{row['status']} | "
            f"{row['license_status']} |"
        )
    if report["errors"]:
        lines.extend(["", "## Errors", ""] + [f"- {item}" for item in report["errors"]])
    if report["warnings"]:
        lines.extend(
            ["", "## Warnings", ""] + [f"- {item}" for item in report["warnings"]]
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--require-frozen-complete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_registry(args.registry)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    if not report["protocol_valid"]:
        return 1
    if args.require_frozen_complete and not report["frozen_test_complete"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
