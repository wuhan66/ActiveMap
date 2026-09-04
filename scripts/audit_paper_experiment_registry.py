#!/usr/bin/env python3
"""Audit the frozen paper experiment registry and current server artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ALLOWED_TEST_POLICIES = {"validation_only", "frozen_once"}
REQUIRED_FAMILIES = {"updater", "selector", "agent", "rl", "writeback"}
REQUIRED_ABLATION_CLAIMS = {
    "no_prior_vector",
    "no_edit_classifier",
    "shared_vs_edit_specific_geometry",
    "no_confidence_gate",
    "no_inria_pretraining",
    "no_stopping",
    "no_agent_tool_loop",
}
PAIRED_GATES = {
    "primary_utility_noninferior",
    "quality_cost_auc_noninferior",
    "false_edit_delta_noninferior",
    "absolute_false_edit_safe",
    "matched_cost",
    "quality_cost_safety_non_dominated",
}


def _nested_value(payload: dict[str, Any], dotted: str) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted)
        value = value[part]
    return value


def audit_registry(registry_path: Path, storage_root: Path) -> dict[str, Any]:
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []
    artifact_results: list[dict[str, Any]] = []

    experiments = registry.get("experiments", [])
    experiment_ids = [str(item.get("id", "")) for item in experiments]
    experiment_by_id = {str(item.get("id", "")): item for item in experiments}
    duplicate_experiments = sorted(
        key for key, count in Counter(experiment_ids).items() if count > 1
    )
    if duplicate_experiments:
        errors.append(f"duplicate experiment ids: {duplicate_experiments}")

    ablation_claims = registry.get("required_ablation_claims", [])
    claim_ids = [str(item.get("id", "")) for item in ablation_claims]
    duplicate_claims = sorted(key for key, count in Counter(claim_ids).items() if count > 1)
    if duplicate_claims:
        errors.append(f"duplicate ablation claim ids: {duplicate_claims}")
    missing_claims = sorted(REQUIRED_ABLATION_CLAIMS - set(claim_ids))
    if registry.get("paper_target") and missing_claims:
        errors.append(f"missing required ablation claims: {missing_claims}")
    for claim in ablation_claims:
        claim_id = str(claim.get("id", "<missing>"))
        experiment_id = str(claim.get("experiment", ""))
        experiment = experiment_by_id.get(experiment_id)
        if experiment is None:
            errors.append(f"{claim_id}: unknown ablation experiment {experiment_id!r}")
            continue
        variant = claim.get("variant")
        if variant is not None and str(variant) not in set(map(str, experiment.get("methods", []))):
            errors.append(
                f"{claim_id}: variant {variant!r} is not registered by {experiment_id}"
            )
        reference = claim.get("reference")
        if reference is not None and str(reference) not in experiment_by_id:
            errors.append(f"{claim_id}: unknown reference experiment {reference!r}")

    paired_comparisons = registry.get("required_paired_comparisons", [])
    paired_ids = [str(item.get("id", "")) for item in paired_comparisons]
    duplicate_paired = sorted(
        key for key, count in Counter(paired_ids).items() if count > 1
    )
    if duplicate_paired:
        errors.append(f"duplicate paired comparison ids: {duplicate_paired}")
    if paired_comparisons and not registry.get("protocol", {}).get(
        "paired_comparison_margins"
    ):
        errors.append("paired comparisons require predeclared protocol margins")
    for comparison in paired_comparisons:
        comparison_id = str(comparison.get("id", "<missing>"))
        for side in ("baseline", "candidate"):
            experiment_id = str(comparison.get(f"{side}_experiment", ""))
            experiment = experiment_by_id.get(experiment_id)
            if experiment is None:
                errors.append(
                    f"{comparison_id}: unknown {side} experiment {experiment_id!r}"
                )
                continue
            variant = comparison.get(f"{side}_variant")
            methods = set(map(str, experiment.get("methods", [])))
            if variant is not None and str(variant) not in methods:
                errors.append(
                    f"{comparison_id}: {side} variant {variant!r} is not registered "
                    f"by {experiment_id}"
                )
        policy = comparison.get("test_policy")
        if policy not in ALLOWED_TEST_POLICIES:
            errors.append(f"{comparison_id}: invalid paired test_policy={policy!r}")
        required_gates = set(map(str, comparison.get("required_gates", [])))
        if not required_gates:
            errors.append(f"{comparison_id}: required_gates must be non-empty")
        unknown_gates = sorted(required_gates - PAIRED_GATES)
        if unknown_gates:
            errors.append(f"{comparison_id}: unknown paired gates {unknown_gates}")

    artifact_specs = registry.get("required_artifacts", [])
    artifact_ids = [str(item.get("id", "")) for item in artifact_specs]
    duplicate_artifacts = sorted(
        key for key, count in Counter(artifact_ids).items() if count > 1
    )
    if duplicate_artifacts:
        errors.append(f"duplicate artifact ids: {duplicate_artifacts}")
    artifact_id_set = set(artifact_ids)

    repo_root = registry_path.resolve().parents[2]
    families = {str(item.get("family", "")) for item in experiments}
    missing_families = sorted(REQUIRED_FAMILIES - families)
    if missing_families:
        errors.append(f"missing required experiment families: {missing_families}")

    for experiment in experiments:
        experiment_id = str(experiment.get("id", "<missing>"))
        family = str(experiment.get("family", ""))
        policy = experiment.get("test_policy")
        if policy not in ALLOWED_TEST_POLICIES:
            errors.append(f"{experiment_id}: invalid test_policy={policy!r}")
        required = set(map(str, experiment.get("requires", [])))
        unknown = sorted(required - artifact_id_set)
        if unknown:
            errors.append(f"{experiment_id}: unknown required artifacts {unknown}")
        dependencies = set(map(str, experiment.get("depends_on", [])))
        unknown_dependencies = sorted(dependencies - set(experiment_ids))
        if unknown_dependencies:
            errors.append(f"{experiment_id}: unknown dependencies {unknown_dependencies}")
        config = experiment.get("config")
        config_path = repo_root / str(config) if config else None
        if config_path is not None and not config_path.is_file():
            errors.append(f"{experiment_id}: missing config {config}")
        matrix_variant = experiment.get("matrix_variant")
        if matrix_variant is not None and config_path is not None and config_path.is_file():
            matrix = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            registered_variants = {
                str(row.get("name")) for row in matrix.get("experiments", [])
            }
            if str(matrix_variant) not in registered_variants:
                errors.append(
                    f"{experiment_id}: matrix_variant {matrix_variant!r} is not in {config}"
                )
        seeds = experiment.get("seeds", [])
        if experiment.get("trainable") and experiment.get("role") == "primary":
            if len(seeds) < 3:
                errors.append(f"{experiment_id}: primary trainable experiment needs >=3 seeds")
        metric_family = "agent" if family == "rl" else family
        default_required = registry.get("required_metrics", {}).get(metric_family, [])
        default_primary = registry.get("primary_metrics", {}).get(family, [])
        required_metrics = list(map(str, experiment.get("required_metrics", default_required)))
        primary_metrics = list(map(str, experiment.get("primary_metrics", default_primary)))
        if registry.get("paper_target") or registry.get("required_metrics"):
            if not required_metrics or not primary_metrics:
                errors.append(
                    f"{experiment_id}: required_metrics and primary_metrics must be non-empty"
                )
            unknown_primary = sorted(set(primary_metrics) - set(required_metrics))
            if unknown_primary:
                errors.append(
                    f"{experiment_id}: primary metrics are not required: {unknown_primary}"
                )
        if experiment.get("budgets"):
            default_curve = registry.get("curve_metrics", {}).get(metric_family, [])
            default_curve_primary = registry.get("curve_primary_metrics", {}).get(family, [])
            curve_metrics = list(map(str, experiment.get("curve_metrics", default_curve)))
            curve_primary = list(
                map(str, experiment.get("curve_primary_metrics", default_curve_primary))
            )
            if registry.get("paper_target") or registry.get("curve_metrics"):
                if not curve_metrics or not curve_primary:
                    errors.append(
                        f"{experiment_id}: curve_metrics and curve_primary_metrics "
                        "must be non-empty for budgeted experiments"
                    )
                unknown_curve_primary = sorted(set(curve_primary) - set(curve_metrics))
                if unknown_curve_primary:
                    errors.append(
                        f"{experiment_id}: curve primary metrics are not curve metrics: "
                        f"{unknown_curve_primary}"
                    )
        if family == "agent" and len(seeds) >= 3:
            if experiment.get("seed_semantics") != "model_training":
                errors.append(
                    f"{experiment_id}: three-seed Agent results must use model_training seeds"
                )
        if family == "writeback" and len(seeds) >= 3:
            if experiment.get("seed_semantics") != "upstream_model_training":
                errors.append(
                    f"{experiment_id}: writeback seeds must track upstream model training"
                )

    for spec in artifact_specs:
        artifact_id = str(spec["id"])
        rendered = str(spec["path"]).replace("${STORAGE_ROOT}", str(storage_root))
        path = Path(rendered)
        kind = str(spec.get("kind", "file"))
        result: dict[str, Any] = {
            "id": artifact_id,
            "stage": spec.get("stage"),
            "path": str(path),
            "kind": kind,
            "ready": False,
            "reason": "missing",
        }
        if kind == "file":
            result["ready"] = path.is_file() and path.stat().st_size > 0
            result["reason"] = "ok" if result["ready"] else "missing_or_empty"
        elif kind == "dir":
            result["ready"] = path.is_dir()
            result["reason"] = "ok" if result["ready"] else "missing"
        elif kind == "json_gate":
            if path.is_file() and path.stat().st_size > 0:
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    actual = _nested_value(payload, str(spec["field"]))
                    expected = spec.get("equals")
                    result["actual"] = actual
                    result["expected"] = expected
                    result["ready"] = actual == expected
                    result["reason"] = "ok" if result["ready"] else "gate_failed"
                except (json.JSONDecodeError, KeyError) as exc:
                    result["reason"] = f"invalid_gate:{exc}"
            else:
                result["reason"] = "missing_or_empty"
        else:
            errors.append(f"{artifact_id}: unsupported kind {kind!r}")
            result["reason"] = "unsupported_kind"
        artifact_results.append(result)

    ready_artifacts = {item["id"] for item in artifact_results if item["ready"]}
    experiment_results: list[dict[str, Any]] = []
    for experiment in experiments:
        required = set(map(str, experiment.get("requires", [])))
        missing = sorted(required - ready_artifacts)
        experiment_results.append(
            {
                "id": experiment["id"],
                "family": experiment["family"],
                "role": experiment["role"],
                "ready": not missing,
                "blocking_artifacts": missing,
                "test_policy": experiment["test_policy"],
            }
        )

    if registry.get("protocol", {}).get("test_policy") != "frozen_once":
        errors.append("global test policy must be frozen_once")
    if any(item["test_policy"] == "frozen_once" for item in experiment_results):
        approval = next(
            (item for item in artifact_results if item["id"] == "manual_qc_approval"), None
        )
        if approval is None:
            errors.append("frozen_once experiments require manual_qc_approval artifact")
        elif not approval["ready"]:
            warnings.append("frozen test execution remains blocked until manual QC approval")

    return {
        "schema_version": "paper-registry-audit-v1",
        "registry": str(registry_path),
        "storage_root": str(storage_root),
        "protocol_valid": not errors,
        "ready_for_frozen_test": not errors
        and all(item["ready"] for item in artifact_results),
        "errors": errors,
        "warnings": warnings,
        "artifact_summary": {
            "ready": sum(bool(item["ready"]) for item in artifact_results),
            "total": len(artifact_results),
        },
        "experiment_summary": {
            "ready": sum(bool(item["ready"]) for item in experiment_results),
            "total": len(experiment_results),
        },
        "artifacts": artifact_results,
        "experiments": experiment_results,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Paper Experiment Preflight",
        "",
        f"- Protocol valid: `{str(report['protocol_valid']).lower()}`",
        f"- Ready for frozen test: `{str(report['ready_for_frozen_test']).lower()}`",
        "- Artifacts ready: "
        f"`{report['artifact_summary']['ready']}/{report['artifact_summary']['total']}`",
        "- Experiments unblocked: "
        f"`{report['experiment_summary']['ready']}/{report['experiment_summary']['total']}`",
        "",
        "## Artifact Gates",
        "",
        "| Artifact | Stage | Ready | Reason | Path |",
        "|---|---|---:|---|---|",
    ]
    for item in report["artifacts"]:
        lines.append(
            f"| {item['id']} | {item['stage']} | {str(item['ready']).lower()} | "
            f"{item['reason']} | `{item['path']}` |"
        )
    lines.extend(
        [
            "",
            "## Experiment Readiness",
            "",
            "| Experiment | Family | Role | Ready | Blocking artifacts |",
            "|---|---|---|---:|---|",
        ]
    )
    for item in report["experiments"]:
        blockers = ", ".join(item["blocking_artifacts"]) or "-"
        lines.append(
            f"| {item['id']} | {item['family']} | {item['role']} | "
            f"{str(item['ready']).lower()} | {blockers} |"
        )
    if report["errors"]:
        lines.extend(["", "## Protocol Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("storage_root", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args()
    report = audit_registry(args.registry, args.storage_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["artifact_summary"]))
    print(json.dumps(report["experiment_summary"]))
    if not report["protocol_valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
