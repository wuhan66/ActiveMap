#!/usr/bin/env python3
"""Finalize immutable MUNO21 test rollouts into paper result bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.build_paired_rollout_report import build_report
from scripts.build_paper_rollout_bundle import build_rollout_bundle
from scripts.build_paper_writeback_bundle import build_writeback_bundle
from scripts.render_paired_rollout_reports import render as render_paired_reports

SEEDS = ("20260821", "20260822", "20260823")

ROLLOUT_EXPERIMENTS = {
    "generic_selector": {
        "rollout": "generic_selector.jsonl",
        "family": "selector",
    },
    "edit_conditioned_selector": {
        "rollout": "edit_conditioned_selector.jsonl",
        "family": "selector",
    },
    "agent_without_tools": {
        "rollout": "qwen3_4b_sft.jsonl",
        "validity": "llm_calls_qwen3_4b_sft.jsonl",
        "family": "agent",
    },
    "agent_forced_tools": {
        "rollout": "forced_tools.jsonl",
        "validity": "llm_calls_forced_tools.jsonl",
        "family": "agent",
    },
    "agent_without_recurrent_belief": {
        "rollout": "qwen3_4b_sft_tools_no_belief.jsonl",
        "validity": "llm_calls_qwen3_4b_sft_tools_no_belief.jsonl",
        "family": "agent",
    },
    "agent_tool_to_belief": {
        "rollout": "qwen3_4b_sft_tool_to_belief.jsonl",
        "validity": "llm_calls_qwen3_4b_sft_tool_to_belief.jsonl",
        "family": "agent",
    },
}

WRITEBACK_VARIANTS = (
    "generic_selector",
    "edit_conditioned_selector",
    "agent_tool_to_belief",
)

HEURISTIC_VARIANTS = (
    "random",
    "cheapest",
    "quality_first",
    "uncertainty",
    "mapex",
    "greedy_utility",
)


def _require_complete_ledger(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("returncode") != 0:
        raise ValueError("frozen test ledger is not complete")
    return payload


def _seed_paths(root: Path, filename: str) -> dict[str, Path]:
    paths = {seed: root / f"seed{seed}" / filename for seed in SEEDS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen test inputs: {missing}")
    return paths


def _deterministic_path(root: Path, filename: str) -> dict[str, Path]:
    path = root / f"seed{SEEDS[0]}" / filename
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen test input: {path}")
    return {"deterministic": path}


def _write_bundle(path: Path, bundle: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")


def _comparison_paths(
    rollout_root: Path,
    experiment_id: str,
    variant: str | None,
) -> dict[str, Path]:
    if experiment_id == "heuristic_policy_suite":
        if variant not in HEURISTIC_VARIANTS:
            raise ValueError(f"unknown heuristic comparison variant: {variant}")
        return _deterministic_path(rollout_root, f"{variant}.jsonl")
    specification = ROLLOUT_EXPERIMENTS.get(experiment_id)
    if specification is None:
        raise ValueError(f"comparison has no rollout mapping: {experiment_id}")
    return _seed_paths(rollout_root, str(specification["rollout"]))


def _is_muno21_comparison(comparison: dict[str, Any]) -> bool:
    supported = set(ROLLOUT_EXPERIMENTS) | {"heuristic_policy_suite"}
    return (
        str(comparison.get("baseline_experiment")) in supported
        and str(comparison.get("candidate_experiment")) in supported
        and comparison.get("test_policy") == "frozen_once"
    )


def finalize(
    registry_path: Path,
    ledger_path: Path,
    rollout_root: Path,
    writeback_root: Path,
    official_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    _require_complete_ledger(ledger_path)
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    budgets = tuple(map(float, registry["protocol"]["muno21_budgets"]))
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse paper result root: {output_root}")

    generated: list[Path] = []
    oracle_paths = _seed_paths(rollout_root, "oracle.jsonl")
    for experiment_id, specification in ROLLOUT_EXPERIMENTS.items():
        rollout_paths = _seed_paths(rollout_root, str(specification["rollout"]))
        validity_name = specification.get("validity")
        validity_paths = (
            _seed_paths(rollout_root, str(validity_name))
            if validity_name is not None
            else None
        )
        for budget in budgets:
            bundle = build_rollout_bundle(
                registry_path,
                rollout_paths,
                experiment_id=experiment_id,
                variant=None,
                budget=budget,
                seed_validity_paths=validity_paths,
                seed_oracle_paths=(
                    oracle_paths if specification["family"] == "selector" else None
                ),
                frozen_test_ledger=ledger_path,
            )
            destination = output_root / "rollout" / experiment_id / f"budget-{budget:g}.json"
            _write_bundle(destination, bundle)
            generated.append(destination)
        curve = build_rollout_bundle(
            registry_path,
            rollout_paths,
            experiment_id=experiment_id,
            variant=None,
            budget=None,
            frozen_test_ledger=ledger_path,
        )
        destination = output_root / "rollout" / experiment_id / "quality_cost_auc.json"
        _write_bundle(destination, curve)
        generated.append(destination)

    deterministic_oracle = _deterministic_path(rollout_root, "oracle.jsonl")
    for variant in HEURISTIC_VARIANTS:
        rollout_paths = _deterministic_path(rollout_root, f"{variant}.jsonl")
        for budget in budgets:
            bundle = build_rollout_bundle(
                registry_path,
                rollout_paths,
                experiment_id="heuristic_policy_suite",
                variant=variant,
                budget=budget,
                seed_oracle_paths=deterministic_oracle,
                frozen_test_ledger=ledger_path,
            )
            destination = (
                output_root
                / "rollout"
                / "heuristic_policy_suite"
                / variant
                / f"budget-{budget:g}.json"
            )
            _write_bundle(destination, bundle)
            generated.append(destination)
        curve = build_rollout_bundle(
            registry_path,
            rollout_paths,
            experiment_id="heuristic_policy_suite",
            variant=variant,
            budget=None,
            frozen_test_ledger=ledger_path,
        )
        destination = (
            output_root
            / "rollout"
            / "heuristic_policy_suite"
            / variant
            / "quality_cost_auc.json"
        )
        _write_bundle(destination, curve)
        generated.append(destination)

    for variant in WRITEBACK_VARIANTS:
        seed_writeback = _seed_paths(writeback_root, f"{variant}/writeback.jsonl")
        seed_official = _seed_paths(official_root, f"{variant}/official_metrics.jsonl")
        for budget in budgets:
            bundle = build_writeback_bundle(
                registry_path,
                seed_writeback,
                seed_official,
                variant=variant,
                budget=budget,
                frozen_test_ledger=ledger_path,
            )
            destination = output_root / "writeback" / variant / f"budget-{budget:g}.json"
            _write_bundle(destination, bundle)
            generated.append(destination)

    paired_reports: list[Path] = []
    for comparison in registry["required_paired_comparisons"]:
        if not _is_muno21_comparison(comparison):
            continue
        report = build_report(
            registry_path,
            _comparison_paths(
                rollout_root,
                str(comparison["baseline_experiment"]),
                comparison.get("baseline_variant"),
            ),
            _comparison_paths(
                rollout_root,
                str(comparison["candidate_experiment"]),
                comparison.get("candidate_variant"),
            ),
            comparison_id=str(comparison["id"]),
            frozen_test_ledger=ledger_path,
        )
        destination = output_root / "paired_comparisons" / f"{comparison['id']}.json"
        _write_bundle(destination, report)
        paired_reports.append(destination)
    paired_table_manifest = render_paired_reports(
        paired_reports,
        output_root / "paired_tables",
    )

    manifest = {
        "schema_version": "activemap-muno21-paper-finalization-v1",
        "ledger": str(ledger_path.resolve()),
        "seeds": list(SEEDS),
        "budgets": list(budgets),
        "rollout_experiments": list(ROLLOUT_EXPERIMENTS),
        "heuristic_variants": list(HEURISTIC_VARIANTS),
        "writeback_variants": list(WRITEBACK_VARIANTS),
        "bundle_count": len(generated),
        "bundles": [str(path.relative_to(output_root)) for path in generated],
        "paired_report_count": len(paired_reports),
        "paired_reports": [
            str(path.relative_to(output_root)) for path in paired_reports
        ],
        "paired_tables": paired_table_manifest,
    }
    manifest_path = output_root / "manifest.json"
    _write_bundle(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("rollout_root", type=Path)
    parser.add_argument("writeback_root", type=Path)
    parser.add_argument("official_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    manifest = finalize(
        args.registry,
        args.ledger,
        args.rollout_root,
        args.writeback_root,
        args.official_root,
        args.output_root,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
