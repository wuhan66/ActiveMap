#!/usr/bin/env python3
"""Build a reviewer-facing evidence bundle from immutable aggregate receipts.

The script never opens raw imagery, policy traces, or frozen-test examples. It
only joins precomputed aggregate receipts so the resulting tables and figure
can be regenerated without re-running evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_R1_R2 = (
    ROOT
    / "artifacts"
    / "paper_evidence"
    / "iclr2027_r1_r2_validation_20260813"
    / "three_seed_summary.json"
)
DEFAULT_CAUSAL = (
    ROOT
    / "artifacts"
    / "paper_evidence"
    / "reviewer_closure_receipts_20260830"
    / "selection_safe_commit_2x2"
    / "summary.json"
)
DEFAULT_COST = (
    ROOT
    / "artifacts"
    / "paper_evidence"
    / "reviewer_closure_receipts_20260830"
    / "active_catalog_cost_accounting_v3"
    / "summary.json"
)
DEFAULT_OPERATIONS = (
    ROOT
    / "artifacts"
    / "paper_evidence"
    / "reviewer_closure_receipts_20260830"
    / "nonkeep_operation_slices_v1.json"
)
DEFAULT_LEARNED_DEFER = (
    ROOT
    / "artifacts"
    / "paper_evidence"
    / "reviewer_closure_receipts_20260830"
    / "learned_defer_extension"
    / "three_seed_summary.json"
)
DEFAULT_OUTPUT = ROOT / "docs" / "figures" / "reviewer_closure_evidence_20260830"

RATE_MATCHED_PREFIX = "activemap_minus_rate_matched_"
POLICY_LABELS = {
    "always_stop": "Always STOP",
    "forced": "Forced verification",
    "cheap_positive": "Cheap positive",
    "clear_per_cost": "Clear per cost",
    "low_confidence": "Low confidence",
    "minimum_entropy": "Minimum entropy",
    "random": "Random",
    "uncertainty": "Uncertainty",
    "learned_defer": "Architecture-matched learned defer",
}
CAUSAL_LABELS = {
    "selection_gain_always_commit": "Selection | Always Commit",
    "selection_gain_safe_commit": "Selection | Safe Commit",
    "safe_commit_gain_no_extra": "Safe Commit | no extra evidence",
    "safe_commit_gain_learned": "Safe Commit | learned selection",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def validate_receipts(
    r1_r2: dict[str, Any],
    causal: dict[str, Any],
    cost: dict[str, Any],
    operations: dict[str, Any],
    learned_defer: dict[str, Any] | None = None,
) -> None:
    expected = (
        (r1_r2, "activemap-r1-r2-validation-three-seed-v1", "val", False),
        (causal, "activemap-selection-safe-commit-2x2-v1", "val", False),
        (cost, "activemap-active-catalog-cost-accounting-v1", "val", False),
    )
    for payload, schema, split, test_read in expected:
        if payload.get("schema_version") != schema:
            raise ValueError(f"unexpected schema: {payload.get('schema_version')}")
        if payload.get("split") != split or payload.get("test_assets_read") is not test_read:
            raise ValueError(f"invalid split contract for {schema}")

    if operations.get("schema_version") != "agent-writeback-operation-stratified-v1":
        raise ValueError("unexpected operation-slice schema")
    if operations.get("split") != "val" or operations.get("test_assets_read") is not False:
        raise ValueError("operation slices must remain validation-only")
    if int(operations.get("seed_count", 0)) != 3:
        raise ValueError("operation slices require exactly three model seeds")
    if set(operations.get("operations", {})) != {"KEEP", "ADD", "DELETE", "RESHAPE"}:
        raise ValueError("operation slices do not cover all typed operations")

    variants = set(r1_r2.get("variants", []))
    required = {
        "activemap",
        "always_stop",
        "forced",
        "rate_matched_random",
        "rate_matched_uncertainty",
        "rate_matched_low_confidence",
        "rate_matched_clear_per_cost",
        "rate_matched_cheap_positive",
        "rate_matched_minimum_entropy",
    }
    if not required.issubset(variants):
        raise ValueError("R1/R2 receipt is missing required matched controls")
    if learned_defer is not None:
        if (
            learned_defer.get("schema_version")
            != "activemap-sn7-learned-defer-extension-three-seed-v1"
        ):
            raise ValueError("unexpected learned-defer schema")
        if (
            learned_defer.get("split") != "val"
            or learned_defer.get("test_assets_read") is not False
        ):
            raise ValueError("learned-defer evidence must remain validation-only")
        if set(learned_defer.get("seeds", [])) != {20260730, 20260731, 20260801}:
            raise ValueError("learned-defer evidence requires the matched three seeds")


def interval_row(
    analysis: str,
    comparison: str,
    metric: str,
    interval: dict[str, Any],
) -> dict[str, Any]:
    return {
        "analysis": analysis,
        "comparison": comparison,
        "metric": metric,
        "delta": float(interval["observed_delta"]),
        "ci95_low": float(interval["ci95_low"]),
        "ci95_high": float(interval["ci95_high"]),
    }


def matched_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for analysis, comparisons in payload["results"].items():
        for comparison, result in comparisons.items():
            baseline = comparison.removeprefix("activemap_minus_")
            for metric, interval in result["intervals"].items():
                row = interval_row(analysis, baseline, metric, interval)
                row["controller_seed_count"] = int(result["controller_seed_count"])
                rows.append(row)
    return rows


def learned_defer_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for analysis, block in payload["results"].items():
        result = block["activemap_minus_learned_defer"]
        for metric, interval in result["intervals"].items():
            row = interval_row(analysis, "learned_defer", metric, interval)
            row["controller_seed_count"] = int(result["controller_seed_count"])
            rows.append(row)
    return rows


def causal_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for comparison, metrics in payload["paired_comparisons"].items():
        for metric, interval in metrics.items():
            row = interval_row("selection_x_safe_commit", comparison, metric, interval)
            row["controller_seed_count"] = int(payload["seed_count"])
            row["aoi_count"] = int(payload["aoi_count"])
            rows.append(row)
    return rows


def causal_cell_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for cell, metrics in payload["cells"].items():
        selection, commit = cell.split("__", maxsplit=1)
        row: dict[str, Any] = {"selection": selection, "commit": commit}
        for metric, value in metrics.items():
            row[metric] = float(value["mean"])
            row[f"{metric}_seed_std"] = float(value["seed_std"])
        rows.append(row)
    return rows


def cost_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for policy, metrics in payload["policies"].items():
        row: dict[str, Any] = {"policy": policy}
        for metric, value in metrics.items():
            row[metric] = float(value["mean"])
            row[f"{metric}_seed_std"] = float(value["seed_std"])
        rows.append(row)
    return rows


def operation_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for operation, result in payload["operations"].items():
        interval = result["candidate_minus_baseline"]["raster_iou"]
        record_count = int(result["record_count"])
        seed_count = int(payload["seed_count"])
        rows.append(
            {
                "operation": operation,
                "record_count": record_count,
                "support_per_seed": record_count // seed_count,
                "seed_aoi_block_count": int(result["seed_aoi_block_count"]),
                "map_quality_delta": float(interval["observed_delta"]),
                "ci95_low": float(interval["ci95_low"]),
                "ci95_high": float(interval["ci95_high"]),
            }
        )
    return rows


def derive_summary(
    matched: list[dict[str, Any]],
    causal: list[dict[str, Any]],
    costs: list[dict[str, Any]],
    operations: list[dict[str, Any]],
) -> dict[str, Any]:
    cost_by_policy = {row["policy"]: row for row in costs}
    benefit = cost_by_policy["benefit"]
    forced = cost_by_policy["forced"]
    tool_reduction = 1.0 - (
        benefit["incremental_tool_calls"] / forced["incremental_tool_calls"]
    )
    budget_reduction = 1.0 - (
        benefit["incremental_total_budget"] / forced["incremental_total_budget"]
    )
    shared_times = [row["shared_preacquisition_perception_ms"] for row in costs]

    rate_rows = [
        row
        for row in matched
        if row["analysis"] == "r1_natural"
        and row["comparison"].startswith("rate_matched_")
    ]
    rate_signatures: dict[str, list[tuple[float, float, float]]] = {}
    for row in rate_rows:
        rate_signatures.setdefault(row["metric"], []).append(
            (row["delta"], row["ci95_low"], row["ci95_high"])
        )
    rate_matched_identical = all(
        len(set(signatures)) == 1 for signatures in rate_signatures.values()
    )

    selection_safe = next(
        row
        for row in causal
        if row["comparison"] == "selection_gain_safe_commit"
        and row["metric"] == "map_quality_after"
    )
    nonkeep = [row for row in operations if row["operation"] != "KEEP"]
    return {
        "incremental_tool_call_reduction_vs_forced": tool_reduction,
        "incremental_budget_reduction_vs_forced": budget_reduction,
        "shared_preacquisition_perception_ms": shared_times[0],
        "shared_perception_identical_across_policies": max(shared_times)
        - min(shared_times)
        < 1e-12,
        "rate_matched_controls_terminally_identical": rate_matched_identical,
        "selection_effect_survives_safe_commit": selection_safe["ci95_low"] > 0.0,
        "nonkeep_map_quality_effect_is_zero": all(
            abs(row["map_quality_delta"]) < 1e-12
            and abs(row["ci95_low"]) < 1e-12
            and abs(row["ci95_high"]) < 1e-12
            for row in nonkeep
        ),
    }


def build_bundle(
    source_paths: dict[str, Path],
) -> dict[str, Any]:
    payloads = {name: load_json(path) for name, path in source_paths.items()}
    validate_receipts(
        payloads["r1_r2"],
        payloads["causal_2x2"],
        payloads["cost"],
        payloads["operations"],
        payloads.get("learned_defer"),
    )
    matched = matched_rows(payloads["r1_r2"])
    if "learned_defer" in payloads:
        matched.extend(learned_defer_rows(payloads["learned_defer"]))
    causal = causal_rows(payloads["causal_2x2"])
    causal_cells = causal_cell_rows(payloads["causal_2x2"])
    costs = cost_rows(payloads["cost"])
    operations = operation_rows(payloads["operations"])
    return {
        "schema_version": "activemap-reviewer-closure-evidence-bundle-v1",
        "evidence_scope": {
            "primary_split": "validation",
            "raw_test_assets_read": False,
            "frozen_test_input": "none",
            "claim": (
                "Benefit-aware post-perception selection has an independent "
                "quality-safety effect and reduces incremental verification work."
            ),
            "boundary": (
                "The current SN7 evidence does not establish non-KEEP recovery, "
                "persistent maintenance, or lower visual-backbone compute."
            ),
        },
        "sources": {name: source_record(path) for name, path in source_paths.items()},
        "matched_baselines": matched,
        "selection_safe_commit_cells": causal_cells,
        "selection_safe_commit_2x2": causal,
        "cost_accounting": costs,
        "operation_slices": operations,
        "derived": derive_summary(matched, causal, costs, operations),
        "excluded_noncomparable_baselines": [
            {
                "name": "legacy SN7 generic selector",
                "reason": (
                    "one-seed executable-selector-v2 result under an older protocol; "
                    "superseded by the three-seed architecture-matched learned-defer "
                    "extension included in this bundle"
                ),
            }
        ],
    }


def source_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        location = str(resolved.relative_to(ROOT))
    except ValueError:
        location = str(resolved)
    return {"path": location, "sha256": sha256(path)}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_row(
    rows: list[dict[str, Any]],
    *,
    analysis: str,
    comparison: str,
    metric: str,
) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if row["analysis"] == analysis
        and row["comparison"] == comparison
        and row["metric"] == metric
    )


def format_interval(row: dict[str, Any], *, scale: float = 1.0) -> str:
    return (
        f"{scale * row['delta']:+.4f} "
        f"[{scale * row['ci95_low']:+.4f}, {scale * row['ci95_high']:+.4f}]"
    )


def render_markdown(bundle: dict[str, Any]) -> str:
    matched = bundle["matched_baselines"]
    causal = bundle["selection_safe_commit_2x2"]
    operations = bundle["operation_slices"]
    derived = bundle["derived"]
    stop_safety = find_row(
        matched,
        analysis="r1_natural",
        comparison="always_stop",
        metric="safety_utility",
    )
    forced_cost = find_row(
        matched,
        analysis="r1_natural",
        comparison="forced",
        metric="mean_cost",
    )
    selection_raw = find_row(
        causal,
        analysis="selection_x_safe_commit",
        comparison="selection_gain_always_commit",
        metric="map_quality_after",
    )
    selection_safe = find_row(
        causal,
        analysis="selection_x_safe_commit",
        comparison="selection_gain_safe_commit",
        metric="map_quality_after",
    )
    learned_safety = next(
        (
            row
            for row in matched
            if row["analysis"] == "r1_natural"
            and row["comparison"] == "learned_defer"
            and row["metric"] == "safety_utility"
        ),
        None,
    )
    lines = [
        "# Reviewer-Closure Evidence Bundle",
        "",
        "This bundle joins immutable aggregate receipts; it does not rerun evaluation or",
        "open raw frozen-test assets.",
        "",
        "## Main findings",
        "",
        f"- ActiveMap minus STOP safety utility: `{format_interval(stop_safety)}`.",
        f"- ActiveMap minus forced incremental cost: `{format_interval(forced_cost)}`.",
        "- Selection map-quality effect before Safe Commit: "
        f"`{format_interval(selection_raw)}`.",
        "- Selection map-quality effect after the identical Safe Commit gate: "
        f"`{format_interval(selection_safe)}`.",
        *(
            [
                "- ActiveMap minus architecture-matched learned defer safety utility: "
                f"`{format_interval(learned_safety)}` (95% CI crosses zero)."
            ]
            if learned_safety is not None
            else []
        ),
        "- Incremental tool-call reduction versus forced verification: "
        f"`{100.0 * derived['incremental_tool_call_reduction_vs_forced']:.1f}%`.",
        "- Incremental abstract-budget reduction versus forced verification: "
        f"`{100.0 * derived['incremental_budget_reduction_vs_forced']:.1f}%`.",
        "- Shared all-candidate updater work remains "
        f"`{derived['shared_preacquisition_perception_ms']:.1f} ms/episode` for every policy.",
        "",
        "## Operation scope",
        "",
        "| Operation | Records (3 seeds) | Map-quality delta [95% CI] |",
        "| --- | ---: | ---: |",
    ]
    for row in operations:
        lines.append(
            f"| {row['operation']} | {row['record_count']} | "
            f"`{row['map_quality_delta']:+.6f} "
            f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}]` |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            bundle["evidence_scope"]["boundary"],
            "The old one-seed SN7 generic selector remains excluded because its support and",
            "protocol are not matched to R1/R2. Its replacement is the included three-seed,",
            "architecture-matched learned-defer baseline. ActiveMap has positive natural-",
            "prevalence point estimates against this baseline, but the confidence intervals",
            "cross zero; edit conditioning is therefore not claimed as independently superior.",
            "",
        ]
    )
    return "\n".join(lines)


def latex_interval(row: dict[str, Any]) -> str:
    return (
        f"{row['delta']:+.4f} "
        f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}]"
    )


def render_latex(bundle: dict[str, Any]) -> str:
    matched = bundle["matched_baselines"]
    causal = bundle["selection_safe_commit_2x2"]
    causal_cells = bundle["selection_safe_commit_cells"]
    operations = bundle["operation_slices"]
    costs = {row["policy"]: row for row in bundle["cost_accounting"]}
    matched_specs = (
        ("Natural", "always_stop", "Always STOP"),
        ("Natural", "learned_defer", "Matched learned defer"),
        ("Natural", "rate_matched_random", "Rate-matched heuristics (6)"),
        ("Natural", "forced", "Forced verification"),
        ("EDIT-only", "always_stop", "Always STOP"),
        ("EDIT-only", "learned_defer", "Matched learned defer"),
    )
    lines = [
        "% Generated by scripts/figures/build_reviewer_closure_evidence.py",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Slice & Reference & $\\Delta$ safety utility & $\\Delta$ cost \\\\",
        "\\midrule",
    ]
    analysis_names = {"Natural": "r1_natural", "EDIT-only": "r2_edit_only"}
    for slice_name, comparison, label in matched_specs:
        analysis = analysis_names[slice_name]
        safety = find_row(
            matched,
            analysis=analysis,
            comparison=comparison,
            metric="safety_utility",
        )
        cost = find_row(
            matched,
            analysis=analysis,
            comparison=comparison,
            metric="mean_cost",
        )
        lines.append(
            f"{slice_name} & {label} & {latex_interval(safety)} & "
            f"{latex_interval(cost)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "", "\\medskip", ""])
    lines.extend(
        [
            "\\begin{tabular}{llrrrr}",
            "\\toprule",
            "Selection & Commit & Map quality & False edit & Missed edit & Commit rate \\\\",
            "\\midrule",
        ]
    )
    cell_order = (
        ("notool", "always_commit"),
        ("benefit", "always_commit"),
        ("notool", "safe_commit"),
        ("benefit", "safe_commit"),
    )
    cell_index = {(row["selection"], row["commit"]): row for row in causal_cells}
    for selection, commit in cell_order:
        row = cell_index[(selection, commit)]
        selection_label = "No extra" if selection == "notool" else "Learned"
        commit_label = "Always" if commit == "always_commit" else "Safe"
        lines.append(
            f"{selection_label} & {commit_label} & {row['map_quality_after']:.6f} & "
            f"{row['false_edit_rate']:.6f} & {row['missed_edit_rate']:.6f} & "
            f"{row['commit_rate']:.6f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "", "\\medskip", ""])
    lines.extend(
        [
            "\\begin{tabular}{lrrr}",
            "\\toprule",
            "Causal contrast & $\\Delta$ map quality & $\\Delta$ balanced utility & "
            "$\\Delta$ false edit \\\\",
            "\\midrule",
        ]
    )
    for comparison in ("selection_gain_always_commit", "selection_gain_safe_commit"):
        quality = find_row(
            causal,
            analysis="selection_x_safe_commit",
            comparison=comparison,
            metric="map_quality_after",
        )
        utility = find_row(
            causal,
            analysis="selection_x_safe_commit",
            comparison=comparison,
            metric="balanced_utility",
        )
        false_edit = find_row(
            causal,
            analysis="selection_x_safe_commit",
            comparison=comparison,
            metric="false_edit_rate",
        )
        lines.append(
            f"{CAUSAL_LABELS[comparison]} & {latex_interval(quality)} & "
            f"{latex_interval(utility)} & {latex_interval(false_edit)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "", "\\medskip", ""])
    lines.extend(
        [
            "\\begin{tabular}{lrrr}",
            "\\toprule",
            "Policy & Shared updater (ms) & Tool calls & Incremental budget \\\\",
            "\\midrule",
        ]
    )
    for policy in ("notool", "benefit", "forced"):
        row = costs[policy]
        label = {"notool": "No tool", "benefit": "ActiveMap", "forced": "Forced"}[policy]
        lines.append(
            f"{label} & {row['shared_preacquisition_perception_ms']:.3f} & "
            f"{row['incremental_tool_calls']:.6f} & "
            f"{row['incremental_total_budget']:.6f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "", "\\medskip", ""])
    lines.extend(
        [
            "\\begin{tabular}{lrr}",
            "\\toprule",
            "Operation & Records (3 seeds) & $\\Delta$ map quality [95\\% CI] \\\\",
            "\\midrule",
        ]
    )
    for row in operations:
        interval = (
            f"{row['map_quality_delta']:+.6f} "
            f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}]"
        )
        lines.append(f"{row['operation']} & {row['record_count']} & {interval} \\\\ ")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def errorbar(axis: Any, row: dict[str, Any], y: float, *, color: str, marker: str) -> None:
    value = row["delta"]
    axis.errorbar(
        value,
        y,
        xerr=[[value - row["ci95_low"]], [row["ci95_high"] - value]],
        color=color,
        marker=marker,
        markersize=5,
        capsize=2.5,
        linewidth=1.2,
        zorder=3,
    )


def render_figure(bundle: dict[str, Any], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "axes.titlesize": 8.0,
            "legend.fontsize": 6.4,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    blue = "#0072B2"
    green = "#009E73"
    orange = "#D55E00"
    gray = "#7A7A7A"
    light_gray = "#D9D9D9"

    figure = plt.figure(figsize=(7.16, 4.65))
    grid = figure.add_gridspec(
        2,
        2,
        left=0.12,
        right=0.985,
        bottom=0.11,
        top=0.93,
        hspace=0.54,
        wspace=0.42,
    )
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    cost_grid = grid[1, 0].subgridspec(1, 2, wspace=0.55)
    axis_c1 = figure.add_subplot(cost_grid[0, 0])
    axis_c2 = figure.add_subplot(cost_grid[0, 1])
    axis_d = figure.add_subplot(grid[1, 1])

    matched = bundle["matched_baselines"]
    matched_specs = (
        ("STOP", "r1_natural", "always_stop", gray),
        ("Learned defer", "r1_natural", "learned_defer", green),
        ("Matched heuristics (6)", "r1_natural", "rate_matched_random", gray),
        ("Forced", "r1_natural", "forced", gray),
        ("EDIT-only: STOP", "r2_edit_only", "always_stop", orange),
    )
    positions = list(reversed(range(len(matched_specs))))
    for y, (_, analysis, comparison, color) in zip(positions, matched_specs, strict=True):
        row = find_row(
            matched,
            analysis=analysis,
            comparison=comparison,
            metric="safety_utility",
        )
        errorbar(axis_a, row, y, color=color, marker="o")
    axis_a.axvline(0.0, color="#222222", linewidth=0.8)
    axis_a.set_yticks(positions, [item[0] for item in matched_specs])
    axis_a.set_xlabel(r"ActiveMap $-$ reference: safety utility")
    axis_a.set_title("A  Matched reject/defer controls", loc="left", fontweight="bold")
    axis_a.grid(axis="x", color=light_gray, linewidth=0.5)

    causal = bundle["selection_safe_commit_2x2"]
    causal_specs = (
        ("Always Commit", "selection_gain_always_commit"),
        ("Safe Commit", "selection_gain_safe_commit"),
    )
    for index, (_label, comparison) in enumerate(causal_specs):
        y = 1 - index
        quality = find_row(
            causal,
            analysis="selection_x_safe_commit",
            comparison=comparison,
            metric="map_quality_after",
        )
        false_edit = find_row(
            causal,
            analysis="selection_x_safe_commit",
            comparison=comparison,
            metric="false_edit_rate",
        ).copy()
        false_edit["delta"], false_edit["ci95_low"], false_edit["ci95_high"] = (
            -false_edit["delta"],
            -false_edit["ci95_high"],
            -false_edit["ci95_low"],
        )
        errorbar(axis_b, quality, y + 0.11, color=blue, marker="o")
        errorbar(axis_b, false_edit, y - 0.11, color=green, marker="s")
    axis_b.axvline(0.0, color="#222222", linewidth=0.8)
    axis_b.set_yticks([1, 0], [item[0] for item in causal_specs])
    axis_b.set_xlabel(r"Selection gain (higher is better)")
    axis_b.set_title("B  Selection effect survives gating", loc="left", fontweight="bold")
    axis_b.grid(axis="x", color=light_gray, linewidth=0.5)
    axis_b.plot([], [], color=blue, marker="o", linestyle="none", label="Map quality")
    axis_b.plot(
        [], [], color=green, marker="s", linestyle="none", label="False-edit reduction"
    )
    axis_b.legend(frameon=False, loc="lower right")

    costs = {row["policy"]: row for row in bundle["cost_accounting"]}
    policies = ("notool", "benefit", "forced")
    labels = ("No tool", "ActiveMap", "Forced")
    colors = (gray, blue, orange)
    axis_c1.bar(
        range(3),
        [costs[policy]["shared_preacquisition_perception_ms"] for policy in policies],
        color=colors,
        width=0.68,
    )
    axis_c1.set_xticks(
        range(3), labels, rotation=28, rotation_mode="anchor", ha="right"
    )
    axis_c1.set_ylabel("Shared updater (ms/episode)")
    axis_c1.set_ylim(bottom=0)
    axis_c1.set_title("C  Cost scope", loc="left", fontweight="bold")
    axis_c1.grid(axis="y", color=light_gray, linewidth=0.5)
    axis_c2.bar(
        range(3),
        [costs[policy]["incremental_tool_calls"] for policy in policies],
        color=colors,
        width=0.68,
    )
    axis_c2.set_xticks(
        range(3), labels, rotation=28, rotation_mode="anchor", ha="right"
    )
    axis_c2.set_ylabel("Incremental tool calls/episode")
    axis_c2.set_ylim(bottom=0)
    axis_c2.grid(axis="y", color=light_gray, linewidth=0.5)
    reduction = 100.0 * bundle["derived"]["incremental_tool_call_reduction_vs_forced"]
    axis_c2.text(
        0.04,
        0.96,
        f"{reduction:.1f}% fewer\nvs forced",
        transform=axis_c2.transAxes,
        ha="left",
        va="top",
        color=blue,
        fontsize=6.5,
        fontweight="bold",
    )

    operations = bundle["operation_slices"]
    op_positions = list(reversed(range(len(operations))))
    for y, row in zip(op_positions, operations, strict=True):
        plot_row = {
            "delta": row["map_quality_delta"],
            "ci95_low": row["ci95_low"],
            "ci95_high": row["ci95_high"],
        }
        color = blue if row["operation"] == "KEEP" else gray
        errorbar(axis_d, plot_row, y, color=color, marker="o")
    axis_d.axvline(0.0, color="#222222", linewidth=0.8)
    axis_d.set_yticks(
        op_positions,
        [f"{row['operation']}  (n={row['record_count']:,})" for row in operations],
    )
    axis_d.set_xlabel(r"Selection $-$ no-tool: map quality")
    axis_d.set_title("D  Typed-operation scope", loc="left", fontweight="bold")
    axis_d.grid(axis="x", color=light_gray, linewidth=0.5)

    figure.text(
        0.5,
        0.02,
        "Points show paired deltas; whiskers are 95% hierarchical bootstrap CIs. "
        "All panels use three-seed validation aggregates; no frozen-test asset is read.",
        ha="center",
        fontsize=6.3,
        color="#4B5563",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "reviewer_closure_evidence"
    outputs = [
        stem.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png", ".tiff")
    ]
    figure.savefig(outputs[0], bbox_inches="tight")
    figure.savefig(outputs[1], bbox_inches="tight")
    figure.savefig(outputs[2], dpi=600, bbox_inches="tight")
    figure.savefig(outputs[3], dpi=600, bbox_inches="tight")
    plt.close(figure)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r1-r2", type=Path, default=DEFAULT_R1_R2)
    parser.add_argument("--causal-2x2", type=Path, default=DEFAULT_CAUSAL)
    parser.add_argument("--cost", type=Path, default=DEFAULT_COST)
    parser.add_argument("--operations", type=Path, default=DEFAULT_OPERATIONS)
    parser.add_argument("--learned-defer", type=Path, default=DEFAULT_LEARNED_DEFER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_paths = {
        "r1_r2": args.r1_r2,
        "causal_2x2": args.causal_2x2,
        "cost": args.cost,
        "operations": args.operations,
        "learned_defer": args.learned_defer,
    }
    bundle = build_bundle(source_paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = args.output_dir / "reviewer_closure_evidence.json"
    bundle_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "matched_baselines.csv", bundle["matched_baselines"])
    write_csv(
        args.output_dir / "selection_safe_commit_2x2.csv",
        bundle["selection_safe_commit_2x2"],
    )
    write_csv(
        args.output_dir / "selection_safe_commit_cells.csv",
        bundle["selection_safe_commit_cells"],
    )
    write_csv(args.output_dir / "cost_accounting.csv", bundle["cost_accounting"])
    write_csv(args.output_dir / "operation_slices.csv", bundle["operation_slices"])
    (args.output_dir / "reviewer_closure_evidence.md").write_text(
        render_markdown(bundle), encoding="utf-8"
    )
    (args.output_dir / "reviewer_closure_tables.tex").write_text(
        render_latex(bundle), encoding="utf-8"
    )
    figure_outputs = render_figure(bundle, args.output_dir)
    output_files = [
        bundle_path,
        args.output_dir / "matched_baselines.csv",
        args.output_dir / "selection_safe_commit_2x2.csv",
        args.output_dir / "selection_safe_commit_cells.csv",
        args.output_dir / "cost_accounting.csv",
        args.output_dir / "operation_slices.csv",
        args.output_dir / "reviewer_closure_evidence.md",
        args.output_dir / "reviewer_closure_tables.tex",
        *figure_outputs,
    ]
    manifest = {
        "schema_version": "activemap-reviewer-closure-figure-manifest-v1",
        "test_assets_read": False,
        "frozen_test_aggregate_reused": False,
        "sources": bundle["sources"],
        "outputs": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in output_files
        },
    }
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(bundle["derived"], indent=2))


if __name__ == "__main__":
    main()
