#!/usr/bin/env python3
"""Generate paper-ready V5 2x2 tables from a registered aggregate receipt.

The program contains no thresholding or promotion logic.  It only turns the
completed validation-only aggregate into literal LaTex tables, so the paper
cannot silently relabel a mixed result as a positive one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

POLICIES = (
    "direct_commit",
    "direct_safe_commit",
    "selected_commit",
    "selected_safe_commit",
)
FORCED_ACQUISITION_POLICY = "forced_safe_commit"
DISPLAY_NAMES = {
    "direct_commit": "Direct Commit",
    "direct_safe_commit": "Direct + Safe Commit",
    "selected_commit": "Selected Commit",
    "selected_safe_commit": "Selected + Safe Commit",
}
METRICS = (
    ("final_map_quality", "Final map quality"),
    ("false_edit_rate", "False edit"),
    ("missed_edit_rate", "Missed edit"),
    ("commit_rate", "Commit"),
    ("additional_evidence_rate", "Extra evidence"),
    ("additional_cost", "Extra cost"),
)
CONTRAST_METRICS = (
    ("final_map_quality", "Final map quality"),
    ("false_edit_rate", "False edit"),
    ("missed_edit_rate", "Missed edit"),
)


def read_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("schema_version") not in {
        "sn7-v5-matched-nonkeep-factorial-v1",
        "sn7-v5-matched-nonkeep-factorial-v2",
    }:
        raise ValueError("unexpected V5 factorial aggregate schema")
    if summary.get("split") != "val" or summary.get("test_assets_read") is not False:
        raise ValueError("V5 factorial table source must be validation-only")
    if summary.get("model_seeds") != [20260817, 20260818, 20260819]:
        raise ValueError("V5 factorial aggregate lacks all registered seeds")
    registered_policies = tuple(summary.get("policies", ()))
    if registered_policies not in {POLICIES, POLICIES + (FORCED_ACQUISITION_POLICY,)}:
        raise ValueError("V5 factorial aggregate lacks the registered 2x2 policies")
    return summary


def number(value: Any) -> str:
    return f"{float(value):.4f}"


def interval(value: dict[str, Any]) -> str:
    return f"{number(value['delta'])} [{number(value['ci95_low'])}, {number(value['ci95_high'])}]"


def policy_table(summary: dict[str, Any]) -> str:
    policies = summary["policy_metrics"]
    lines = [
        "% Generated from a registered V5 validation aggregate. Do not edit numbers by hand.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Registered V5 matched non-KEEP validation matrix. Values are "
        "three-seed AOI-macro averages; all four cells use the same V5 candidate bank.}",
        "\\label{tab:v5-factorial-policy}",
        "\\scriptsize",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Policy & Final quality & False edit & Missed edit & Commit & Extra evidence & "
        "Extra cost \\\\",
        "\\midrule",
    ]
    for policy in POLICIES:
        values = policies[policy]["three_seed_aoi_macro"]
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} \\\\".format(
                DISPLAY_NAMES[policy],
                *(number(values[name]) for name, _ in METRICS),
            )
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def _contrast_rows(
    summary: dict[str, Any],
    *,
    key: str,
    display: str,
) -> list[str]:
    result = []
    overall = summary[key]["paired_delta"]
    result.append(
        "{} & All & {} & {} & {} \\\\".format(
            display,
            *(interval(overall[name]) for name, _ in CONTRAST_METRICS),
        )
    )
    for operation in ("ADD", "DELETE", "RESHAPE"):
        values = summary["operation_slices"][operation][key]["paired_delta"]
        result.append(
            "{} & {} & {} & {} & {} \\\\".format(
                display,
                operation.title(),
                *(interval(values[name]) for name, _ in CONTRAST_METRICS),
            )
        )
    return result


def contrast_table(summary: dict[str, Any]) -> str:
    lines = [
        "% Generated from a registered V5 validation aggregate. Intervals are paired 95% CIs.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Factorized V5 validation contrasts. Selection compares Selected + Safe "
        "Commit with Direct + Safe Commit. Terminal safety compares Selected + Safe Commit "
        "with Selected Commit. Entries are paired deltas and confidence intervals.}",
        "\\label{tab:v5-factorial-contrast}",
        "\\scriptsize",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Factor & Slice & Final map quality & False edit & Missed edit \\\\",
        "\\midrule",
    ]
    lines.extend(_contrast_rows(summary, key="selection_factor", display="Selection"))
    lines.append("\\midrule")
    lines.extend(_contrast_rows(summary, key="safe_commit_factor", display="Safe Commit"))
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def forced_cost_table(summary: dict[str, Any]) -> str:
    control = summary["promotion"].get("forced_acquisition_cost_control")
    if not isinstance(control, dict):
        raise ValueError("forced-acquisition control is not available")
    cost = control["paired_delta"]["additional_cost"]
    lines = [
        "% Generated from a registered V5 validation aggregate. Do not edit numbers by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Registered V5 sparse-acquisition cost control. The forced policy uses the "
        "same frozen selector ranker but acquires its top candidate even when STOP wins. "
        "Negative values favor Selected + Safe Commit.}",
        "\\label{tab:v5-forced-cost-control}",
        "\\scriptsize",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Contrast & Extra candidate cost delta [paired confidence interval] \\\\",
        "\\midrule",
        "Selected + Safe Commit minus Forced + Safe Commit & " + interval(cost) + " \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def generate(summary_path: Path, output_dir: Path) -> dict[str, Path]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite V5 LaTex directory: {output_dir}")
    summary = read_summary(summary_path)
    output_dir.mkdir(parents=True)
    outputs = {
        "policy": output_dir / "sn7_v5_factorial_policy_table.tex",
        "contrasts": output_dir / "sn7_v5_factorial_contrast_table.tex",
    }
    outputs["policy"].write_text(policy_table(summary), encoding="utf-8")
    outputs["contrasts"].write_text(contrast_table(summary), encoding="utf-8")
    if FORCED_ACQUISITION_POLICY in summary["policies"]:
        outputs["forced_cost"] = output_dir / "sn7_v5_forced_cost_control_table.tex"
        outputs["forced_cost"].write_text(forced_cost_table(summary), encoding="utf-8")
    receipt = {
        "schema_version": "sn7-v5-factorial-tex-v2",
        "split": "val",
        "test_assets_read": False,
        "source_summary": str(summary_path.resolve()),
        "source_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "policies": summary["policies"],
        "promotion": summary.get("promotion"),
        "outputs": {name: str(path.name) for name, path in outputs.items()},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            {name: str(path) for name, path in generate(args.summary, args.output_dir).items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
