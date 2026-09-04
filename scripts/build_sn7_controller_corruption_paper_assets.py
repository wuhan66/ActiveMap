#!/usr/bin/env python3
"""Build paper tables and a joint claim gate for controller corruption."""

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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _delta(comparison: dict[str, Any], metric: str) -> dict[str, float]:
    value = comparison[metric]
    return {key: float(value[key]) for key in ("delta", "ci95_low", "ci95_high")}


def build_payload(
    summary: dict[str, Any],
    sensitivity: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    gates = []
    for source in summary["rows"]:
        severity = int(source["severity"])
        notool = source["writeback"]["notool"]
        forced = source["writeback"]["forced"]
        benefit = source["writeback"]["benefit"]
        benefit_notool = source["paired_writeback"]["benefit_vs_notool"]
        benefit_forced = source["paired_writeback"]["benefit_vs_forced"]
        iou = _delta(benefit_notool, "raster_iou_auc")
        false_edit = _delta(benefit_notool, "false_edit_auc")
        missed_edit = _delta(benefit_notool, "missed_edit_auc")
        cost_forced = _delta(benefit_forced, "spent_cost_auc")
        utility = _delta(benefit_notool, "episode_utility_v2_balanced_auc")
        denominator = float(forced["raster_iou_auc"]) - float(
            notool["raster_iou_auc"]
        )
        recovery = iou["delta"] / denominator if abs(denominator) > 1e-12 else None
        row = {
            "severity": severity,
            "notool_iou_auc": float(notool["raster_iou_auc"]),
            "benefit_iou_auc": float(benefit["raster_iou_auc"]),
            "benefit_vs_notool_iou_delta": iou["delta"],
            "benefit_vs_notool_iou_ci_low": iou["ci95_low"],
            "benefit_vs_notool_iou_ci_high": iou["ci95_high"],
            "notool_false_edit_auc": float(notool["false_edit_auc"]),
            "benefit_false_edit_auc": float(benefit["false_edit_auc"]),
            "benefit_vs_notool_false_edit_delta": false_edit["delta"],
            "benefit_vs_notool_false_edit_ci_low": false_edit["ci95_low"],
            "benefit_vs_notool_false_edit_ci_high": false_edit["ci95_high"],
            "benefit_vs_notool_missed_edit_delta": missed_edit["delta"],
            "benefit_vs_notool_balanced_utility_delta": utility["delta"],
            "benefit_call_rate": float(
                source["controller"]["benefit"]["tool_call_episode_rate"]
            ),
            "forced_call_rate": float(
                source["controller"]["forced"]["tool_call_episode_rate"]
            ),
            "benefit_vs_forced_cost_delta": cost_forced["delta"],
            "benefit_vs_forced_cost_ci_low": cost_forced["ci95_low"],
            "benefit_vs_forced_cost_ci_high": cost_forced["ci95_high"],
            "forced_quality_recovery_fraction": recovery,
        }
        rows.append(row)
        if severity:
            gates.append(
                {
                    "severity": severity,
                    "iou_gain_ci_positive": iou["ci95_low"] > 0.0,
                    "false_edit_ci_negative": false_edit["ci95_high"] < 0.0,
                    "cost_vs_forced_ci_negative": cost_forced["ci95_high"] < 0.0,
                }
            )
    sensitivity_iou = _delta(sensitivity["paired_delta"], "raster_iou_auc")
    sensitivity_false_edit = _delta(
        sensitivity["paired_delta"],
        "false_edit_auc",
    )
    sensitivity_gate = {
        "iou_gain_ci_positive": sensitivity_iou["ci95_low"] > 0.0,
        "false_edit_ci_negative": sensitivity_false_edit["ci95_high"] < 0.0,
    }
    audit_gate = {
        "status_pass": audit.get("status") == "pass",
        "no_forbidden_hits": not audit.get("forbidden_hits"),
    }
    all_checks = [
        value
        for gate in gates
        for key, value in gate.items()
        if key != "severity"
    ] + list(sensitivity_gate.values()) + list(audit_gate.values())
    return {
        "schema_version": "sn7-controller-corruption-paper-assets-v1",
        "rows": rows,
        "claim_gate": {
            "per_severity": gates,
            "corruption_realization_sensitivity": sensitivity_gate,
            "tool_independence": audit_gate,
            "passed": all(all_checks),
            "interpretation": (
                "frozen selective controller significantly recovers executable "
                "quality and false-edit safety under prior-input corruption while "
                "using significantly less cost than forced tools"
            ),
        },
        "test_assets_read": False,
    }


def _write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _interval(row: dict[str, Any], stem: str) -> str:
    return (
        f"{row[f'{stem}_delta']:+.6f} "
        f"[{row[f'{stem}_ci_low']:+.6f}, {row[f'{stem}_ci_high']:+.6f}]"
    )


def _write_markdown(rows: list[dict[str, Any]], output: Path) -> None:
    lines = [
        "| Shift | No-tool IoU | ActiveMap IoU | Delta IoU (95% CI) | "
        "No-tool FE | ActiveMap FE | Delta FE (95% CI) | Call rate | "
        "Cost vs forced (95% CI) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['severity']}px | {row['notool_iou_auc']:.6f} | "
            f"{row['benefit_iou_auc']:.6f} | "
            f"{_interval(row, 'benefit_vs_notool_iou')} | "
            f"{row['notool_false_edit_auc']:.6f} | "
            f"{row['benefit_false_edit_auc']:.6f} | "
            f"{_interval(row, 'benefit_vs_notool_false_edit')} | "
            f"{row['benefit_call_rate']:.4f} | "
            f"{_interval(row, 'benefit_vs_forced_cost')} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(rows: list[dict[str, Any]], output: Path) -> None:
    lines = [
        "\\begin{tabular}{rcccc}",
        "\\toprule",
        "Shift & No-tool IoU & ActiveMap IoU & $\\Delta$ IoU & Call rate \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['severity']} px & {row['notool_iou_auc']:.3f} & "
            f"{row['benefit_iou_auc']:.3f} & "
            f"{row['benefit_vs_notool_iou_delta']:+.3f} & "
            f"{100.0 * row['benefit_call_rate']:.1f}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("sensitivity", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    inputs = (args.summary, args.sensitivity, args.audit)
    payload = build_payload(
        *(json.loads(path.read_text(encoding="utf-8")) for path in inputs)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(payload["rows"], args.output_dir / "controller_corruption_table.csv")
    _write_markdown(
        payload["rows"],
        args.output_dir / "controller_corruption_table.md",
    )
    _write_latex(
        payload["rows"],
        args.output_dir / "controller_corruption_table.tex",
    )
    (args.output_dir / "claim_gate.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "paper_assets_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "sn7-controller-corruption-paper-manifest-v1",
                "inputs": [
                    {"path": str(path.resolve()), "sha256": _sha256(path)}
                    for path in inputs
                ],
                "claim_gate_passed": payload["claim_gate"]["passed"],
                "test_assets_read": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["claim_gate"], indent=2))


if __name__ == "__main__":
    main()
