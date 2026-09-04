#!/usr/bin/env python3
"""Summarize completed Changer/TinyCD cross-region controller results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for rank in (2, 100):
        for backend in ("changer", "tinycd"):
            stem = f"rank{rank}_{backend}"
            summary = read_json(args.root / f"{stem}_selector_safe_commit/summary.json")
            bootstrap = read_json(args.root / f"{stem}_bootstrap5000/summary.json")
            selection = summary["ambiguous_selection"]
            terminal = summary["terminal_all_validation"]
            selection_ci = bootstrap["selection_bootstrap"]["learned_minus_first"]
            safe_ci = bootstrap["terminal_bootstrap"]["safe_minus_commit"]
            row = {
                "rank": rank,
                "backend": backend,
                "ambiguous_episodes": summary["protocol"]["ambiguous_val_episodes"],
                "first_iou": selection["first"]["mean_map_iou"],
                "random_iou": selection["random_expected"]["mean_map_iou"],
                "learned_iou": selection["learned_ranker"]["mean_map_iou"],
                "oracle_iou": selection["oracle"]["mean_map_iou"],
                "learned_minus_first": selection_ci["mean"],
                "learned_minus_first_ci_low": selection_ci["ci95_low"],
                "learned_minus_first_ci_high": selection_ci["ci95_high"],
                "learned_minus_first_probability_positive": selection_ci[
                    "probability_positive"
                ],
                "safe_iou": terminal["safe_commit"]["mean_map_iou"],
                "always_commit_iou": terminal["always_commit"]["mean_map_iou"],
                "safe_false_edit": terminal["safe_commit"]["false_edit_rate"],
                "always_commit_false_edit": terminal["always_commit"]["false_edit_rate"],
                "safe_minus_commit_iou": safe_ci["map_iou"]["mean"],
                "safe_minus_commit_false_edit": safe_ci["false_edit"]["mean"],
                "test_assets_read": bootstrap["test_assets_read"],
            }
            rows.append(row)

    fieldnames = list(rows[0])
    with (args.output / "modern_cross_region.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "modern_cross_region.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "| Region | Backend | First | Random | Learned | Oracle | "
        "Learned-First (95% CI) | Safe-Commit IoU delta | False-edit delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| rank{row['rank']} | {row['backend']} | {row['first_iou']:.4f} | "
            f"{row['random_iou']:.4f} | {row['learned_iou']:.4f} | "
            f"{row['oracle_iou']:.4f} | {row['learned_minus_first']:+.4f} "
            f"[{row['learned_minus_first_ci_low']:+.4f}, "
            f"{row['learned_minus_first_ci_high']:+.4f}] | "
            f"{row['safe_minus_commit_iou']:+.4f} | "
            f"{row['safe_minus_commit_false_edit']:+.4f} |"
        )
    (args.output / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
