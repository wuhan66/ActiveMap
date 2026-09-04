#!/usr/bin/env python3
"""Select one SFT-anchor weight on validation before seed replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.summarize_muno21_grpo_v2 import metrics, read_rows


def parse_candidate(value: str) -> tuple[str, float, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("candidate requires LABEL=WEIGHT=JSONL")
    return parts[0], float(parts[1]), Path(parts[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--false-edit-margin", type=float, default=0.005)
    args = parser.parse_args()

    baseline = metrics(read_rows(args.baseline))
    candidates = []
    for label, weight, path in args.candidate:
        values = metrics(read_rows(path))
        candidates.append(
            {
                "label": label,
                "weight": weight,
                "path": str(path),
                "metrics": values,
                "false_edit_eligible": values["false_edit_rate"]
                <= baseline["false_edit_rate"] + args.false_edit_margin,
                "utility_positive": values["episode_utility_v2_proxy_balanced_auc"]
                > baseline["episode_utility_v2_proxy_balanced_auc"],
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            row["metrics"]["episode_utility_v2_proxy_balanced_auc"],
            -row["metrics"]["false_edit_rate"],
            row["metrics"]["macro_f1"],
        ),
    )
    payload = {
        "schema_version": "muno21-sft-anchored-grpo-weight-selection-v1",
        "selection_split": "val",
        "selection_rule": "max balanced utility; tie-break lower false edit then macro-F1",
        "false_edit_margin": args.false_edit_margin,
        "baseline": baseline,
        "candidates": candidates,
        "selected": selected,
        "test_assets_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"{selected['weight']:.8g}")


if __name__ == "__main__":
    main()
