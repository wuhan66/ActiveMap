#!/usr/bin/env python3
"""Apply a train-fitted terminal arbitration gate to paired rollout files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.analyze_terminal_arbitration_frontier import load_rows, summarize
except ModuleNotFoundError:
    from analyze_terminal_arbitration_frontier import load_rows, summarize


def passes_gate(value: float, direction: str, threshold: float) -> bool:
    if direction == "ge":
        return value >= threshold
    if direction == "le":
        return value <= threshold
    raise ValueError(f"unknown direction: {direction}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("gate", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    baseline = load_rows(args.baseline)
    candidate = load_rows(args.candidate)
    if baseline.keys() != candidate.keys():
        raise ValueError("baseline and candidate sample IDs do not match")
    config = json.loads(args.gate.read_text(encoding="utf-8"))
    if config.get("fit_split") != "train":
        raise ValueError("gate must be fitted on train split")
    feature = str(config["feature"])
    direction = str(config["direction"])
    threshold = float(config["threshold"])

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_path = args.output_dir / "hybrid_terminal_arbitration.jsonl"
    selected_rows: list[dict[str, Any]] = []
    accepted = 0
    disagreements = 0
    with output_path.open("x", encoding="utf-8") as handle:
        for sample_id in sorted(baseline):
            base_row = baseline[sample_id]
            candidate_row = candidate[sample_id]
            disagreement = base_row["prediction"] != candidate_row["prediction"]
            disagreements += int(disagreement)
            accepted_candidate = disagreement and passes_gate(
                float(candidate_row[feature]), direction, threshold
            )
            accepted += int(accepted_candidate)
            source = candidate_row if accepted_candidate else base_row
            row = dict(source)
            row["terminal_arbitration_gate"] = {
                "schema_version": config["schema_version"],
                "feature": feature,
                "direction": direction,
                "threshold": threshold,
                "fit_split": "train",
                "accepted_candidate": accepted_candidate,
                "baseline_prediction": base_row["prediction"],
                "candidate_prediction": candidate_row["prediction"],
                "uses_ground_truth_at_inference": False,
            }
            selected_rows.append(row)
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    result = {
        "schema_version": "terminal-arbitration-application-v1",
        "rows": len(selected_rows),
        "disagreements": disagreements,
        "accepted_candidate_disagreements": accepted,
        "gate": config,
        "baseline": summarize(list(baseline.values())),
        "candidate": summarize(list(candidate.values())),
        "hybrid": summarize(selected_rows),
        "split": next(iter({row["split"] for row in selected_rows})),
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
