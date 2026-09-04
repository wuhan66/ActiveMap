from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MAXIMIZE = (
    "realized_utility_mean",
    "selection_macro_f1",
    "exact_evidence_recall",
)
MINIMIZE = (
    "false_call_rate",
    "harmful_call_rate_all_states",
    "mean_cost_per_state",
    "mean_regret",
)


def dominates(left: dict[str, Any], right: dict[str, Any], tolerance: float = 1e-12) -> bool:
    left_metrics, right_metrics = left["metrics"], right["metrics"]
    no_worse = all(
        left_metrics[key] >= right_metrics[key] - tolerance for key in MAXIMIZE
    ) and all(left_metrics[key] <= right_metrics[key] + tolerance for key in MINIMIZE)
    strictly_better = any(
        left_metrics[key] > right_metrics[key] + tolerance for key in MAXIMIZE
    ) or any(left_metrics[key] < right_metrics[key] - tolerance for key in MINIMIZE)
    return no_worse and strictly_better


def mark_pareto(rows: list[dict[str, Any]]) -> None:
    eligible = [row for row in rows if row["gate_passed"]]
    for row in rows:
        row["pareto"] = row["gate_passed"] and not any(
            other is not row and dominates(other, row) for other in eligible
        )


def load_rows(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        summaries = sorted(run_dir.glob("sentinel_eval_*/summary.json"))
        for summary_path in summaries:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            rows.append(
                {
                    "run": run_dir.name,
                    "candidate": summary_path.parent.name.removeprefix("sentinel_eval_"),
                    "summary": str(summary_path.resolve()),
                    "gate_passed": bool(payload.get("promotion_gate", {}).get("passed")),
                    "metrics": {key: float(metrics[key]) for key in MAXIMIZE + MINIMIZE},
                }
            )
    mark_pareto(rows)
    return rows


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    flat_rows = [
        {
            "run": row["run"],
            "candidate": row["candidate"],
            "gate_passed": row["gate_passed"],
            "pareto": row["pareto"],
            **row["metrics"],
            "summary": row["summary"],
        }
        for row in rows
    ]
    csv_path = output_dir / "sentinel_candidate_metrics.csv"
    fields = list(flat_rows[0]) if flat_rows else [
        "run",
        "candidate",
        "gate_passed",
        "pareto",
        *MAXIMIZE,
        *MINIMIZE,
        "summary",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)

    lines = [
        "# Active-Catalog Sentinel Pareto Table",
        "",
        "Diagnostic validation only. Pareto status is computed among candidates that pass the frozen gate.",
        "",
        "| Run | Candidate | Gate | Pareto | Utility | Macro-F1 | Exact recall | False call | Harmful call | Cost/state | Regret |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        m = row["metrics"]
        lines.append(
            f"| {row['run']} | {row['candidate']} | {int(row['gate_passed'])} | "
            f"{int(row['pareto'])} | {m['realized_utility_mean']:.5f} | "
            f"{m['selection_macro_f1']:.4f} | {m['exact_evidence_recall']:.4f} | "
            f"{m['false_call_rate']:.4f} | {m['harmful_call_rate_all_states']:.4f} | "
            f"{m['mean_cost_per_state']:.4f} | {m['mean_regret']:.5f} |"
        )
    (output_dir / "sentinel_candidate_metrics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output_dir / "sentinel_candidate_metrics.json").write_text(
        json.dumps(
            {
                "schema_version": "active-catalog-sentinel-pareto-v1",
                "selection_role": "diagnostic_screening_not_final_promotion",
                "maximize": MAXIMIZE,
                "minimize": MINIMIZE,
                "rows": rows,
                "test_assets_read": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.run_dirs)
    if not rows:
        raise SystemExit("no sentinel summaries found")
    write_outputs(rows, args.output_dir)
    print(f"wrote {len(rows)} candidates to {args.output_dir}")


if __name__ == "__main__":
    main()
