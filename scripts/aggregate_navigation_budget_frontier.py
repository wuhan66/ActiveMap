"""Aggregate immutable navigation rollout summaries into a budget frontier."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.output_dir.exists():
        raise FileExistsError(arguments.output_dir)
    rows: list[dict[str, object]] = []
    shared_protocol: dict[str, object] | None = None
    for path in arguments.summaries:
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocol = payload["protocol"]
        if shared_protocol is None:
            shared_protocol = {
                key: protocol[key]
                for key in (
                    "split",
                    "test_assets_read",
                    "max_steps",
                    "commit_rule",
                    "policy_input",
                    "observation_access",
                    "state_transition",
                )
            }
        elif any(protocol[key] != value for key, value in shared_protocol.items()):
            raise ValueError(f"protocol mismatch in {path}")
        budget = protocol["budget_override"]
        if not isinstance(budget, float | int):
            raise ValueError(f"budget_override is required in {path}")
        for result in payload["results"]:
            rows.append({"budget": budget, **result})

    arguments.output_dir.mkdir(parents=True)
    rows.sort(key=lambda row: (str(row["policy"]), float(row["budget"])))
    (arguments.output_dir / "budget_frontier.json").write_text(
        json.dumps(
            {
                "schema_version": "activemap-navigation-budget-frontier-v1",
                "protocol": shared_protocol,
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    csv_path = arguments.output_dir / "budget_frontier.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
