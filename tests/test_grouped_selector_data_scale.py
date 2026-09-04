from __future__ import annotations

import json
from pathlib import Path

from scripts.build_grouped_selector_data_scale import build_subsets


def _row(task: str, split: str, edit: str, suffix: int) -> dict[str, object]:
    return {
        "sample_id": f"{task}__b1p5__s{suffix}",
        "split": split,
        "edit_type": edit,
    }


def test_grouped_scale_subsets_are_nested_and_keep_validation(tmp_path: Path) -> None:
    source = tmp_path / "states.jsonl"
    rows = []
    for index in range(16):
        edit = "ADD" if index % 2 == 0 else "KEEP"
        rows.extend(_row(f"train-{index}", "train", edit, suffix) for suffix in range(2))
    rows.extend(_row("val-0", "val", "ADD", suffix) for suffix in range(3))
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    summary = build_subsets(source, tmp_path / "scaled", [0.25, 0.5, 0.75], 7)

    previous: set[str] = set()
    for record in summary["outputs"]:
        path = Path(record["path"])
        subset = [json.loads(line) for line in path.read_text().splitlines()]
        train_tasks = {
            row["sample_id"].split("__", 1)[0]
            for row in subset
            if row["split"] == "train"
        }
        assert previous.issubset(train_tasks)
        previous = train_tasks
        assert sum(row["split"] == "val" for row in subset) == 3
        assert record["validation_record_count"] == 3
