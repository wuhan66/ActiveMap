#!/usr/bin/env python3
"""Build nested, task-grouped selector training subsets for scale ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _task_id(row: dict[str, Any]) -> str:
    sample_id = str(row.get("sample_id", ""))
    if "__" not in sample_id:
        raise ValueError(f"sample_id lacks task separator: {sample_id}")
    return sample_id.split("__", 1)[0]


def _fraction_label(value: float) -> str:
    return f"p{int(round(value * 1000)):04d}"


def build_subsets(
    source: Path,
    output_dir: Path,
    fractions: list[float],
    seed: int,
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("selector state file is empty")
    if output_dir.exists():
        raise FileExistsError(f"refusing existing output: {output_dir}")
    if any(not 0.0 < value < 1.0 for value in fractions):
        raise ValueError("fractions must be in (0, 1)")
    fractions = sorted(set(fractions))

    train_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    validation: list[dict[str, Any]] = []
    edit_by_task: dict[str, str] = {}
    for row in rows:
        split = str(row.get("split"))
        if split == "val":
            validation.append(row)
            continue
        if split != "train":
            raise ValueError(f"unsupported split: {split}")
        task_id = _task_id(row)
        edit = str(row.get("edit_type"))
        previous = edit_by_task.setdefault(task_id, edit)
        if previous != edit:
            raise ValueError(f"task spans edit types: {task_id}")
        train_by_task[task_id].append(row)

    tasks_by_edit: dict[str, list[str]] = defaultdict(list)
    for task_id, edit in edit_by_task.items():
        tasks_by_edit[edit].append(task_id)
    for offset, edit in enumerate(sorted(tasks_by_edit)):
        random.Random(seed + offset).shuffle(tasks_by_edit[edit])

    output_dir.mkdir(parents=True)
    outputs: list[dict[str, Any]] = []
    previous_tasks: set[str] = set()
    for fraction in fractions:
        selected: set[str] = set()
        for edit in sorted(tasks_by_edit):
            tasks = tasks_by_edit[edit]
            count = max(1, math.ceil(len(tasks) * fraction))
            selected.update(tasks[:count])
        if not previous_tasks.issubset(selected):
            raise AssertionError("scale subsets are not nested")
        previous_tasks = selected
        subset_rows = [
            row
            for row in rows
            if str(row.get("split")) == "val" or _task_id(row) in selected
        ]
        label = _fraction_label(fraction)
        output = output_dir / f"selector_states_{label}.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for row in subset_rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        train_rows = [row for row in subset_rows if row["split"] == "train"]
        record = {
            "label": label,
            "fraction": fraction,
            "train_task_count": len(selected),
            "train_record_count": len(train_rows),
            "validation_record_count": len(validation),
            "train_edit_task_counts": dict(
                sorted(Counter(edit_by_task[task] for task in selected).items())
            ),
            "path": str(output.resolve()),
            "sha256": _sha256(output),
        }
        (output.with_suffix(output.suffix + ".summary.json")).write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        outputs.append(record)

    summary = {
        "schema_version": "activemap-selector-data-scale-v1",
        "source": str(source.resolve()),
        "source_sha256": _sha256(source),
        "seed": seed,
        "train_task_count_full": len(train_by_task),
        "validation_record_count": len(validation),
        "nested": True,
        "test_assets_read": False,
        "outputs": outputs,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--fractions", default="0.125,0.25,0.5,0.75")
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    fractions = [float(value) for value in args.fractions.split(",")]
    print(json.dumps(build_subsets(args.source, args.output_dir, fractions, args.seed), indent=2))


if __name__ == "__main__":
    main()
