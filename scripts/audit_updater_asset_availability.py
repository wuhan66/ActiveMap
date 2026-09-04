"""Audit which updater records have all local array assets available."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--splits", default="train,val")
    args = parser.parse_args()

    splits = {value.strip() for value in args.splits.split(",") if value.strip()}
    root = args.manifest.parent
    complete: list[str] = []
    counts: Counter[tuple[str, str]] = Counter()
    total: Counter[tuple[str, str]] = Counter()
    with args.manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["split"] not in splits:
                continue
            key = (row["split"], row["edit_type"])
            total[key] += 1
            paths = [
                _resolve(root, row["image_path"]),
                _resolve(root, row["prior_mask_path"]),
                _resolve(root, row["target_mask_path"]),
                _resolve(root, row.get("valid_mask_path")),
            ]
            if all(path is None or path.is_file() for path in paths):
                counts[key] += 1
                complete.append(line if line.endswith("\n") else line + "\n")

    report = {
        "schema_version": "updater-asset-availability-v1",
        "manifest": str(args.manifest),
        "requested_splits": sorted(splits),
        "available_count": len(complete),
        "available_by_split_edit": {
            f"{split}:{edit}": counts[(split, edit)]
            for split, edit in sorted(total)
        },
        "total_by_split_edit": {
            f"{split}:{edit}": total[(split, edit)]
            for split, edit in sorted(total)
        },
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(complete), encoding="utf-8")
        report["output"] = str(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
