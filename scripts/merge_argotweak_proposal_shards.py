from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge validated ArgoTweak proposal shards.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite merged proposals: {args.output}")
    rows = []
    keys: set[tuple[str, str, str]] = set()
    for path in args.inputs:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["split"]), str(row["segment_id"]), str(row["timestamp"]))
            if key in keys:
                raise ValueError(f"duplicate ArgoTweak proposal frame: {key}")
            if "gt_operation_counts" not in row:
                raise ValueError(f"proposal shard lacks native supervision: {path}")
            keys.add(key)
            rows.append(row)
    if not rows:
        raise ValueError("no proposal rows found")
    rows.sort(key=lambda row: (str(row["split"]), str(row["segment_id"]), str(row["timestamp"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "activemap-argotweak-native-proposal-merge-v1",
        "inputs": [str(path.resolve()) for path in args.inputs],
        "frames": len(rows),
        "split_counts": {
            split: sum(str(row["split"]) == split for row in rows)
            for split in sorted({str(row["split"]) for row in rows})
        },
        "test_assets_read": False,
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
