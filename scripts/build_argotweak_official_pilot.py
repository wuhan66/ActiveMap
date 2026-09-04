from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an official-format ArgoTweak pickle for selected TbV logs."
    )
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--tbv-root", type=Path, required=True)
    parser.add_argument("--extension-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--segment-id", action="append")
    source.add_argument("--subset-file", type=Path)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    official_root = args.official_root.resolve()
    sys.path.insert(0, str(official_root))

    # The official converter parses its own CLI at import time.
    original_argv = sys.argv
    sys.argv = ["convert_data.py", "--tbv-root", str(args.tbv_root)]
    try:
        from tools import convert_data
    finally:
        sys.argv = original_argv

    args.output_dir.mkdir(parents=True, exist_ok=True)
    convert_data.save_path = str(args.output_dir)
    segment_ids = args.segment_id
    if args.subset_file is not None:
        subset = json.loads(args.subset_file.read_text(encoding="utf-8"))
        segment_ids = list(subset["splits"][args.split].values())
    if args.offset < 0 or (args.limit is not None and args.limit < 1):
        raise ValueError("offset must be non-negative and limit must be positive")
    segment_ids = segment_ids[args.offset : None if args.limit is None else args.offset + args.limit]
    if not segment_ids:
        raise ValueError("selected ArgoTweak shard is empty")
    segment_map = {
        f"pilot_{index:04d}": segment_id
        for index, segment_id in enumerate(segment_ids)
    }
    convert_data.collect(
        str(args.tbv_root),
        str(args.extension_root),
        {args.split: segment_map},
        args.output_name,
        n_points=10,
    )


if __name__ == "__main__":
    main()
