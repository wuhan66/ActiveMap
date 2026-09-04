#!/usr/bin/env python3
"""Audit temporal image multiplicity for MUNO21 train/validation samples."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def _image_prefix(path: Path) -> str:
    match = re.match(r"^(.*)_(?:19|20)\d{2}.*$", path.stem)
    return match.group(1) if match else path.stem.rsplit("_", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--splits", default="train,val")
    args = parser.parse_args()
    splits = {value.strip() for value in args.splits.split(",") if value.strip()}
    if not splits or not splits <= {"train", "val"}:
        parser.error("this pre-test audit only permits train and val")

    histogram: Counter[int] = Counter()
    split_histograms: dict[str, Counter[int]] = {split: Counter() for split in splits}
    sample_count = 0
    missing_sources = 0
    examples: list[dict[str, object]] = []
    with args.manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            split = str(record["split"])
            if split not in splits:
                continue
            source = Path(record["source_metadata"]["source_image"])
            if not source.is_file():
                missing_sources += 1
                continue
            candidates = sorted(source.parent.glob(f"{_image_prefix(source)}_*.jpg"))
            count = len(candidates)
            histogram[count] += 1
            split_histograms[split][count] += 1
            sample_count += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "sample_id": record["sample_id"],
                        "split": split,
                        "count": count,
                        "images": [path.name for path in candidates],
                    }
                )

    summary = {
        "requested_splits": sorted(splits),
        "test_assets_read": False,
        "sample_count": sample_count,
        "missing_sources": missing_sources,
        "candidate_count_histogram": dict(sorted(histogram.items())),
        "split_histograms": {
            split: dict(sorted(values.items())) for split, values in split_histograms.items()
        },
        "multi_evidence_fraction": (
            sum(count for size, count in histogram.items() if size >= 2) / sample_count
            if sample_count
            else 0.0
        ),
        "examples": examples,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
