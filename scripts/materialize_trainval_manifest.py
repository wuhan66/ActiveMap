#!/usr/bin/env python3
"""Create a hash-recorded train/validation manifest without touching image assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    source = arguments.source.resolve()
    output = arguments.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite derived manifest: {output}")

    frame = pd.read_parquet(source)
    if "split" not in frame.columns:
        raise ValueError("source manifest has no split column")
    split_values = frame["split"].astype(str)
    selected = frame.loc[split_values.isin({"train", "val"})].copy()
    if selected.empty:
        raise ValueError("source manifest has no train/validation rows")
    if set(selected["split"].astype(str)) - {"train", "val"}:
        raise RuntimeError("derived manifest contains a non-train/validation split")
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(output, index=False)
    summary = {
        "source_manifest": str(source),
        "source_manifest_sha256": _sha256(source),
        "derived_manifest": str(output),
        "allowed_splits": ["train", "val"],
        "source_row_counts": split_values.value_counts().sort_index().to_dict(),
        "derived_row_counts": selected["split"].astype(str).value_counts().sort_index().to_dict(),
        "derived_rows": int(len(selected)),
        "test_assets_read": False,
        "note": "The source parquet metadata is filtered before any image, UDM, or label asset is opened.",
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
