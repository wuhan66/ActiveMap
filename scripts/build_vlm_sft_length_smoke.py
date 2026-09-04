#!/usr/bin/env python3
"""Build a worst-case visual-SFT smoke set from a completed token audit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


def _load_example(path: Path, example_id: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("example_id")) == example_id:
                result = copy.deepcopy(row)
                for part in result["messages"][1]["content"]:
                    image = part.get("image")
                    if part.get("type") == "image" and isinstance(image, str):
                        image_path = Path(image)
                        if not image_path.is_absolute():
                            part["image"] = str((path.parent / image_path).resolve())
                return result
    raise ValueError(f"example {example_id} not found in {path}")


def _first_example_id(path: Path) -> str:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return str(json.loads(line)["example_id"])
    raise ValueError(f"empty input: {path}")


def _write(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(row, separators=(",", ":")) + "\n"
    path.write_text(payload, encoding="utf-8")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "example_id": str(row["example_id"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    parser.add_argument("train_jsonl", type=Path)
    parser.add_argument("val_jsonl", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if audit.get("schema_version") != "visual-sft-token-audit-v1":
        raise ValueError("unexpected token-audit schema")
    if audit.get("test_assets_read") is not False:
        raise ValueError("audit is not train/validation-only")

    sources = {"train": args.train_jsonl, "val": args.val_jsonl}
    selected: dict[str, dict[str, Any]] = {}
    for split, source in sources.items():
        offenders = audit["splits"][split]["over_max_length_examples"]
        if offenders:
            offender = max(offenders, key=lambda item: int(item["tokens"]))
            example_id = str(offender["example_id"])
            tokens: int | None = int(offender["tokens"])
            selection = "longest_reported_offender"
        else:
            example_id = _first_example_id(source)
            tokens = None
            selection = "deterministic_fallback_no_offender"
        selected[split] = {
            "tokens": tokens,
            "selection": selection,
            "split_max_tokens": int(audit["splits"][split]["tokens"]["maximum"]),
            "row": _load_example(source, example_id),
        }

    args.output_root.mkdir(parents=True)
    report = {
        "schema_version": "visual-sft-length-smoke-v1",
        "source_audit": str(args.audit.resolve()),
        "source_max_length": int(audit["max_length"]),
        "train": {
            **_write(args.output_root / "train.jsonl", selected["train"]["row"]),
            "tokens": selected["train"]["tokens"],
            "selection": selected["train"]["selection"],
            "split_max_tokens": selected["train"]["split_max_tokens"],
        },
        "val": {
            **_write(args.output_root / "val.jsonl", selected["val"]["row"]),
            "tokens": selected["val"]["tokens"],
            "selection": selected["val"]["selection"],
            "split_max_tokens": selected["val"]["split_max_tokens"],
        },
        "test_assets_read": False,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
