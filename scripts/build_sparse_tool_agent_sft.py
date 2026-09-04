#!/usr/bin/env python3
"""Create sparse post-acquisition USE_TOOL SFT and preference records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from activemap.agent.tool_belief_data import ToolBeliefSequenceExample
from activemap.agent.tool_sft import write_sparse_tool_dataset

ModelT = TypeVar("ModelT", bound=BaseModel)


def _read_models(path: Path, model: type[ModelT]) -> list[ModelT]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(model.model_validate_json(line))
                except Exception as exc:
                    raise ValueError(f"invalid row {line_number} in {path}") from exc
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def _read_dicts(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"row {line_number} in {path} is not an object")
                rows.append(row)
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequences", type=Path)
    parser.add_argument("details", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--minimum-gain", type=float, default=1e-6)
    parser.add_argument("--frozen-test", action="store_true")
    args = parser.parse_args()
    if args.frozen_test:
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
        if args.output_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite frozen test tool labels: {args.output_dir}"
            )
    sequences = _read_models(args.sequences, ToolBeliefSequenceExample)
    has_test = any(sequence.split == "test" for sequence in sequences)
    if has_test and not args.frozen_test:
        raise PermissionError("test sparse-tool labels require --frozen-test")
    if args.frozen_test and not has_test:
        raise ValueError("--frozen-test requires test-only sequences")
    summary = write_sparse_tool_dataset(
        sequences,
        _read_dicts(args.details),
        args.output_dir,
        minimum_gain=args.minimum_gain,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
