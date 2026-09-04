#!/usr/bin/env python3
"""Audit that controller tool traces do not expose evaluation targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

FORBIDDEN_KEYS = frozenset(
    {
        "ground_truth",
        "gt_edit",
        "label",
        "oracle_utilities",
        "oracle_utility",
        "target_edit",
        "target_geometry",
        "target_map",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _forbidden_paths(value: Any, prefix: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{prefix}.{key}"
            if str(key).lower() in FORBIDDEN_KEYS:
                hits.append(child_path)
            hits.extend(_forbidden_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(_forbidden_paths(child, f"{prefix}[{index}]"))
    return hits


def discover(inputs: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for item in inputs:
        if item.is_file():
            paths.add(item.resolve())
        elif item.is_dir():
            paths.update(path.resolve() for path in item.rglob("tool_results.jsonl"))
        else:
            raise FileNotFoundError(item)
    if not paths:
        raise FileNotFoundError("no tool_results.jsonl traces found")
    return sorted(paths)


def audit(paths: list[Path]) -> dict[str, Any]:
    tool_counts: Counter[str] = Counter()
    tool_failure_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    trace_count = 0
    invocation_count = 0
    forbidden: list[dict[str, Any]] = []
    files = []
    for path in paths:
        file_traces = 0
        file_invocations = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                trace_count += 1
                file_traces += 1
                hits = _forbidden_paths(row)
                if hits:
                    forbidden.append(
                        {
                            "path": str(path),
                            "line": line_number,
                            "key_paths": hits,
                        }
                    )
                for result in row.get("results", []):
                    invocation_count += 1
                    file_invocations += 1
                    tool = str(result.get("tool", "UNKNOWN"))
                    tool_counts[tool] += 1
                    if not bool(result.get("success", False)):
                        tool_failure_counts[tool] += 1
                        error = str(result.get("error") or "UNKNOWN")
                        error_counts[f"{tool}: {error}"] += 1
        files.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "trace_count": file_traces,
                "invocation_count": file_invocations,
            }
        )
    failed_invocations = sum(tool_failure_counts.values())
    return {
        "schema_version": "sn7-controller-tool-independence-audit-v1",
        "status": "pass" if not forbidden else "fail",
        "trace_count": trace_count,
        "invocation_count": invocation_count,
        "tool_counts": dict(sorted(tool_counts.items())),
        "tool_failure_counts": dict(sorted(tool_failure_counts.items())),
        "tool_success_rates": {
            tool: (count - tool_failure_counts[tool]) / count
            for tool, count in sorted(tool_counts.items())
        },
        "failed_invocations": failed_invocations,
        "failure_rate": failed_invocations / invocation_count if invocation_count else 0.0,
        "error_counts": dict(sorted(error_counts.items())),
        "forbidden_keys": sorted(FORBIDDEN_KEYS),
        "forbidden_hits": forbidden,
        "files": files,
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = audit(discover(args.inputs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "status",
        "trace_count",
        "invocation_count",
        "tool_counts",
        "tool_success_rates",
        "failed_invocations",
        "failure_rate",
    )}, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
