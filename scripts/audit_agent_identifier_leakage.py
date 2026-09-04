#!/usr/bin/env python3
"""Fail when language-model-visible IDs expose semantic edit labels."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

TASK_PATTERN = re.compile(r"^task-[0-9a-f]{16}$")
EVIDENCE_PATTERN = re.compile(r"^evidence-[0-9a-f]{16}$")
LABEL_PATTERN = re.compile(r"(?:^|[-_])(keep|add|delete|reshape)(?:[-_]|$)", re.I)


def _ids(observation: dict[str, Any]) -> tuple[list[str], list[str]]:
    evidence = [str(item) for item in observation.get("selected_evidence_ids", [])]
    evidence.extend(str(item["evidence_id"]) for item in observation.get("candidates", []))
    return [str(observation.get("task_id", ""))], evidence


def _record_payloads(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = row.get("messages", [])
    if messages:
        user = next((item for item in messages if item.get("role") == "user"), None)
        assistant = next(
            (item for item in messages if item.get("role") == "assistant"), None
        )
        if user is None or assistant is None:
            raise ValueError("missing user/assistant message")
        return json.loads(user["content"]), json.loads(assistant["content"])

    if "prompt" in row and "chosen" in row:
        observation_lines = [
            line for line in str(row["prompt"]).splitlines() if line.startswith('{"task_id"')
        ]
        if len(observation_lines) != 1:
            raise ValueError("preference prompt does not contain one observation")
        return json.loads(observation_lines[0]), json.loads(row["chosen"])
    raise ValueError("unsupported Agent record schema")


def audit(path: Path) -> dict[str, int]:
    records = 0
    failures: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            records += 1
            row = json.loads(line)
            try:
                observation, action = _record_payloads(row)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                failures.append(f"line {line_number}: {error}")
                continue
            task_ids, evidence_ids = _ids(observation)
            if action.get("action") == "ACQUIRE":
                evidence_ids.append(str(action.get("evidence_id", "")))
            for value in task_ids:
                if not TASK_PATTERN.fullmatch(value) or LABEL_PATTERN.search(value):
                    failures.append(f"line {line_number}: unsafe task_id {value!r}")
            for value in evidence_ids:
                if not EVIDENCE_PATTERN.fullmatch(value) or LABEL_PATTERN.search(value):
                    failures.append(f"line {line_number}: unsafe evidence_id {value!r}")
    if failures:
        preview = "\n".join(failures[:20])
        raise SystemExit(f"identifier leakage audit failed ({len(failures)}):\n{preview}")
    return {"records": records, "failures": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    summaries = {str(path): audit(path) for path in args.paths}
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
