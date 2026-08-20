"""JSONL validation with line-level error reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


def validate_jsonl(jsonl_path: Path, schema_path: Path) -> tuple[int, list[str]]:
    schema: dict[str, Any] = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    valid_count = 0
    errors: list[str] = []

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc.msg}")
                continue
            record_errors = sorted(validator.iter_errors(record), key=lambda item: list(item.path))
            if record_errors:
                for error in record_errors:
                    location = ".".join(str(part) for part in error.absolute_path) or "$"
                    errors.append(f"line {line_number} at {location}: {error.message}")
                continue
            valid_count += 1
    return valid_count, errors
