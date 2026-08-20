"""Atomic progress files for long-running data construction jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
