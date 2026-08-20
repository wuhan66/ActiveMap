"""Stable public handles for identifiers exposed to language models."""

from __future__ import annotations

import hashlib


def public_task_id(raw_id: str) -> str:
    return _public_id("task", raw_id)


def public_evidence_id(raw_id: str) -> str:
    return _public_id("evidence", raw_id)


def resolve_evidence_id(exposed_id: str, raw_ids: list[str]) -> str:
    """Resolve either a raw or public evidence handle without reversing its hash."""

    direct = [raw_id for raw_id in raw_ids if raw_id == exposed_id]
    public = [raw_id for raw_id in raw_ids if public_evidence_id(raw_id) == exposed_id]
    matches = list(dict.fromkeys(direct + public))
    if len(matches) != 1:
        raise ValueError(f"unknown or ambiguous evidence_id: {exposed_id}")
    return matches[0]


def _public_id(kind: str, raw_id: str) -> str:
    digest = hashlib.sha256(f"activemap:{kind}:{raw_id}".encode()).hexdigest()[:16]
    return f"{kind}-{digest}"
