#!/usr/bin/env python3
"""Replace label-derived selector history with an online-observable state.

Historical sequential selector records store the realised utility of previously
selected evidence in ``state_features[7]``. That quantity is valid as an
offline training label but unavailable when an online controller makes its next
decision. This tool creates a new immutable train/validation manifest whose
state 7 is the confidence of the current fused updater belief. It preserves all
supervision fields exactly and rejects test records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from activemap.agent.tools import CounterfactualBeliefUpdater
from activemap.features import ONLINE_OBSERVABLE_STATE_CONTRACT
from activemap.selector_records import SelectorSample

ONLINE_STATE_CONTRACT = ONLINE_OBSERVABLE_STATE_CONTRACT


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def supervision_digest(samples: Iterable[SelectorSample]) -> str:
    """Hash every target-bearing field to prove feature-only transformation."""

    digest = hashlib.sha256()
    for sample in samples:
        payload = {
            "sample_id": sample.sample_id,
            "split": sample.split,
            "evidence_ids": sample.evidence_ids,
            "evidence_costs": sample.evidence_costs,
            "false_edit_risks": sample.false_edit_risks,
            "oracle_utilities": sample.oracle_utilities,
            "stop_utility": sample.stop_utility,
            "false_edit_penalty_weight": sample.false_edit_penalty_weight,
        }
        digest.update(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
    return digest.hexdigest()


def observable_fused_confidence(sample: SelectorSample) -> float:
    selected = sample.metadata.get("selected_evidence_ids")
    if not isinstance(selected, list) or not selected:
        raise ValueError(f"{sample.sample_id} lacks selected_evidence_ids")
    predictions = sample.metadata.get("evidence_predictions")
    if not isinstance(predictions, dict):
        raise ValueError(f"{sample.sample_id} lacks evidence_predictions")
    missing = [str(evidence_id) for evidence_id in selected if evidence_id not in predictions]
    if missing:
        raise ValueError(f"{sample.sample_id} has no predictions for selected evidence: {missing}")
    belief = CounterfactualBeliefUpdater(sample).fuse([str(item) for item in selected])
    confidence = float(belief.confidence)
    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{sample.sample_id} has invalid fused confidence")
    return confidence


def sanitize_sample(sample: SelectorSample) -> SelectorSample:
    if sample.split == "test" or sample.metadata.get("test_assets_read") is True:
        raise ValueError("online selector sanitization only permits train/validation records")
    state = list(sample.state_features)
    state[7] = observable_fused_confidence(sample)
    metadata = dict(sample.metadata)
    metadata["online_state_contract"] = dict(ONLINE_STATE_CONTRACT)
    metadata["runtime_state7_source"] = "fused_belief_confidence"
    return sample.model_copy(update={"state_features": state, "metadata": metadata})


def load_samples(path: Path) -> list[SelectorSample]:
    samples: list[SelectorSample] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                samples.append(SelectorSample.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"invalid selector sample at line {line_number}") from exc
    if not samples:
        raise ValueError("selector manifest is empty")
    return samples


def sanitize_manifest(source: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    samples = load_samples(source)
    sanitized = [sanitize_sample(sample) for sample in samples]
    before_digest = supervision_digest(samples)
    after_digest = supervision_digest(sanitized)
    if before_digest != after_digest:
        raise RuntimeError("feature sanitization changed supervision")
    old_state7 = np.asarray([sample.state_features[7] for sample in samples], dtype=np.float64)
    new_state7 = np.asarray([sample.state_features[7] for sample in sanitized], dtype=np.float64)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(sample.model_dump_json() + "\n" for sample in sanitized), encoding="utf-8"
    )
    result = {
        "schema_version": "online-observable-selector-features-v1",
        "source": {"path": str(source.resolve()), "sha256": sha256_path(source)},
        "output": {"path": str(output.resolve()), "sha256": sha256_path(output)},
        "record_count": len(samples),
        "split_counts": {
            split: sum(sample.split == split for sample in samples)
            for split in ("train", "val")
        },
        "online_state_contract": ONLINE_STATE_CONTRACT,
        "state7": {
            "legacy_mean": float(old_state7.mean()),
            "observable_mean": float(new_state7.mean()),
            "mean_absolute_change": float(np.mean(np.abs(new_state7 - old_state7))),
            "changed_record_count": int(np.sum(np.abs(new_state7 - old_state7) > 1e-12)),
        },
        "supervision_digest": before_digest,
        "test_assets_read": False,
    }
    summary = output.with_suffix(output.suffix + ".summary.json")
    summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    print(json.dumps(sanitize_manifest(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
