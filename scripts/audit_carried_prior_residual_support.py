"""Audit the train/internal residual labels induced by carried-prior corruption."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from activemap.config import load_yaml
from activemap.models import EditOperation
from activemap.training.updater import _filter_samples_by_edit
from activemap.training.updater_data import (
    CarriedPriorResidualConfig,
    _corrupt_carried_prior,
    _load_mask,
    _residual_operation,
)
from activemap.updater_records import load_updater_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    return parser.parse_args()


def _operation_name(index: int) -> str:
    return tuple(EditOperation)[index].value


def audit(config_path: Path, *, split: str) -> dict[str, Any]:
    config = load_yaml(config_path)
    data_settings = config.get("data", {})
    samples_path = Path(data_settings["samples"])
    if not samples_path.is_absolute():
        samples_path = (config_path.parent / samples_path).resolve()
    samples = [
        sample
        for sample in _filter_samples_by_edit(
            load_updater_samples(samples_path), data_settings.get("allowed_edits")
        )
        if sample.split == split
    ]
    residual_config = CarriedPriorResidualConfig.from_dict(
        data_settings.get("carried_prior_residual")
    )
    source_counts: Counter[str] = Counter()
    residual_counts: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    applied = 0
    for sample in samples:
        source = sample.edit_type.value
        prior = _load_mask(sample.prior_mask_path).squeeze(0)
        target = _load_mask(sample.target_mask_path).squeeze(0)
        valid = (
            _load_mask(sample.valid_mask_path).squeeze(0)
            if sample.valid_mask_path is not None
            else np.ones_like(target, dtype=np.float32)
        )
        corrupted_prior, was_corrupted = _corrupt_carried_prior(
            prior,
            sample_id=sample.sample_id,
            config=residual_config,
        )
        residual = (
            _residual_operation(corrupted_prior, target, valid).value
            if was_corrupted
            else source
        )
        source_counts[source] += 1
        residual_counts[residual] += 1
        transitions[f"{source}->{residual}"] += 1
        applied += int(was_corrupted)
    return {
        "schema_version": "carried-prior-residual-support-v1",
        "config": str(config_path.resolve()),
        "samples": str(samples_path.resolve()),
        "split": split,
        "sample_count": len(samples),
        "corruption": {
            "enabled": residual_config.enabled,
            "probability": residual_config.probability,
            "max_translation_pixels": residual_config.max_translation_pixels,
            "morphology_pixels": residual_config.morphology_pixels,
            "seed": residual_config.seed,
        },
        "corruption_applied_count": applied,
        "source_operation_counts": dict(sorted(source_counts.items())),
        "residual_operation_counts": dict(sorted(residual_counts.items())),
        "source_to_residual_counts": dict(sorted(transitions.items())),
        "residual_non_keep_count": sum(
            count
            for operation, count in residual_counts.items()
            if operation != EditOperation.KEEP.value
        ),
    }


def main() -> None:
    args = parse_args()
    summary = audit(args.config, split=args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
