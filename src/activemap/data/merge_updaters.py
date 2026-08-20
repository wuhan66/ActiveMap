"""Merge updater manifests while preserving dataset and supervision provenance."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from activemap.updater_records import UpdaterSample, load_updater_samples


def merge_updater_manifests(
    manifests: list[Path],
    output: Path,
) -> dict[str, Any]:
    if not manifests:
        raise ValueError("at least one updater manifest is required")
    samples: list[UpdaterSample] = []
    sample_ids: set[str] = set()
    for manifest in manifests:
        for sample in load_updater_samples(manifest):
            if sample.sample_id in sample_ids:
                raise ValueError(f"duplicate sample_id across manifests: {sample.sample_id}")
            sample_ids.add(sample.sample_id)
            samples.append(sample)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(sample.model_dump_json() + "\n")

    summary: dict[str, Any] = {
        "samples": len(samples),
        "manifests": [str(path.resolve()) for path in manifests],
        "output": str(output.resolve()),
        "datasets": dict(sorted(Counter(s.dataset_name for s in samples).items())),
        "splits": dict(sorted(Counter(s.split for s in samples).items())),
        "operations": dict(sorted(Counter(s.edit_type.value for s in samples).items())),
        "geometry_families": dict(
            sorted(Counter(s.geometry_family for s in samples).items())
        ),
        "supervision_types": dict(
            sorted(Counter(s.supervision_type for s in samples).items())
        ),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
