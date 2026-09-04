#!/usr/bin/env python3
"""Create a source-episode-disjoint tune split from V5 selector *train* states.

The formal V5 validation states are reserved for the final four-cell writeback
matrix. This utility relabels a deterministic subset of original training
source episodes as ``val`` only for selector checkpoint selection, while
retaining provenance that both resulting files originated from the train split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from activemap.selector_records import SelectorSample


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def group_key(sample: SelectorSample) -> tuple[str, str]:
    source = str(sample.metadata.get("source_episode", ""))
    if not source:
        raise ValueError(f"selector state {sample.sample_id} lacks source_episode")
    target = "ACQUIRE" if sample.target_index() < len(sample.evidence_ids) else "STOP"
    return str(sample.metadata.get("gt_edit", sample.edit_type.value)), target


def split_samples(
    samples: list[SelectorSample], *, tune_fraction: float, seed: int
) -> tuple[list[SelectorSample], list[SelectorSample], dict[str, Any]]:
    if not 0.05 <= tune_fraction <= 0.4:
        raise ValueError("tune_fraction must be in [0.05, 0.4]")
    by_source: dict[str, list[SelectorSample]] = defaultdict(list)
    for sample in samples:
        if sample.split != "train":
            raise ValueError("internal selector split accepts original train states only")
        by_source[str(sample.metadata["source_episode"])].append(sample)
    if len(by_source) < 2:
        raise ValueError("internal split requires at least two source episodes")
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for source, source_samples in by_source.items():
        signatures = {group_key(sample) for sample in source_samples}
        # A source can contain several typed targets; use a stable composite
        # rather than silently splitting one source episode across partitions.
        signature = "+".join(f"{edit}:{target}" for edit, target in sorted(signatures))
        strata[(signature, "source")].append(source)
    tune_sources: set[str] = set()
    stratum_summary: dict[str, dict[str, int]] = {}
    for stratum, sources in sorted(strata.items()):
        ordered = sorted(
            sources,
            key=lambda source: hashlib.sha256(
                f"{seed}:{stratum}:{source}".encode("utf-8")
            ).hexdigest(),
        )
        count = round(len(ordered) * tune_fraction)
        if len(ordered) > 1:
            count = min(max(count, 1), len(ordered) - 1)
        tune_sources.update(ordered[:count])
        stratum_summary[str(stratum[0])] = {
            "source_episodes": len(ordered),
            "tune_source_episodes": count,
        }
    fit = [sample for sample in samples if str(sample.metadata["source_episode"]) not in tune_sources]
    tune = [
        sample.model_copy(update={"split": "val"})
        for sample in samples
        if str(sample.metadata["source_episode"]) in tune_sources
    ]
    fit_sources = {str(sample.metadata["source_episode"]) for sample in fit}
    tune_sources_from_rows = {str(sample.metadata["source_episode"]) for sample in tune}
    if not fit or not tune or fit_sources & tune_sources_from_rows:
        raise ValueError("internal source-episode split is degenerate")
    summary = {
        "source_episode_count": len(by_source),
        "fit_source_episode_count": len(fit_sources),
        "tune_source_episode_count": len(tune_sources_from_rows),
        "source_episode_overlap": 0,
        "strata": stratum_summary,
        "fit_target_counts": dict(
            Counter("ACQUIRE" if item.target_index() < len(item.evidence_ids) else "STOP" for item in fit)
        ),
        "tune_target_counts": dict(
            Counter("ACQUIRE" if item.target_index() < len(item.evidence_ids) else "STOP" for item in tune)
        ),
    }
    return fit, tune, summary


def write_rows(path: Path, rows: list[SelectorSample]) -> str:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")
    return sha256(path)


def create_internal_split(
    source: Path, output_dir: Path, *, tune_fraction: float, seed: int
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite internal selector split: {output_dir}")
    samples = [
        SelectorSample.model_validate_json(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not samples:
        raise ValueError("no selector states supplied")
    fit, tune, split_summary = split_samples(samples, tune_fraction=tune_fraction, seed=seed)
    output_dir.mkdir(parents=True)
    fit_path = output_dir / "fit.jsonl"
    tune_path = output_dir / "tune.jsonl"
    combined_path = output_dir / "fit_tune.jsonl"
    fit_sha = write_rows(fit_path, fit)
    tune_sha = write_rows(tune_path, tune)
    combined_sha = write_rows(combined_path, fit + tune)
    summary = {
        "schema_version": "sn7-v5-selector-train-internal-split-v1",
        "source": {
            "path": str(source.resolve()),
            "sha256": sha256(source),
            "original_split": "train",
            "state_count": len(samples),
        },
        "split": {"seed": seed, "tune_fraction": tune_fraction, **split_summary},
        "outputs": {
            "fit": {"path": str(fit_path.resolve()), "sha256": fit_sha, "state_count": len(fit)},
            "tune": {"path": str(tune_path.resolve()), "sha256": tune_sha, "state_count": len(tune)},
            "fit_tune": {
                "path": str(combined_path.resolve()),
                "sha256": combined_sha,
                "state_count": len(fit) + len(tune),
            },
        },
        "formal_validation_assets_read": False,
        "test_assets_read": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--tune-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            create_internal_split(
                args.source,
                args.output_dir,
                tune_fraction=args.tune_fraction,
                seed=args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
