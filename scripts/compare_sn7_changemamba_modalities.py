#!/usr/bin/env python3
"""Compare protocol-matched ChangeMamba input-modality aggregates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

MODES = ("image_prior", "image_only", "prior_only")
METRICS = (
    "committed_map_iou",
    "map_iou_delta",
    "change_iou",
    "operation_accuracy",
    "keep_false_change_fraction",
)
IGNORED_PROTOCOL_KEYS = {"input_contract", "input_mode"}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "sn7-changemamba-three-seed-aggregate-v1":
        raise ValueError(f"unexpected aggregate schema: {path}")
    if value.get("run_count") != 3:
        raise ValueError(f"expected exactly three seeds: {path}")
    if value.get("protocol", {}).get("test_assets_read") is not False:
        raise ValueError(f"aggregate is not test-free: {path}")
    return value


def _protocol_without_mode(value: dict[str, Any]) -> dict[str, Any]:
    protocol = {
        key: item
        for key, item in value["protocol"].items()
        if key not in IGNORED_PROTOCOL_KEYS
    }
    protocol.setdefault("max_translation_pixels", 0)
    return protocol


def compare_modalities(paths: dict[str, Path]) -> dict[str, Any]:
    if set(paths) != set(MODES):
        raise ValueError(f"expected modes {MODES}, got {tuple(paths)}")
    values = {mode: _read(paths[mode]) for mode in MODES}
    reference = _protocol_without_mode(values["image_prior"])
    for mode in MODES[1:]:
        if _protocol_without_mode(values[mode]) != reference:
            raise ValueError(f"protocol mismatch for {mode}")

    by_mode_seed: dict[str, dict[int, dict[str, float]]] = {}
    for mode, value in values.items():
        rows = {
            int(run["seed"]): {
                metric: float(run["metrics"][metric]) for metric in METRICS
            }
            for run in value["runs"]
        }
        if len(rows) != 3:
            raise ValueError(f"duplicate seed in {mode}")
        by_mode_seed[mode] = rows
    seeds = sorted(by_mode_seed["image_prior"])
    if any(sorted(by_mode_seed[mode]) != seeds for mode in MODES[1:]):
        raise ValueError("seed support differs across modalities")

    summaries: dict[str, Any] = {}
    for mode in MODES:
        summaries[mode] = {
            metric: {
                "mean": float(
                    np.mean([by_mode_seed[mode][seed][metric] for seed in seeds])
                ),
                "std": float(
                    np.std(
                        [by_mode_seed[mode][seed][metric] for seed in seeds],
                        ddof=1,
                    )
                ),
            }
            for metric in METRICS
        }

    comparisons: dict[str, Any] = {}
    for ablation, interpretation in (
        ("image_only", "editable_prior_contribution"),
        ("prior_only", "new_image_contribution"),
    ):
        per_seed = {
            str(seed): {
                metric: (
                    by_mode_seed["image_prior"][seed][metric]
                    - by_mode_seed[ablation][seed][metric]
                )
                for metric in METRICS
            }
            for seed in seeds
        }
        comparisons[interpretation] = {
            "full_minus": ablation,
            "per_seed": per_seed,
            "mean_delta": {
                metric: float(
                    np.mean([per_seed[str(seed)][metric] for seed in seeds])
                )
                for metric in METRICS
            },
        }

    best_single = max(
        ("image_only", "prior_only"),
        key=lambda mode: summaries[mode]["map_iou_delta"]["mean"],
    )
    return {
        "schema_version": "sn7-changemamba-modality-comparison-v1",
        "test_assets_read": False,
        "seeds": seeds,
        "protocol": reference,
        "modalities": summaries,
        "paired_comparisons": comparisons,
        "joint_input_gain_over_best_single_modality": {
            "best_single_modality": best_single,
            "map_iou_delta": (
                summaries["image_prior"]["map_iou_delta"]["mean"]
                - summaries[best_single]["map_iou_delta"]["mean"]
            ),
        },
        "inference_scope": (
            "Three-seed paired descriptive statistics; AOI-level inference "
            "requires per-sample audits."
        ),
    }


def _markdown(result: dict[str, Any]) -> str:
    rows = [
        "| Input | Committed map IoU | Map-IoU delta | Change IoU "
        "| Operation accuracy | KEEP false change |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in MODES:
        metrics = result["modalities"][mode]
        rows.append(
            f"| {mode} | {metrics['committed_map_iou']['mean']:.5f} "
            f"+/- {metrics['committed_map_iou']['std']:.5f} | "
            f"{metrics['map_iou_delta']['mean']:.5f} "
            f"+/- {metrics['map_iou_delta']['std']:.5f} | "
            f"{metrics['change_iou']['mean']:.5f} | "
            f"{metrics['operation_accuracy']['mean']:.5f} | "
            f"{metrics['keep_false_change_fraction']['mean']:.5f} |"
        )
    rows.extend(
        [
            "",
            "| Paired contribution | Full minus ablation map-IoU delta |",
            "| --- | ---: |",
        ]
    )
    for name, comparison in result["paired_comparisons"].items():
        rows.append(
            f"| {name} | {comparison['mean_delta']['map_iou_delta']:+.5f} |"
        )
    gain = result["joint_input_gain_over_best_single_modality"]
    rows.append(
        f"| joint gain over {gain['best_single_modality']} | "
        f"{gain['map_iou_delta']:+.5f} |"
    )
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_prior", type=Path)
    parser.add_argument("image_only", type=Path)
    parser.add_argument("prior_only", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args()
    result = compare_modalities(
        {
            "image_prior": args.image_prior,
            "image_only": args.image_only,
            "prior_only": args.prior_only,
        }
    )
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.output_markdown is not None:
        args.output_markdown.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
