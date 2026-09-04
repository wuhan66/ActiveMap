#!/usr/bin/env python3
"""Export the protocol-matched SN7 replaceable-perception baseline table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_table(concat_path: Path, comparison_path: Path) -> dict[str, Any]:
    concat = _load(concat_path)
    comparison = _load(comparison_path)
    if concat.get("schema_version") != "sn7-concat-unet-three-seed-validation-v1":
        raise ValueError("unexpected concat U-Net schema")
    if concat.get("split") != "val" or concat.get("test_assets_read") is not False:
        raise ValueError("concat U-Net input must be validation-only")
    if comparison.get("schema_version") != "sn7-updater-paired-aoi-bootstrap-v1":
        raise ValueError("unexpected external comparison schema")
    if comparison.get("test_assets_read") is not False:
        raise ValueError("external comparison must be validation-only")
    if int(concat["sample_count_per_seed"]) != int(comparison["sample_count"]):
        raise ValueError("baseline inputs do not share validation support")
    if int(concat["aoi_count"]) != int(comparison["aoi_count"]):
        raise ValueError("baseline inputs do not share AOI support")

    external = comparison["results"]
    prior_iou = (
        float(external["changemamba"]["committed_map_iou"]["observed"])
        - float(external["changemamba"]["map_iou_delta"]["observed"])
    )
    concat_metrics = concat["aggregate"]
    rows = [
        {
            "method": "Concat U-Net",
            "source": "in-tree control",
            "seed_count": int(concat["seed_count"]),
            "committed_map_iou": float(concat_metrics["mean_raster_iou"]["mean"]),
            "map_iou_gain": (
                float(concat_metrics["mean_raster_iou"]["mean"]) - prior_iou
            ),
            "operation_accuracy": float(concat_metrics["edit_accuracy"]["mean"]),
            "false_edit_rate": float(concat_metrics["false_edit_rate"]["mean"]),
            "uncertainty": "sample std across three seeds",
        }
    ]
    for key, label, source in (
        ("changemamba", "ChangeMamba", "official code adapted"),
        ("ban", "BAN (Open-CD)", "official toolbox adapted"),
    ):
        metrics = external[key]
        rows.append(
            {
                "method": label,
                "source": source,
                "seed_count": 3,
                "committed_map_iou": float(
                    metrics["committed_map_iou"]["observed"]
                ),
                "map_iou_gain": float(metrics["map_iou_delta"]["observed"]),
                "operation_accuracy": float(
                    metrics["operation_accuracy"]["observed"]
                ),
                "false_edit_rate": float(metrics["false_edit_rate"]["observed"]),
                "uncertainty": "paired seed-mean AOI bootstrap",
            }
        )

    return {
        "schema_version": "sn7-perception-baseline-paper-table-v1",
        "split": "val",
        "test_assets_read": False,
        "paper_role": "replaceable perception backend comparison",
        "sample_count_per_seed": int(comparison["sample_count"]),
        "aoi_count": int(comparison["aoi_count"]),
        "prior_map_iou": prior_iou,
        "protocol_note": (
            "All methods consume the same current image and editable prior and "
            "are evaluated after executable map writeback. ActiveMap does not "
            "claim a new segmentation architecture."
        ),
        "inputs": {
            "concat_unet": {
                "path": str(concat_path.resolve()),
                "sha256": _sha256(concat_path),
            },
            "external_comparison": {
                "path": str(comparison_path.resolve()),
                "sha256": _sha256(comparison_path),
            },
        },
        "rows": rows,
    }


def _write_csv(path: Path, payload: dict[str, Any]) -> None:
    fields = (
        "method",
        "source",
        "seed_count",
        "committed_map_iou",
        "map_iou_gain",
        "operation_accuracy",
        "false_edit_rate",
        "uncertainty",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(payload["rows"])


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# SN7 Replaceable Perception Baselines",
        "",
        "| Method | Committed map IoU | Map IoU gain | Operation accuracy | False edit |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['method']} | {row['committed_map_iou']:.6f} | "
            f"{row['map_iou_gain']:+.6f} | {row['operation_accuracy']:.6f} | "
            f"{row['false_edit_rate']:.6f} |"
        )
    lines.extend(["", payload["protocol_note"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_latex(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Commit IoU $\uparrow$ & $\Delta$IoU $\uparrow$ & Op. Acc. $\uparrow$ & False Edit $\downarrow$ \\",
        r"\midrule",
    ]
    for row in payload["rows"]:
        lines.append(
            f"{row['method']} & {row['committed_map_iou']:.4f} & "
            f"{row['map_iou_gain']:+.4f} & {row['operation_accuracy']:.4f} & "
            f"{row['false_edit_rate']:.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--concat", type=Path, required=True)
    parser.add_argument("--external-comparison", type=Path, required=True)
    args = parser.parse_args()
    payload = build_table(args.concat, args.external_comparison)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "table.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_dir / "table.csv", payload)
    _write_markdown(args.output_dir / "table.md", payload)
    _write_latex(args.output_dir / "table.tex", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
