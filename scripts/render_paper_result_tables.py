#!/usr/bin/env python3
"""Render provenance-bound paper tables from audited result bundles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_paper_result_bundles import (
    _cell_id,
    _expected_cells,
    audit_result_bundles,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bundle_index(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    bundles: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "activemap-paper-result-v1":
            continue
        cell_id = _cell_id(
            str(payload["experiment_id"]),
            payload.get("variant"),
            float(payload["budget"]) if payload.get("budget") is not None else None,
        )
        bundles[cell_id] = payload
        paths[cell_id] = path
    return bundles, paths


def _metric_text(summary: dict[str, Any]) -> str:
    low, high = map(float, summary["ci95"])
    return (
        f"{float(summary['mean']):.4f} +/- {float(summary['std']):.4f} "
        f"[{low:.4f}, {high:.4f}]"
    )


def render_result_tables(
    registry_path: Path,
    bundles_root: Path,
    output_dir: Path,
    *,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    audit = audit_result_bundles(registry_path, bundles_root)
    if not audit["contract_valid"]:
        raise ValueError(f"bundle contract errors: {audit['errors']}")
    if not allow_incomplete and not audit["paper_tables_complete"]:
        raise ValueError(
            f"paper result set is incomplete: {len(audit['missing_cells'])} cells missing"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    expected = _expected_cells(registry)
    bundles, bundle_paths = _bundle_index(bundles_root)
    cells_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in expected:
        bundle = bundles.get(cell["id"])
        if bundle is not None:
            cells_by_family[str(cell["family"])].append({"cell": cell, "bundle": bundle})

    output_paths: list[Path] = []
    for family, entries in sorted(cells_by_family.items()):
        metric_names = sorted(
            {
                metric
                for entry in entries
                for metric in entry["bundle"].get("metrics", {})
            }
        )
        csv_path = output_dir / f"{family}.csv"
        fieldnames = [
            "cell_id",
            "experiment_id",
            "variant",
            "budget",
            "split",
            "seeds",
            "sample_count",
            "unit_count",
        ]
        for metric in metric_names:
            fieldnames.extend(
                [f"{metric}_mean", f"{metric}_std", f"{metric}_ci95_low", f"{metric}_ci95_high"]
            )
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for entry in entries:
                cell = entry["cell"]
                bundle = entry["bundle"]
                row: dict[str, Any] = {
                    "cell_id": cell["id"],
                    "experiment_id": cell["experiment_id"],
                    "variant": cell["variant"] or "",
                    "budget": "" if cell["budget"] is None else cell["budget"],
                    "split": cell["split"],
                    "seeds": ";".join(map(str, bundle["seeds"])),
                    "sample_count": bundle["sample_count"],
                    "unit_count": bundle["unit_count"],
                }
                for metric, summary in bundle["metrics"].items():
                    row[f"{metric}_mean"] = summary["mean"]
                    row[f"{metric}_std"] = summary["std"]
                    row[f"{metric}_ci95_low"] = summary["ci95"][0]
                    row[f"{metric}_ci95_high"] = summary["ci95"][1]
                writer.writerow(row)
        output_paths.append(csv_path)

        display_metrics = list(
            dict.fromkeys(
                metric
                for entry in entries
                for metric in entry["cell"]["primary_metrics"]
            )
        )
        markdown = [
            f"# {family.title()} Results",
            "",
            "All values are mean +/- seed standard deviation [grouped-bootstrap 95% CI].",
            "",
            "| Cell | Split | " + " | ".join(display_metrics) + " |",
            "|---|---|" + "---:|" * len(display_metrics),
        ]
        for entry in entries:
            cell = entry["cell"]
            metrics = entry["bundle"]["metrics"]
            markdown.append(
                f"| `{cell['id']}` | {cell['split']} | "
                + " | ".join(
                    _metric_text(metrics[metric]) if metric in metrics else "-"
                    for metric in display_metrics
                )
                + " |"
            )
        markdown_path = output_dir / f"{family}.md"
        markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
        output_paths.append(markdown_path)

    manifest = {
        "schema_version": "activemap-paper-table-manifest-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": bool(audit["paper_tables_complete"]),
        "allow_incomplete": allow_incomplete,
        "registry": {"path": str(registry_path.resolve()), "sha256": _sha256(registry_path)},
        "bundles": [
            {"cell_id": cell_id, "path": str(path.resolve()), "sha256": _sha256(path)}
            for cell_id, path in sorted(bundle_paths.items())
        ],
        "tables": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in output_paths
        ],
        "audit": {
            "expected_cell_count": audit["expected_cell_count"],
            "complete_cell_count": audit["complete_cell_count"],
            "missing_cells": audit["missing_cells"],
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("bundles_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    manifest = render_result_tables(
        args.registry,
        args.bundles_root,
        args.output_dir,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(manifest["audit"], indent=2))


if __name__ == "__main__":
    main()
