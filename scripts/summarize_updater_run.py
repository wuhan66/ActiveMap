"""Create a compact, immutable-facing report for one updater run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

METRICS = (
    "edit_accuracy",
    "macro_f1",
    "stable_f1",
    "update_precision",
    "update_recall",
    "update_f1",
    "false_edit_rate",
    "missed_update_rate",
    "commit_precision",
    "mean_raster_iou",
    "mean_polygon_iou",
    "topology_valid_rate",
    "ece",
    "brier",
    "nll",
    "aurc",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_count": payload["sample_count"],
        "aoi_count": payload["aoi_count"],
        **{name: payload[name] for name in METRICS},
        "per_edit": payload["per_edit"],
        "confusion": payload["confusion"],
        "bootstrap": payload["bootstrap"],
    }


def _metric_table(best: dict[str, Any], last: dict[str, Any]) -> list[str]:
    rows = ["| Metric | validation-selected best | final epoch diagnostic |", "|---|---:|---:|"]
    for metric in METRICS:
        rows.append(f"| `{metric}` | {best[metric]:.6f} | {last[metric]:.6f} |")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    training = _load(run_dir / "metrics.json")
    provenance = _load(run_dir / "provenance" / "run_provenance.json")
    best_test = _compact(_load(run_dir / "test_best" / "summary.json"))
    last_test = _compact(_load(run_dir / "test_last" / "summary.json"))
    history = training["history"]
    scored = [
        (
            int(record["epoch"]),
            float(record["val"]["iou"])
            + float(record["val"]["edit_accuracy"])
            - float(record["val"]["false_edit_rate"]),
        )
        for record in history
    ]
    best_epoch, computed_best_score = max(scored, key=lambda item: item[1])
    checkpoints = {
        name: {
            "path": str((run_dir / name).resolve()),
            "sha256": _sha256(run_dir / name),
            "size_bytes": (run_dir / name).stat().st_size,
        }
        for name in ("best.pt", "last.pt")
    }
    report = {
        "run_dir": str(run_dir),
        "dataset": provenance["dataset"],
        "environment": {
            key: provenance[key]
            for key in (
                "created_at",
                "python_executable",
                "torch_version",
                "cuda_version",
                "cuda_visible_devices",
                "cuda_devices",
            )
        },
        "training": {
            "epochs_completed": training["epochs_completed"],
            "early_stopped": training["epochs_completed"] < 30,
            "selection_formula": "val_iou + val_edit_accuracy - val_false_edit_rate",
            "best_epoch": best_epoch,
            "best_score": computed_best_score,
            "first_epoch": history[0],
            "last_epoch": history[-1],
        },
        "checkpoints": checkpoints,
        "test_best": best_test,
        "test_last_diagnostic_only": last_test,
        "selection_note": (
            "test_last is diagnostic only; checkpoint selection remains frozen on validation"
        ),
    }
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "run_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Updater Base v3 Run Report",
        "",
        f"- Run: `{run_dir}`",
        f"- Dataset SHA-256: `{provenance['dataset']['sha256']}`",
        f"- Samples: `{provenance['dataset']['samples']}`",
        f"- Epochs: `{training['epochs_completed']}` (early stopped)",
        f"- Best epoch: `{best_epoch}`",
        "- Selection: `val IoU + val edit accuracy - val false-edit rate`",
        "- `last.pt` test values below are diagnostic only and were not used for selection.",
        "",
        "## Test Comparison",
        "",
        *_metric_table(best_test, last_test),
        "",
        "## Validation Losses",
        "",
        (
            "| Epoch | Total | Segmentation | Edit | Geometry | False edit | "
            "Missed edit | Confidence |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in history:
        val = record["val"]
        lines.append(
            f"| {record['epoch']} | {val['loss']:.6f} | {val['loss_segmentation']:.6f} "
            f"| {val['loss_edit']:.6f} | {val['loss_geometry']:.6f} "
            f"| {val['loss_false_edit']:.6f} | {val['loss_missed_edit']:.6f} "
            f"| {val['loss_confidence']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- `best.pt`: `{checkpoints['best.pt']['sha256']}`",
            f"- `last.pt`: `{checkpoints['last.pt']['sha256']}`",
            "- Full loss history: `history.jsonl`, `history.csv`",
            "- Curves: `curves/training_curves.png`",
            "- Per-epoch predictions: `visualizations/epoch_*.png`",
            "- TensorBoard: `tensorboard/`",
            "- Input snapshot and environment: `provenance/`",
            "- Test predictions and bootstrap: `test_best/`, `test_last/`",
        ]
    )
    (report_dir / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    written = {
        "json": str(report_dir / "run_summary.json"),
        "markdown": str(report_dir / "run_summary.md"),
    }
    print(json.dumps(written, indent=2))


if __name__ == "__main__":
    main()
