from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_summary(path: Path) -> dict[str, Any]:
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    metrics = checkpoint["val_metrics"]
    val = {
        name: float(metrics[name])
        for name in (
            "loss",
            "iou",
            "edit_accuracy",
            "false_edit_rate",
            "missed_edit_rate",
            "delete_recall",
        )
    }
    for name in ("added_change_iou", "removed_change_iou"):
        if name in metrics:
            val[name] = float(metrics[name])
    if "added_change_iou" in val and "removed_change_iou" in val:
        total = val["added_change_iou"] + val["removed_change_iou"]
        val["temporal_change_harmonic_iou"] = (
            2.0 * val["added_change_iou"] * val["removed_change_iou"] / total if total else 0.0
        )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "epoch": int(checkpoint["epoch"]),
        "seed": int(checkpoint["seed"]),
        "learning_rate": float(checkpoint["learning_rate"]),
        "val": val,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inventory validation-selected updater checkpoints without test access."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    report = {
        "run_dir": str(run_dir),
        "selection_scope": "validation_only",
        "checkpoints": {
            name: _checkpoint_summary(run_dir / f"{name}.pt")
            for name in ("best_quality", "best_tradeoff", "best_safety", "best_val_loss")
            if (run_dir / f"{name}.pt").is_file()
        },
    }
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
