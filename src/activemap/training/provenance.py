"""Immutable-enough training input and environment snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml

from activemap.updater_records import UpdaterSample


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_updater_provenance(
    output_dir: Path,
    *,
    config: dict[str, Any],
    config_path: Path,
    samples_path: Path,
    samples: list[UpdaterSample],
) -> None:
    provenance_dir = output_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    (provenance_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    shutil.copy2(samples_path, provenance_dir / "updater_samples.jsonl")
    companions: dict[str, dict[str, Any]] = {}
    for filename in ("summary.json", "audit.json"):
        source = samples_path.parent / filename
        if source.is_file():
            destination = provenance_dir / filename
            shutil.copy2(source, destination)
            companions[filename] = {
                "source": str(source.resolve()),
                "sha256": _sha256(source),
                "size_bytes": source.stat().st_size,
            }
    split_counts = Counter(sample.split for sample in samples)
    edit_counts = Counter(sample.edit_type.value for sample in samples)
    split_edit_counts = Counter(f"{sample.split}/{sample.edit_type.value}" for sample in samples)
    cuda_devices = []
    if torch.cuda.is_available():
        cuda_devices = [
            {
                "logical_index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": sys.argv,
        "working_directory": str(Path.cwd()),
        "config_source": str(config_path.resolve()),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_devices": cuda_devices,
        "dataset": {
            "manifest_source": str(samples_path.resolve()),
            "manifest_snapshot": str((provenance_dir / "updater_samples.jsonl").resolve()),
            "sha256": _sha256(samples_path),
            "size_bytes": samples_path.stat().st_size,
            "samples": len(samples),
            "split_counts": dict(sorted(split_counts.items())),
            "edit_counts": dict(sorted(edit_counts.items())),
            "split_edit_counts": dict(sorted(split_edit_counts.items())),
            "companions": companions,
        },
    }
    (provenance_dir / "run_provenance.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def archive_updater_run(output_dir: Path, archive_dir: Path) -> None:
    """Mirror compact training artifacts into the code-side outputs directory."""

    archive_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "best.pt",
        "best_val_loss.pt",
        "best_quality.pt",
        "best_safety.pt",
        "last.pt",
        "history.jsonl",
        "history.csv",
        "metrics.json",
        "state.json",
        "launch_command.txt",
        "train.log",
        "queue.log",
    ):
        source = output_dir / filename
        if source.is_file():
            shutil.copy2(source, archive_dir / filename)
    for dirname in (
        "provenance",
        "curves",
        "visualizations",
        "tensorboard",
        "checkpoints",
        "queue",
    ):
        source = output_dir / dirname
        if source.is_dir():
            shutil.copytree(source, archive_dir / dirname, dirs_exist_ok=True)
    (archive_dir / "ARCHIVED_FROM.txt").write_text(
        str(output_dir.resolve()) + "\n", encoding="utf-8"
    )
