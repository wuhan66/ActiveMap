"""Observable and controllable training-run state."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

from activemap.training.curves import render_training_curves


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class RunMonitor:
    """Writes live metrics and honors file-based pause/stop controls."""

    def __init__(self, output_dir: Path, settings: dict[str, Any] | None = None) -> None:
        self.output_dir = output_dir
        self.settings = settings or {}
        self.control_dir = output_dir / str(self.settings.get("control_dir", "control"))
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = output_dir / "history.jsonl"
        self.state_path = output_dir / "state.json"
        self.pause_path = self.control_dir / "PAUSE"
        self.stop_path = self.control_dir / "STOP"
        self.pause_poll_seconds = float(self.settings.get("pause_poll_seconds", 10.0))
        self.writer: Any | None = None
        self.tensorboard_error: str | None = None
        if bool(self.settings.get("tensorboard", True)):
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
            except (ImportError, ModuleNotFoundError) as exc:
                self.tensorboard_error = str(exc)

    def write_state(self, status: str, **fields: Any) -> None:
        payload = {
            "status": status,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "pid": os.getpid(),
            **fields,
        }
        if self.tensorboard_error:
            payload["tensorboard_error"] = self.tensorboard_error
        _atomic_json(self.state_path, payload)

    def record_epoch(self, record: dict[str, Any]) -> None:
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        records = []
        with self.history_path.open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        flattened: list[dict[str, Any]] = []
        for epoch_record in records:
            row: dict[str, Any] = {
                key: value
                for key, value in epoch_record.items()
                if key not in {"train", "val"} and isinstance(value, (int, float, str, bool))
            }
            for split in ("train", "val"):
                for name, value in epoch_record.get(split, {}).items():
                    row[f"{split}/{name}"] = value
            flattened.append(row)
        fieldnames = sorted(
            {key for row in flattened for key in row},
            key=lambda key: (key != "epoch", key),
        )
        csv_path = self.output_dir / "history.csv"
        temporary = csv_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flattened)
        os.replace(temporary, csv_path)
        render_training_curves(records, self.output_dir / "curves" / "training_curves.png")
        if self.writer is None:
            return
        step = int(record["epoch"])
        for split in ("train", "val"):
            metrics = record.get(split, {})
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{split}/{name}", float(value), step)
        learning_rate = record.get("learning_rate")
        if isinstance(learning_rate, (int, float)):
            self.writer.add_scalar("optimization/learning_rate", float(learning_rate), step)
        self.writer.flush()

    def should_stop(self) -> bool:
        return self.stop_path.exists()

    def wait_if_paused(self, *, epoch: int) -> bool:
        announced = False
        while self.pause_path.exists() and not self.should_stop():
            if not announced:
                self.write_state("paused", epoch=epoch)
                announced = True
            time.sleep(self.pause_poll_seconds)
        return self.should_stop()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Atomically save a torch checkpoint without importing torch at module import time."""

    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
