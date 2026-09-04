from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from activemap.nn.updater import PriorConditionedUNet, UpdaterConfig
from activemap.training.selector import resolve_device
from activemap.training.visualization import render_updater_progress
from activemap.updater_records import load_updater_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a class-balanced diagnostic panel for a promoted updater."
    )
    parser.add_argument("decision", type=Path)
    parser.add_argument("samples", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--count", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    decision: dict[str, Any] = json.loads(args.decision.read_text(encoding="utf-8"))
    checkpoint_name = decision.get("promoted_checkpoint")
    if not checkpoint_name:
        raise ValueError(f"decision has no promoted checkpoint: {args.decision}")
    run_dir = args.decision.parent
    checkpoint_path = run_dir / f"{checkpoint_name}.pt"
    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = PriorConditionedUNet(UpdaterConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    samples = load_updater_samples(args.samples, split=args.split)
    render_updater_progress(
        model,
        samples,
        device=device,
        output_path=args.output,
        count=args.count,
    )
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path.resolve()),
                "split": args.split,
                "sample_count": args.count,
                "output": str(args.output.resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
