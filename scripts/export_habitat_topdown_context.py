#!/usr/bin/env python3
"""Export a Habitat navmesh top-down context for qualitative visualization."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_online_helpers() -> Any:
    script = Path(__file__).with_name("run_habitat_online_mapping_pilot.py")
    spec = importlib.util.spec_from_file_location("habitat_online_mapping", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load online mapping helpers from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--height", type=float, required=True)
    parser.add_argument("--meters-per-pixel", type=float, default=0.025)
    parser.add_argument("--gpu-id", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.scene.is_file() or args.meters_per_pixel <= 0:
        raise ValueError("scene must exist and meters-per-pixel must be positive")

    online = load_online_helpers()
    habitat = online.load_habitat_helpers()
    simulator = habitat.build_simulator(
        scene_config=None,
        scene_id=str(args.scene.resolve()),
        gpu_id=args.gpu_id,
        resolution=64,
    )
    args.output.mkdir(parents=True)
    try:
        pathfinder = simulator.pathfinder
        lower, upper = pathfinder.get_bounds()
        topdown = np.asarray(
            pathfinder.get_topdown_view(args.meters_per_pixel, args.height), dtype=np.uint8
        )
        np.save(args.output / "navigability.npy", topdown)
        metadata = {
            "schema_version": "activemap-habitat-topdown-context-v1",
            "scene": str(args.scene.resolve()),
            "height": args.height,
            "meters_per_pixel": args.meters_per_pixel,
            "lower_bound_xyz": [float(value) for value in lower],
            "upper_bound_xyz": [float(value) for value in upper],
            "shape": list(topdown.shape),
            "navigable_fraction": float(np.mean(topdown > 0)),
            "visualization_only": True,
        }
        (args.output / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
