#!/usr/bin/env python3
"""Render one immutable selector config for the registered SN7 V5 run."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def render_config(
    base_config: Path,
    samples: Path,
    output: Path,
    run_dir: Path,
    *,
    seed: int,
) -> None:
    """Write a seed-specific config without mutating the registered base file."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite selector config: {output}")
    payload = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("selector base config must be a mapping")
    data = payload.get("data")
    training = payload.get("training")
    if not isinstance(data, dict) or not isinstance(training, dict):
        raise ValueError("selector base config is missing data or training mapping")
    payload["seed"] = seed
    data["samples"] = str(samples.resolve())
    training["device"] = "cuda:0"
    payload["output_dir"] = str(run_dir.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_config", type=Path)
    parser.add_argument("samples", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    render_config(
        args.base_config,
        args.samples,
        args.output,
        args.run_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
