from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conservatively interpolate two compatible PyTorch checkpoints."
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--adapter-weight",
        type=float,
        required=True,
        help="Interpolation coefficient in [0, 1]; 0 preserves base.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.adapter_weight <= 1.0:
        raise ValueError("--adapter-weight must be in [0, 1]")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    base = torch.load(args.base, map_location="cpu")
    adapter = torch.load(args.adapter, map_location="cpu")
    base_state = base["state_dict"]
    adapter_state = adapter["state_dict"]
    if base_state.keys() != adapter_state.keys():
        raise ValueError("checkpoint state-dict keys differ")

    alpha = args.adapter_weight
    blended = {}
    interpolated = 0
    for key, base_value in base_state.items():
        adapter_value = adapter_state[key]
        if base_value.shape != adapter_value.shape:
            raise ValueError(f"shape mismatch for {key}")
        if torch.is_floating_point(base_value):
            blended[key] = (1.0 - alpha) * base_value + alpha * adapter_value
            interpolated += 1
        else:
            blended[key] = base_value

    base["state_dict"] = blended
    base.setdefault("meta", {})["activemap_weight_interpolation"] = {
        "base": str(args.base),
        "adapter": str(args.adapter),
        "adapter_weight": alpha,
        "interpolated_tensors": interpolated,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, args.output)
    print(f"wrote {args.output} ({interpolated} floating tensors interpolated)")


if __name__ == "__main__":
    main()
