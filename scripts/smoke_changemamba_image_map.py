"""Run a ChangeMamba image-map forward/backward smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("change_mamba_repo", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()

    sys.path.insert(0, str(args.change_mamba_repo))
    import torch
    import torch.nn.functional as F

    from changedetection.configs.config import _C
    from changedetection.models.ChangeMambaBCD import ChangeMambaBCD
    from changedetection.script.script_utils import get_vssm_kwargs

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = _C.clone()
    config.defrost()
    config.merge_from_file(str(args.config))
    config.freeze()

    model = ChangeMambaBCD(pretrained=None, **get_vssm_kwargs(config)).to(device)
    model.train()

    new_rgb = torch.rand(
        1, 3, args.image_size, args.image_size, device=device
    )
    old_mask = (
        torch.rand(1, 1, args.image_size, args.image_size, device=device) > 0.8
    ).float()
    old_map_rgb = old_mask.repeat(1, 3, 1, 1)
    target_change = (
        (new_rgb[:, 0] > 0.8).logical_xor(old_mask[:, 0].bool()).long()
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    logits = model(old_map_rgb, new_rgb)
    loss = F.cross_entropy(logits, target_change)
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    report = {
        "schema_version": "changemamba-image-map-smoke-v1",
        "source_commit": "9ce9cec13f9ea14bc0ad91f071577ec9b3a97983",
        "config": str(args.config),
        "device": str(device),
        "image_size": args.image_size,
        "input_contract": {
            "pre_data": "old editable mask repeated to three channels",
            "post_data": "new RGB image",
            "target": "binary changed-pixel mask",
        },
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "gradient_tensor_count": len(gradients),
        "all_gradients_finite": bool(
            gradients and all(torch.isfinite(item).all().item() for item in gradients)
        ),
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else 0.0
        ),
        "uses_real_data": False,
        "test_assets_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
