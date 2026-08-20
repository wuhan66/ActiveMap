"""Optional SAM-Road road-probability backend.

The upstream project is intentionally loaded at runtime so its research stack does
not become a dependency of the core ActiveMap environment.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


class SAMRoadPredictor:
    """Expose the official SAM-Road mask head through SegmentationPredictor."""

    def __init__(
        self,
        repo_root: Path,
        config_path: Path,
        checkpoint: Path,
        sam_checkpoint: Path,
        *,
        device: str = "cuda",
        road_threshold: float | None = None,
        upstream_commit: str | None = None,
    ) -> None:
        for path in (repo_root, config_path, checkpoint, sam_checkpoint):
            if not path.exists():
                raise FileNotFoundError(path)
        if not 0.0 <= (road_threshold if road_threshold is not None else 0.5) <= 1.0:
            raise ValueError("road_threshold must be in [0, 1]")

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional environment
            raise RuntimeError("SAM-Road requires its isolated PyTorch environment") from exc

        repo_root = repo_root.resolve()
        import_roots = (repo_root, repo_root / "sam")
        for import_root in reversed(import_roots):
            import_text = str(import_root)
            if import_text not in sys.path:
                sys.path.insert(0, import_text)
        with contextlib.redirect_stdout(io.StringIO()):
            upstream_utils = importlib.import_module("utils")
            upstream_model = importlib.import_module("model")
            for module in (upstream_utils, upstream_model):
                module_path = Path(module.__file__).resolve()
                if repo_root not in module_path.parents:
                    raise RuntimeError(
                        f"upstream module collision: {module.__name__} from {module_path}"
                    )
            config = upstream_utils.load_config(str(config_path.resolve()))
            config.SAM_CKPT_PATH = str(sam_checkpoint.resolve())
            model = upstream_model.SAMRoad(config)
            payload = torch.load(str(checkpoint.resolve()), map_location="cpu")
            state_dict = payload.get("state_dict", payload)
            model.load_state_dict(state_dict, strict=True)

        self._torch = torch
        self.model = model.eval().to(torch.device(device))
        self.device = torch.device(device)
        self.patch_size = int(config.PATCH_SIZE)
        self.road_threshold = self.resolve_road_threshold(
            payload,
            config_threshold=float(config.ROAD_THRESHOLD),
            explicit_threshold=road_threshold,
        )
        self.upstream_commit = upstream_commit or "unverified"
        self.checkpoint_name = checkpoint.name

    @staticmethod
    def resolve_road_threshold(
        payload: Any,
        *,
        config_threshold: float,
        explicit_threshold: float | None,
    ) -> float:
        if explicit_threshold is not None:
            threshold = float(explicit_threshold)
        else:
            metadata = payload.get("activemap_adapter", {}) if isinstance(payload, Mapping) else {}
            checkpoint_threshold = (
                metadata.get("road_threshold") if isinstance(metadata, Mapping) else None
            )
            threshold = float(
                config_threshold if checkpoint_threshold is None else checkpoint_threshold
            )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("resolved road threshold must be in [0, 1]")
        return threshold

    @staticmethod
    def _rgb_255(image: np.ndarray) -> np.ndarray:
        array = np.asarray(image, dtype=np.float32)
        if array.ndim != 3 or array.shape[0] != 3:
            raise ValueError("SAM-Road expects a three-band CHW image")
        if not np.all(np.isfinite(array)):
            raise ValueError("SAM-Road input contains non-finite pixels")
        if float(np.max(array)) <= 1.5:
            array = array * 255.0
        return np.clip(array, 0.0, 255.0)

    def predict(
        self, image: np.ndarray, prior_mask: np.ndarray
    ) -> Mapping[str, Any]:
        torch = self._torch
        rgb = self._rgb_255(image)
        original_shape = tuple(int(value) for value in rgb.shape[1:])
        tensor = torch.from_numpy(rgb).unsqueeze(0).to(self.device)
        if original_shape != (self.patch_size, self.patch_size):
            tensor = torch.nn.functional.interpolate(
                tensor,
                size=(self.patch_size, self.patch_size),
                mode="bilinear",
                align_corners=False,
            )
        tensor = tensor.permute(0, 2, 3, 1).contiguous()
        with torch.inference_mode():
            mask_scores, _ = self.model.infer_masks_and_img_features(tensor)
            road = mask_scores[..., 1].unsqueeze(1)
            if original_shape != (self.patch_size, self.patch_size):
                road = torch.nn.functional.interpolate(
                    road,
                    size=original_shape,
                    mode="bilinear",
                    align_corners=False,
                )
        probability = road[0, 0].float().cpu().numpy()
        if probability.shape != np.asarray(prior_mask).squeeze().shape:
            raise ValueError("SAM-Road output and prior mask shapes do not match")
        return {
            "mask_probability": probability,
            "backend_version": self.upstream_commit,
            "model_checkpoint": self.checkpoint_name,
            "road_threshold": self.road_threshold,
        }
