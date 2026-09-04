"""Run one ActiveMap refinement request inside the isolated RSPrompter environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--overlap-threshold", type=float, default=0.10)
    return parser.parse_args()


def _select_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    coarse: np.ndarray,
    *,
    score_threshold: float,
    overlap_threshold: float,
) -> np.ndarray:
    selected: list[np.ndarray] = []
    overlap_scores: list[float] = []
    coarse_area = max(float(coarse.sum()), 1.0)
    for mask, score in zip(masks, scores, strict=True):
        candidate = np.asarray(mask, dtype=bool)
        overlap = float(np.logical_and(candidate, coarse).sum()) / coarse_area
        overlap_scores.append(overlap)
        if float(score) >= score_threshold and overlap >= overlap_threshold:
            selected.append(candidate)
    if selected:
        return np.logical_or.reduce(selected).astype(np.float32)
    if len(overlap_scores) and max(overlap_scores) > 0:
        index = int(np.argmax(overlap_scores))
        if float(scores[index]) >= score_threshold:
            return np.asarray(masks[index], dtype=np.float32)
    return coarse.astype(np.float32)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.repository.resolve()))
    from mmdet.apis import inference_detector, init_detector

    request = np.load(args.input)
    image = request["image"]
    coarse = request["coarse_mask"] >= 0.5
    model = init_detector(str(args.config), str(args.checkpoint), device=args.device)
    result = inference_detector(model, image)
    instances = result.pred_instances.cpu()
    if "masks" not in instances or len(instances) == 0:
        refined = coarse.astype(np.float32)
    else:
        refined = _select_masks(
            instances.masks.numpy(),
            instances.scores.numpy(),
            coarse,
            score_threshold=args.score_threshold,
            overlap_threshold=args.overlap_threshold,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, refined)


if __name__ == "__main__":
    main()
