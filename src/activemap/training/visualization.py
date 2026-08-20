"""Fixed-sample visual diagnostics for updater training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image, ImageDraw

from activemap.models import EditOperation
from activemap.nn.updater import operation_probabilities
from activemap.training.updater_data import UpdaterDataset
from activemap.updater_records import UpdaterSample


def _rgb(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().float().numpy()
    if array.shape[0] == 1:
        array = np.repeat(array, 3, axis=0)
    return np.moveaxis(np.clip(array[:3], 0.0, 1.0), 0, -1)


def _gray(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().float().squeeze().numpy()
    return np.repeat(np.clip(array, 0.0, 1.0)[..., None], 3, axis=-1)


def _overlay(image: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    output = image.copy()
    prediction_mask = prediction >= 0.5
    target_mask = target >= 0.5
    output[target_mask, 1] = 1.0
    output[prediction_mask, 0] = 1.0
    output[prediction_mask | target_mask] *= 0.75
    output[target_mask, 1] = 1.0
    output[prediction_mask, 0] = 1.0
    return output


def _select_visualization_samples(
    samples: list[UpdaterSample], count: int
) -> list[UpdaterSample]:
    """Choose deterministic, class-balanced samples spread across each class bucket."""

    if count <= 0:
        return []
    buckets = {
        operation: sorted(
            (sample for sample in samples if sample.edit_type == operation),
            key=lambda sample: sample.sample_id,
        )
        for operation in EditOperation
    }
    selected: list[UpdaterSample] = []
    active = [operation for operation in EditOperation if buckets[operation]]
    while len(selected) < count and active:
        next_active = []
        for operation in active:
            bucket = buckets[operation]
            already_selected = sum(item.edit_type == operation for item in selected)
            if already_selected >= len(bucket):
                continue
            position = int(
                round(already_selected * (len(bucket) - 1) / max(1, count // len(active)))
            )
            position = min(position, len(bucket) - 1)
            candidate = bucket[position]
            if candidate not in selected:
                selected.append(candidate)
            if already_selected + 1 < len(bucket):
                next_active.append(operation)
            if len(selected) >= count:
                break
        active = next_active
    return selected


def _zoom_box(
    prior: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    minimum_size: int = 48,
) -> tuple[int, int, int, int]:
    """Return a square object-centered crop shared by every rendered panel."""

    signal = (prior >= 0.5) | (target >= 0.5)
    if not signal.any():
        signal = prediction >= 0.5
    height, width = signal.shape
    if not signal.any():
        return 0, 0, width, height
    ys, xs = np.nonzero(signal)
    extent = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    size = min(max(minimum_size, int(np.ceil(extent * 2.5))), min(height, width))
    center_x = float(xs.min() + xs.max() + 1) / 2.0
    center_y = float(ys.min() + ys.max() + 1) / 2.0
    left = max(0, min(width - size, int(round(center_x - size / 2.0))))
    top = max(0, min(height - size, int(round(center_y - size / 2.0))))
    return left, top, left + size, top + size


def _crop_resize(panel: np.ndarray, box: tuple[int, int, int, int], tile: int) -> Image.Image:
    image = Image.fromarray((np.clip(panel, 0, 1) * 255).astype(np.uint8))
    return image.crop(box).resize((tile, tile), Image.Resampling.NEAREST)


def render_updater_progress(
    model: Any,
    samples: list[UpdaterSample],
    *,
    device: torch.device,
    output_path: Path,
    count: int = 6,
) -> None:
    """Render fixed, class-balanced, object-centered validation diagnostics."""

    selected = _select_visualization_samples(samples, count)
    if not selected:
        return
    temporal_pair_input = bool(getattr(model.config, "temporal_pair_input", False))
    dataset = UpdaterDataset(selected, temporal_pair_input=temporal_pair_input)
    rows: list[tuple[dict[str, Any], dict[str, torch.Tensor]]] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset[index]
            image = item["image"]
            prior = item["prior_mask"]
            if not isinstance(image, torch.Tensor) or not isinstance(prior, torch.Tensor):
                continue
            outputs = model(image[None].to(device), prior[None].to(device))
            rows.append((item, outputs))
    model.train(was_training)

    tile = 192
    label_height = 34
    columns = 6 if temporal_pair_input else 5
    canvas = Image.new("RGB", (tile * columns, len(rows) * (tile + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    headings = (
        ("RGB t-1", "RGB t", "prior", "prediction", "target", "overlay")
        if temporal_pair_input
        else ("RGB", "prior", "prediction", "target", "overlay")
    )
    edit_names = [operation.value for operation in EditOperation]
    for row_index, (item, outputs) in enumerate(rows):
        image_tensor = cast(torch.Tensor, item["image"])
        prior_tensor = cast(torch.Tensor, item["prior_mask"])
        target_tensor = cast(torch.Tensor, item["target_mask"])
        edit_target_tensor = cast(torch.Tensor, item["edit_target"])
        display_image = _rgb(image_tensor[3:6] if temporal_pair_input else image_tensor)
        prior_image = _rgb(image_tensor[:3]) if temporal_pair_input else None
        display_prior = _gray(prior_tensor)
        target = target_tensor.detach().cpu().float().squeeze().numpy()
        probability = torch.sigmoid(outputs["segmentation_logits"])[0, 0].cpu().numpy()
        prediction = np.repeat(probability[..., None], 3, axis=-1)
        target_rgb = np.repeat(target[..., None], 3, axis=-1)
        prior_array = prior_tensor.detach().cpu().float().squeeze().numpy()
        box = _zoom_box(prior_array, target, probability)
        panels = (
            (
                prior_image,
                display_image,
                display_prior,
                prediction,
                target_rgb,
                _overlay(display_image, probability, target),
            )
            if temporal_pair_input
            else (
                display_image,
                display_prior,
                prediction,
                target_rgb,
                _overlay(display_image, probability, target),
            )
        )
        y = row_index * (tile + label_height)
        for column, (heading, panel) in enumerate(zip(headings, panels, strict=True)):
            panel_image = _crop_resize(panel, box, tile)
            canvas.paste(panel_image, (column * tile, y))
            draw.text(
                (column * tile + 3, y + 3),
                heading,
                fill="white",
                stroke_width=1,
                stroke_fill="black",
            )
        predicted_edit = int(
            torch.argmax(
                operation_probabilities(outputs, prior_tensor[None].to(device)), dim=-1
            )[0].cpu()
        )
        confidence = float(torch.sigmoid(outputs["confidence_logits"])[0].cpu())
        target_edit = edit_names[int(edit_target_tensor)]
        caption = (
            f"{item['sample_id']} | target={target_edit} | "
            f"pred={edit_names[predicted_edit]} | conf={confidence:.3f}"
        )
        draw.text((3, y + tile + 4), caption[:160], fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
