"""COCO bridge and isolated training command for RSPrompter."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from activemap.models import EditOperation
from activemap.updater_records import UpdaterSample, load_updater_samples


def _channels_first(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError("updater image must have three dimensions")
    if array.shape[0] in {1, 3, 4}:
        return array
    if array.shape[-1] in {1, 3, 4}:
        return np.moveaxis(array, -1, 0)
    raise ValueError(f"cannot infer channel axis from {array.shape}")


def _rgb_uint8(array: np.ndarray) -> np.ndarray:
    channels = _channels_first(array).astype(np.float32)
    if channels.shape[0] == 1:
        channels = np.repeat(channels, 3, axis=0)
    channels = channels[:3]
    finite = channels[np.isfinite(channels)]
    if finite.size == 0:
        raise ValueError("image has no finite values")
    if float(np.max(finite)) <= 1.0:
        channels *= 255.0
    elif float(np.max(finite)) > 255.0:
        low, high = np.percentile(finite, [1, 99])
        channels = 255.0 * np.clip((channels - low) / max(float(high - low), 1e-6), 0, 1)
    return np.moveaxis(np.clip(np.nan_to_num(channels), 0, 255).astype(np.uint8), 0, -1)


def _mask(path: str) -> np.ndarray:
    array = np.load(path)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"mask must be 2-D or [1,H,W], got {array.shape}")
    return array >= 0.5


def _safe_stem(sample: UpdaterSample) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "-", sample.sample_id).strip("-")[:80]
    digest = hashlib.sha1(sample.sample_id.encode("utf-8")).hexdigest()[:10]
    return f"{readable}-{digest}"


def _uncompressed_rle(mask: np.ndarray) -> dict[str, Any]:
    flattened = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    counts: list[int] = []
    previous = 0
    run_length = 0
    for value in flattened:
        current = int(value)
        if current == previous:
            run_length += 1
        else:
            counts.append(run_length)
            run_length = 1
            previous = current
    counts.append(run_length)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def _mask_annotations(
    mask: np.ndarray,
    *,
    image_id: int,
    first_annotation_id: int,
    category_id: int,
    minimum_area: float,
) -> tuple[list[dict[str, Any]], int]:
    annotations: list[dict[str, Any]] = []
    annotation_id = first_annotation_id
    labels, component_count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    for component_id in range(1, component_count + 1):
        component = labels == component_id
        area = int(np.sum(component))
        if area < minimum_area:
            continue
        y_coordinates, x_coordinates = np.nonzero(component)
        x_min, x_max = int(np.min(x_coordinates)), int(np.max(x_coordinates))
        y_min, y_max = int(np.min(y_coordinates)), int(np.max(y_coordinates))
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": _uncompressed_rle(component),
                "area": area,
                "bbox": [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1],
                "iscrowd": 0,
            }
        )
        annotation_id += 1
    return annotations, annotation_id


def _export_split(
    samples: list[UpdaterSample],
    output_dir: Path,
    *,
    split: str,
    include_delete_negatives: bool,
    minimum_area: float,
    max_samples: int | None,
    category_ids: dict[str, int],
    categories: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = [
        sample
        for sample in samples
        if sample.split == split
        and (include_delete_negatives or sample.edit_type != EditOperation.DELETE)
    ]
    selected = sorted(selected, key=lambda item: item.sample_id)
    if max_samples is not None:
        selected = selected[:max_samples]
    image_dir = output_dir / "images" / split
    annotation_dir = output_dir / "annotations"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    empty_count = 0

    for image_id, sample in enumerate(selected, start=1):
        image = _rgb_uint8(np.load(sample.image_path))
        target = _mask(sample.target_mask_path)
        if sample.valid_mask_path is not None:
            target &= _mask(sample.valid_mask_path)
        if target.shape != image.shape[:2]:
            raise ValueError(f"image/mask shape mismatch for {sample.sample_id}")
        filename = f"{_safe_stem(sample)}.png"
        Image.fromarray(image, mode="RGB").save(image_dir / filename)
        images.append(
            {
                "id": image_id,
                "file_name": filename,
                "width": image.shape[1],
                "height": image.shape[0],
                "activemap_sample_id": sample.sample_id,
                "aoi_id": sample.aoi_id,
                "edit_type": sample.edit_type.value,
                "dataset_name": sample.dataset_name,
                "geometry_family": sample.geometry_family,
                "supervision_type": sample.supervision_type,
            }
        )
        sample_annotations, annotation_id = _mask_annotations(
            target,
            image_id=image_id,
            first_annotation_id=annotation_id,
            category_id=category_ids[sample.geometry_family],
            minimum_area=minimum_area,
        )
        annotations.extend(sample_annotations)
        empty_count += int(not sample_annotations)

    payload = {
        "info": {
            "description": "ActiveMap crops exported for isolated RSPrompter training",
            "version": "1.0",
            "split": split,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    annotation_path = annotation_dir / f"{split}.json"
    annotation_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    return {
        "split": split,
        "images": len(images),
        "annotations": len(annotations),
        "empty_images": empty_count,
        "annotation_path": str(annotation_path.resolve()),
        "image_dir": str(image_dir.resolve()),
    }


def export_rsprompter_dataset(
    samples_path: Path,
    output_dir: Path,
    *,
    include_delete_negatives: bool = True,
    minimum_area: float = 4.0,
    max_samples_per_split: int | None = None,
    splits: tuple[str, ...] = ("train", "val"),
) -> dict[str, Any]:
    if not splits or len(splits) != len(set(splits)):
        raise ValueError("splits must contain unique train/val/test values")
    unknown = set(splits) - {"train", "val", "test"}
    if unknown:
        raise ValueError(f"unsupported RSPrompter export splits: {sorted(unknown)}")
    samples = load_updater_samples(samples_path)
    known_categories = {
        "polygon": ("building", "structure"),
        "polyline": ("road", "transportation"),
    }
    families = [
        family for family in ("polygon", "polyline") if any(
            sample.geometry_family == family for sample in samples
        )
    ]
    if not families:
        raise ValueError("RSPrompter export has no supported geometry families")
    category_ids = {family: index for index, family in enumerate(families, start=1)}
    categories = [
        {
            "id": category_ids[family],
            "name": known_categories[family][0],
            "supercategory": known_categories[family][1],
        }
        for family in families
    ]
    summaries = [
        _export_split(
            samples,
            output_dir,
            split=split,
            include_delete_negatives=include_delete_negatives,
            minimum_area=minimum_area,
            max_samples=max_samples_per_split,
            category_ids=category_ids,
            categories=categories,
        )
        for split in splits
    ]
    summary = {
        "source": str(samples_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "include_delete_negatives": include_delete_negatives,
        "minimum_area": minimum_area,
        "categories": categories,
        "requested_splits": list(splits),
        "test_assets_read": "test" in splits,
        "splits": summaries,
    }
    (output_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def audit_rsprompter_dataset(
    output_dir: Path,
    *,
    require_test_free: bool = True,
) -> dict[str, Any]:
    """Validate a COCO export independently of its source manifest."""
    summary_path = output_dir / "export_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if require_test_free and summary.get("test_assets_read") is not False:
        errors.append("export is not test-free")
    categories = summary.get("categories")
    if not isinstance(categories, list) or not categories:
        errors.append("summary has no categories")
        categories = []
    category_ids = {int(item["id"]) for item in categories}
    sample_ids_by_split: dict[str, set[str]] = {}
    aoi_ids_by_split: dict[str, set[str]] = {}
    split_reports: list[dict[str, Any]] = []

    for split_summary in summary.get("splits", []):
        split = str(split_summary["split"])
        annotation_path = Path(split_summary["annotation_path"])
        image_dir = Path(split_summary["image_dir"])
        if not annotation_path.is_file() or not image_dir.is_dir():
            errors.append(f"{split}: missing annotation file or image directory")
            continue
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        if payload.get("categories") != categories:
            errors.append(f"{split}: categories differ from export summary")
        images = payload.get("images", [])
        annotations = payload.get("annotations", [])
        image_ids = [int(item["id"]) for item in images]
        if len(image_ids) != len(set(image_ids)):
            errors.append(f"{split}: duplicate image ids")
        image_by_id = {int(item["id"]): item for item in images}
        filenames = [str(item["file_name"]) for item in images]
        if len(filenames) != len(set(filenames)):
            errors.append(f"{split}: duplicate image filenames")
        actual_files = {path.name for path in image_dir.glob("*.png")}
        if actual_files != set(filenames):
            errors.append(f"{split}: referenced and on-disk PNG sets differ")
        sample_ids = {str(item["activemap_sample_id"]) for item in images}
        if len(sample_ids) != len(images):
            errors.append(f"{split}: duplicate ActiveMap sample ids")
        sample_ids_by_split[split] = sample_ids
        aoi_ids_by_split[split] = {
            str(item["aoi_id"]) for item in images if item.get("aoi_id") is not None
        }
        annotation_counts = {image_id: 0 for image_id in image_ids}

        for image in images:
            path = image_dir / str(image["file_name"])
            try:
                with Image.open(path) as handle:
                    handle.verify()
                with Image.open(path) as handle:
                    actual_size = handle.size
            except Exception as exc:
                errors.append(f"{split}: corrupt image {path.name}: {exc}")
                continue
            expected_size = (int(image["width"]), int(image["height"]))
            if actual_size != expected_size:
                errors.append(f"{split}: size mismatch for {path.name}")

        annotation_ids: set[int] = set()
        for annotation in annotations:
            annotation_id = int(annotation["id"])
            image_id = int(annotation["image_id"])
            if annotation_id in annotation_ids:
                errors.append(f"{split}: duplicate annotation id {annotation_id}")
            annotation_ids.add(annotation_id)
            image = image_by_id.get(image_id)
            if image is None:
                errors.append(f"{split}: annotation references missing image {image_id}")
                continue
            annotation_counts[image_id] += 1
            if int(annotation["category_id"]) not in category_ids:
                errors.append(f"{split}: annotation has unknown category")
            width, height = float(image["width"]), float(image["height"])
            bbox = np.asarray(annotation.get("bbox", []), dtype=np.float64)
            if (
                bbox.shape != (4,)
                or not np.all(np.isfinite(bbox))
                or bbox[2] <= 0
                or bbox[3] <= 0
                or bbox[0] < 0
                or bbox[1] < 0
                or bbox[0] + bbox[2] > width + 1e-6
                or bbox[1] + bbox[3] > height + 1e-6
            ):
                errors.append(f"{split}: invalid bbox in annotation {annotation_id}")
            if float(annotation.get("area", 0.0)) <= 0:
                errors.append(f"{split}: non-positive area in annotation {annotation_id}")
            segmentation = annotation.get("segmentation")
            if not isinstance(segmentation, dict):
                errors.append(f"{split}: annotation {annotation_id} is not RLE encoded")
                continue
            size = segmentation.get("size")
            counts = segmentation.get("counts")
            expected_size = [int(height), int(width)]
            if size != expected_size or not isinstance(counts, list) or not counts:
                errors.append(f"{split}: invalid RLE header in annotation {annotation_id}")
                continue
            if any(not isinstance(value, int) or value < 0 for value in counts):
                errors.append(f"{split}: invalid RLE counts in annotation {annotation_id}")
                continue
            if sum(counts) != int(width * height):
                errors.append(f"{split}: RLE size mismatch in annotation {annotation_id}")
            encoded_area = sum(counts[1::2])
            if abs(encoded_area - float(annotation.get("area", 0.0))) > 1e-6:
                errors.append(f"{split}: RLE area mismatch in annotation {annotation_id}")

        empty_images = sum(count == 0 for count in annotation_counts.values())
        if len(images) != int(split_summary["images"]):
            errors.append(f"{split}: image count differs from summary")
        if len(annotations) != int(split_summary["annotations"]):
            errors.append(f"{split}: annotation count differs from summary")
        if empty_images != int(split_summary["empty_images"]):
            errors.append(f"{split}: empty-image count differs from summary")
        split_reports.append(
            {
                "split": split,
                "images": len(images),
                "annotations": len(annotations),
                "empty_images": empty_images,
                "aoi_count": len(aoi_ids_by_split[split]),
            }
        )

    splits = sorted(sample_ids_by_split)
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1 :]:
            if sample_ids_by_split[left] & sample_ids_by_split[right]:
                errors.append(f"sample id leakage between {left} and {right}")
            if aoi_ids_by_split[left] & aoi_ids_by_split[right]:
                errors.append(f"AOI leakage between {left} and {right}")

    report = {
        "schema_version": "activemap-rsprompter-coco-audit-v1",
        "valid": not errors,
        "require_test_free": require_test_free,
        "test_assets_read": summary.get("test_assets_read"),
        "categories": categories,
        "mask_encoding": "coco_uncompressed_rle",
        "splits": split_reports,
        "errors": errors,
    }
    (output_dir / "audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if errors:
        raise ValueError("RSPrompter COCO audit failed: " + "; ".join(errors[:10]))
    return report


def rsprompter_train_command(
    *,
    python: Path,
    repository: Path,
    config: Path,
    data_root: Path,
    work_dir: Path,
    resume: bool = False,
) -> list[str]:
    command = [
        str(python),
        str(repository / "tools" / "train.py"),
        str(config),
        "--cfg-options",
        f"data_root={data_root}",
        f"work_dir={work_dir}",
    ]
    if resume:
        command.extend(["resume=True"])
    return command


def run_rsprompter_training(
    command: list[str], *, repository: Path, log_path: Path
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=repository,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(process.returncode)


@dataclass(frozen=True)
class RSPrompterRefinementConfig:
    python: Path
    repository: Path
    model_config: Path
    checkpoint: Path
    adapter_script: Path
    work_dir: Path
    device: str = "cuda:0"
    score_threshold: float = 0.35
    overlap_threshold: float = 0.10


class RSPrompterRefiner:
    """Isolated RSPrompter subprocess with a coarse-mask fallback."""

    def __init__(self, config: RSPrompterRefinementConfig) -> None:
        self.config = config

    def refine(
        self,
        image: np.ndarray,
        coarse_mask: np.ndarray,
        prior_mask: np.ndarray,
        *,
        edit_type: EditOperation,
    ) -> np.ndarray:
        coarse = np.asarray(coarse_mask, dtype=np.float32).squeeze()
        if coarse.ndim != 2:
            raise ValueError("coarse_mask must resolve to a 2-D array")
        if edit_type not in {EditOperation.ADD, EditOperation.RESHAPE}:
            return coarse
        for path, label in (
            (self.config.python, "RSPrompter Python"),
            (self.config.repository, "RSPrompter repository"),
            (self.config.model_config, "RSPrompter model config"),
            (self.config.checkpoint, "RSPrompter checkpoint"),
            (self.config.adapter_script, "ActiveMap adapter script"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.config.work_dir) as directory:
            request_path = Path(directory) / "request.npz"
            response_path = Path(directory) / "refined.npy"
            np.savez_compressed(
                request_path,
                image=_rgb_uint8(np.asarray(image)),
                coarse_mask=coarse,
                prior_mask=np.asarray(prior_mask, dtype=np.float32).squeeze(),
                edit_type=edit_type.value,
            )
            command = [
                str(self.config.python),
                str(self.config.adapter_script),
                "--repository",
                str(self.config.repository),
                "--config",
                str(self.config.model_config),
                "--checkpoint",
                str(self.config.checkpoint),
                "--input",
                str(request_path),
                "--output",
                str(response_path),
                "--device",
                self.config.device,
                "--score-threshold",
                str(self.config.score_threshold),
                "--overlap-threshold",
                str(self.config.overlap_threshold),
            ]
            process = subprocess.run(
                command,
                cwd=self.config.repository,
                capture_output=True,
                text=True,
                check=False,
            )
            if process.returncode != 0:
                message = process.stderr.strip() or process.stdout.strip()
                raise RuntimeError(f"RSPrompter refinement failed: {message[-2000:]}")
            if not response_path.is_file():
                raise RuntimeError("RSPrompter adapter did not write a refined mask")
            refined = np.load(response_path).astype(np.float32).squeeze()
        if refined.shape != coarse.shape or not np.isfinite(refined).all():
            raise ValueError("RSPrompter returned an invalid mask")
        return np.clip(refined, 0.0, 1.0)
