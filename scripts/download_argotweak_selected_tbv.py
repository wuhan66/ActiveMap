from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


CAMERAS = (
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_rear_left",
    "ring_rear_right",
    "ring_side_left",
    "ring_side_right",
)
REQUIRED_FILES = (
    "city_SE3_egovehicle.feather",
    "calibration/egovehicle_SE3_sensor.feather",
    "calibration/intrinsics.feather",
)


def _audit(segment_root: Path) -> dict[str, Any]:
    missing = [relative for relative in REQUIRED_FILES if not (segment_root / relative).is_file()]
    camera_counts = {
        camera: len(list((segment_root / "sensors" / "cameras" / camera).glob("*.jpg")))
        for camera in CAMERAS
    }
    if missing or any(count == 0 for count in camera_counts.values()):
        raise RuntimeError(
            f"incomplete TbV segment {segment_root.name}: missing={missing}, cameras={camera_counts}"
        )
    sampled_counts = {
        camera: (count + 9) // 10
        for camera, count in camera_counts.items()
    }
    synchronized_count = max(sampled_counts["ring_front_center"] - 1, 0)
    if any(
        count < synchronized_count
        for camera, count in sampled_counts.items()
        if camera != "ring_front_center"
    ):
        raise RuntimeError(
            f"TbV camera is shorter than the official front-center iteration in "
            f"{segment_root.name}: raw={camera_counts}, sampled={sampled_counts}"
        )
    dropped_bundles = {
        camera: count - synchronized_count for camera, count in sampled_counts.items()
    }
    return {
        "schema_version": "activemap-argotweak-tbv-download-v1",
        "segment_id": segment_root.name,
        "camera_counts": camera_counts,
        "official_sampled_counts": sampled_counts,
        "synchronized_bundle_count": synchronized_count,
        "dropped_bundles_for_synchronization": dropped_bundles,
        "required_files": list(REQUIRED_FILES),
        "test_assets_read": False,
    }


def _download(row: dict[str, Any], target_root: Path, aws_bin: str) -> dict[str, Any]:
    segment_id = row["segment_id"]
    segment_root = target_root / segment_id
    marker = segment_root / ".activemap_download_complete.json"
    if marker.is_file():
        audit = _audit(segment_root)
        return {**audit, "split": row["split"], "status": "already_complete"}
    segment_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            aws_bin,
            "s3",
            "sync",
            row["s3_prefix"],
            str(segment_root),
            "--no-sign-request",
            "--only-show-errors",
        ],
        check=True,
    )
    audit = _audit(segment_root)
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    temporary.replace(marker)
    return {**audit, "split": row["split"], "status": "downloaded"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify a frozen ArgoTweak TbV subset.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--aws-bin", default="aws")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split", choices=("train", "val"))
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    if any(row.get("split") not in {"train", "val"} for row in rows):
        raise ValueError("download manifest may contain only train/val rows")
    if args.split is not None:
        rows = [row for row in rows if row["split"] == args.split]
    args.target_root.mkdir(parents=True, exist_ok=True)
    completed = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_download, row, args.target_root, args.aws_bin): row
            for row in rows
        }
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                row = futures[future]
                failures.append({"split": row["split"], "segment_id": row["segment_id"], "error": str(exc)})
                print(
                    f"[failed] {row['split']} {row['segment_id']}: {exc}",
                    flush=True,
                )
                continue
            completed.append(result)
            print(
                f"[{len(completed):02d}/{len(rows):02d}] {result['split']} "
                f"{result['segment_id']} {result['status']}",
                flush=True,
            )
    completed.sort(key=lambda row: (row["split"], row["segment_id"]))
    summary = {
        "schema_version": "activemap-argotweak-tbv-download-summary-v1",
        "manifest": str(args.manifest.resolve()),
        "target_root": str(args.target_root.resolve()),
        "segments": len(completed),
        "split_counts": {
            split: sum(row["split"] == split for row in completed) for split in ("train", "val")
        },
        "camera_images": sum(sum(row["camera_counts"].values()) for row in completed),
        "rows": completed,
        "failures": failures,
        "test_assets_read": False,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.summary.with_suffix(args.summary.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.summary)
    print(json.dumps({key: summary[key] for key in ("segments", "split_counts", "camera_images")}))
    if failures:
        raise RuntimeError(f"{len(failures)} TbV segments failed download or validation")


if __name__ == "__main__":
    main()
