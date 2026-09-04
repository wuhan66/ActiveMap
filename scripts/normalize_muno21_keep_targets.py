"""Make MUNO21 no-change targets identical to the pre-change base map."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from activemap.models import EditOperation
from activemap.updater_records import UpdaterSample


def _resolved(parent: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else parent / path


def normalize_keep_targets(manifest: Path, report: Path) -> dict[str, object]:
    parent = manifest.parent
    records: list[UpdaterSample] = []
    repaired = 0
    copied_bytes = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = UpdaterSample.model_validate_json(line)
        if (
            sample.dataset_name == "muno21"
            and sample.supervision_type == "full_scene_temporal"
            and sample.edit_type == EditOperation.KEEP
        ):
            prior_path = _resolved(parent, sample.prior_mask_path)
            target_path = _resolved(parent, sample.target_mask_path)
            temporary_target = target_path.with_suffix(target_path.suffix + ".tmp")
            shutil.copyfile(prior_path, temporary_target)
            os.replace(temporary_target, target_path)
            metadata = dict(sample.source_metadata)
            metadata["target_graph"] = metadata.get("prior_graph")
            metadata["keep_target_policy"] = "identity_2013_no_physical_change"
            sample = sample.model_copy(
                update={
                    "target_geometry": sample.prior_geometry,
                    "source_metadata": metadata,
                }
            )
            repaired += 1
            copied_bytes += target_path.stat().st_size
        records.append(sample)

    temporary_manifest = manifest.with_suffix(manifest.suffix + ".tmp")
    with temporary_manifest.open("w", encoding="utf-8") as handle:
        for sample in records:
            handle.write(sample.model_dump_json() + "\n")
    os.replace(temporary_manifest, manifest)
    summary: dict[str, object] = {
        "manifest": str(manifest.resolve()),
        "samples": len(records),
        "keep_targets_normalized": repaired,
        "copied_bytes": copied_bytes,
        "policy": "MUNO21 no-change scenarios preserve the 2013 base graph",
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    print(json.dumps(normalize_keep_targets(args.manifest, args.report), indent=2))


if __name__ == "__main__":
    main()
