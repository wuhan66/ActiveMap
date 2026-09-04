import argparse
import importlib.util
from pathlib import Path

from activemap.models import EditOperation
from activemap.updater_records import UpdaterSample, load_updater_samples

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "prepare_sn7_carried_runtime_temporal_manifest.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_carried_runtime_temporal", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sample(
    sample_id: str,
    *,
    split: str,
    dataset_name: str,
    aoi_id: str,
    object_id: str,
) -> UpdaterSample:
    temporal = dataset_name == "sn7_carried_runtime_temporal"
    return UpdaterSample(
        sample_id=sample_id,
        aoi_id=aoi_id,
        split=split,
        image_path=f"{sample_id}_image.npy",
        prior_image_path=f"{sample_id}_prior_image.npy",
        prior_mask_path=f"{sample_id}_prior.npy",
        target_mask_path=f"{sample_id}_target.npy",
        valid_mask_path=f"{sample_id}_valid.npy",
        edit_type=EditOperation.ADD,
        geometry_delta=[0.0] * 8,
        object_id=object_id,
        dataset_name=dataset_name,
        source_metadata=(
            {
                "replay_policy": "carried-runtime-temporal-v2",
                "target_alignment": "current_anchor_only",
            }
            if temporal
            else {}
        ),
    )


def _write(path: Path, samples: list[UpdaterSample]) -> None:
    path.write_text(
        "".join(sample.model_dump_json() + "\n" for sample in samples),
        encoding="utf-8",
    )


def test_prepare_manifest_is_group_disjoint_and_temporal(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.jsonl"
    replay = tmp_path / "replay.jsonl"
    output = tmp_path / "merged.jsonl"
    _write(
        canonical,
        [
            _sample(
                "keep",
                split="train",
                dataset_name="spacenet7",
                aoi_id="train-aoi",
                object_id="train-object",
            ),
            _sample(
                "exclude",
                split="train",
                dataset_name="spacenet7",
                aoi_id="val-aoi",
                object_id="val-object",
            ),
        ],
    )
    _write(
        replay,
        [
            _sample(
                "replay-train",
                split="train",
                dataset_name="sn7_carried_runtime_temporal",
                aoi_id="replay-aoi",
                object_id="replay-object",
            ),
            _sample(
                "replay-val",
                split="val",
                dataset_name="sn7_carried_runtime_temporal",
                aoi_id="val-aoi",
                object_id="val-object",
            ),
        ],
    )

    summary = MODULE.prepare_manifest(
        argparse.Namespace(
            canonical_manifest=canonical,
            replay_manifest=replay,
            output_manifest=output,
        )
    )

    samples = load_updater_samples(output)
    assert [sample.sample_id for sample in samples] == [
        "keep",
        "replay-train",
        "replay-val",
    ]
    assert summary["canonical_train_excluded_for_replay_validation"] == 1
    assert summary["temporal_pair_input"] is True
    assert summary["train_validation_group_overlap"] == 0
