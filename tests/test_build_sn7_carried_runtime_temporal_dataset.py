import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from activemap.updater_records import load_updater_samples

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "build_sn7_carried_runtime_temporal_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("carried_runtime_temporal_dataset", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_dataset_pairs_previous_and_current_rgb_with_current_target(
    tmp_path: Path, monkeypatch
) -> None:
    item = SimpleNamespace(
        evidence_id="anchor",
        timestamp="2020-02",
        prior_timestamp="2020-01",
        prior_image_path="prior.tif",
    )
    episode = SimpleNamespace(
        split="train",
        episode_id="episode",
        aoi_id="aoi",
        prior_geometry="prior",
        target_geometry="target",
        hypothesis=SimpleNamespace(object_id="object"),
        metadata={},
        anchor_timestamp="2020-02",
        evidence_catalog=[item],
    )
    monkeypatch.setattr(MODULE, "load_episodes", lambda _: [episode])
    monkeypatch.setattr(MODULE, "_remap_episode_assets", lambda rows, _: rows)
    monkeypatch.setattr(
        MODULE, "build_contiguous_chains", lambda *_args, **_kwargs: [[episode], [episode]]
    )
    monkeypatch.setattr(MODULE, "_episode_geometry", lambda value: value)
    monkeypatch.setattr(MODULE, "_same_state", lambda left, right, **_: left == right)
    calls = []

    def fake_read(_item, **kwargs):
        calls.append(kwargs)
        image = np.concatenate(
            (
                np.full((3, 16, 16), 0.25, dtype=np.float32),
                np.full((3, 16, 16), 0.75, dtype=np.float32),
            )
        )
        prior = np.zeros((16, 16), dtype=np.float32)
        target = np.zeros((16, 16), dtype=np.float32)
        target[4:12, 4:12] = 1.0
        return image, prior, target, np.ones((16, 16), dtype=np.float32), (16, 16)

    monkeypatch.setattr(MODULE, "_read_paired_candidate", fake_read)
    output = tmp_path / "dataset"
    summary = MODULE.build_dataset(
        argparse.Namespace(
            episodes=tmp_path / "episodes.jsonl",
            output_dir=output,
            image_size=16,
            minimum_chain_length=2,
            continuity_tolerance=1e-6,
            validation_chain_index=1,
            max_chains=None,
            state_policies=None,
            asset_root_map=[],
        )
    )

    samples = load_updater_samples(output / "updater_samples.jsonl")
    assert summary["sample_counts"] == {"train": 1, "val": 1}
    assert summary["target_alignment"] == "current_anchor_only"
    assert all(call["temporal_pair_input"] is True for call in calls)
    assert all(call["target_geometry"] == "target" for call in calls)
    assert np.load(samples[0].prior_image_path).mean() == 64
    assert np.load(samples[0].image_path).mean() == 191
    assert samples[0].source_metadata["state_policies"] == [
        "teacher_corrected",
        "one_step_defer",
        "defer_from_start",
    ]
