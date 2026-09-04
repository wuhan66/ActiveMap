import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCRIPT = Path(__file__).parents[1] / "scripts" / "build_sn7_carried_replay_updater_dataset.py"
SPEC = importlib.util.spec_from_file_location("carried_replay_dataset", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_dataset_writes_manifest_after_creating_array_directories(
    tmp_path: Path, monkeypatch
) -> None:
    item = SimpleNamespace(evidence_id="evidence-0")
    episode = SimpleNamespace(
        split="train",
        episode_id="episode-0",
        aoi_id="aoi-0",
        prior_geometry="canonical-prior",
        target_geometry="target",
        hypothesis=SimpleNamespace(object_id="object-0"),
        metadata={},
        anchor_timestamp="2020-01",
        evidence_catalog=[item],
    )
    prior = np.zeros((16, 16), dtype=np.float32)
    target = np.zeros((16, 16), dtype=np.float32)
    target[4:12, 4:12] = 1.0

    monkeypatch.setattr(MODULE, "load_episodes", lambda _: [episode])
    monkeypatch.setattr(MODULE, "_remap_episode_assets", lambda rows, _: rows)
    monkeypatch.setattr(
        MODULE,
        "build_contiguous_chains",
        lambda *_args, **_kwargs: [[episode], [episode]],
    )
    monkeypatch.setattr(MODULE, "_episode_geometry", lambda value: value)
    monkeypatch.setattr(MODULE, "_same_state", lambda *_args: False)
    monkeypatch.setattr(MODULE, "_observable_items", lambda *_args, **_kwargs: [item])
    monkeypatch.setattr(
        MODULE,
        "_read_candidate",
        lambda *_args, **_kwargs: (
            np.zeros((3, 16, 16), dtype=np.float32),
            prior,
            target,
            np.ones((16, 16), dtype=np.float32),
            None,
        ),
    )
    output_dir = tmp_path / "carried-replay"
    args = argparse.Namespace(
        episodes=tmp_path / "episodes.jsonl",
        output_dir=output_dir,
        image_size=16,
        image_channels=3,
        minimum_chain_length=2,
        continuity_tolerance=1e-6,
        validation_chain_index=1,
        max_chains=None,
        max_evidence_per_step=1,
        asset_root_map=[],
    )

    summary = MODULE.build_dataset(args)

    assert summary["sample_counts"] == {"train": 1, "val": 1}
    assert summary["residual_operation_counts"] == {"train:ADD": 1, "val:ADD": 1}
    assert (output_dir / "updater_samples.jsonl").read_text(encoding="utf-8").count("\n") == 2
    assert (output_dir / "summary.json").is_file()
