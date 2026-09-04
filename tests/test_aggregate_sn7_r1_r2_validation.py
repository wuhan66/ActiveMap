import hashlib
import json

import pytest

from scripts.aggregate_sn7_r1_r2_validation import aggregate


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(seed, policy):
    value = 1.0 if policy == "activemap" else 0.0
    return "".join(
        json.dumps(
            {
                "source_episode": f"episode-{aoi}",
                "budget": 3.0,
                "aoi_id": f"aoi-{aoi}",
                "split": "val",
                "test_assets_read": False,
                "terminal_correct": value,
                "false_edit": 0.0,
                "missed_edit": 0.0,
                "spent_cost": 0.0,
                "quality_gain": value + seed * 0.0,
                "quality_cost_utility": value,
                "episode_utility_v2_proxy_balanced": value,
                "episode_utility_v2_proxy_safety": value,
                "episode_utility_v2_proxy_cost_aware": value,
            }
        )
        + "\n"
        for aoi in range(2)
    )


def _receipt(root, seed, *, manifest="manifest", val_hash="val"):
    natural, filtered = {}, {}
    for analysis, target in (("natural", natural), ("filtered", filtered)):
        for policy in ("activemap", "always_stop"):
            path = root / f"seed{seed}" / analysis / f"{policy}.jsonl"
            target[policy] = {
                "path": str(path.resolve()),
                "sha256": _write(path, _rows(seed, policy)),
            }
    receipt = {
        "split": "val",
        "test_assets_read": False,
        "controller_seed": seed,
        "inputs": {
            "train_states": {"sha256": "train"},
            "train_episodes": {"sha256": "train_episodes"},
            "val_states": {"sha256": val_hash},
            "val_episodes": {"sha256": "val_episodes"},
            "edit_manifest": {"sha256": manifest},
        },
        "natural_validation_traces": natural,
        "r2_filtered_traces": filtered,
    }
    _write(root / f"seed{seed}" / "COMPLETE.json", json.dumps(receipt))


def test_three_seed_aggregate_requires_shared_inputs_and_produces_both_analyses(tmp_path):
    for seed in (20260730, 20260731, 20260801):
        _receipt(tmp_path, seed)
    result = aggregate(tmp_path, repetitions=10, seed=1)
    assert result["split"] == "val"
    assert result["test_assets_read"] is False
    assert set(result["results"]) == {"r1_natural", "r2_edit_only"}
    interval = result["results"]["r1_natural"]["activemap_minus_always_stop"]["intervals"]
    assert interval["terminal_accuracy"]["observed_delta"] == 1.0


def test_three_seed_aggregate_rejects_mismatched_manifest(tmp_path):
    for seed in (20260730, 20260731, 20260801):
        _receipt(tmp_path, seed, manifest="other" if seed == 20260801 else "manifest")
    with pytest.raises(ValueError, match="generated R2 manifest"):
        aggregate(tmp_path, repetitions=10, seed=1)
