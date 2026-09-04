import json

from scripts.finalize_sn7_changemamba_validation import finalize


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _jsonl(path, count):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"sample_id": index}) + "\n" for index in range(count)),
        encoding="utf-8",
    )


def test_finalize_accepts_complete_test_free_evidence(tmp_path):
    run_root = tmp_path / "runs"
    bundle_root = tmp_path / "bundles"
    seeds = [7, 8]
    for seed in seeds:
        run = run_root / f"full_weight5_seed{seed}_v1"
        _json(
            run / "summary.json",
            {
                "seed": seed,
                "best_epoch": 3,
                "test_assets_read": False,
            },
        )
        _json(
            run / "train_audit" / "summary.json",
            {
                "split": "train",
                "sample_count": 6,
                "masks_saved": False,
                "checkpoint_sha256": f"checkpoint-{seed}",
                "test_assets_read": False,
            },
        )
        _json(
            run / "val_audit" / "summary.json",
            {
                "split": "val",
                "sample_count": 4,
                "masks_saved": True,
                "checkpoint_sha256": f"checkpoint-{seed}",
                "test_assets_read": False,
            },
        )
        _json(
            run / "safe_commit" / "summary.json",
            {
                "calibration": {"train_sample_count": 6},
                "validation": {"sample_count": 4},
                "test_assets_read": False,
            },
        )
        for directory, count in (
            (run / "train_audit", 6),
            (run / "val_audit", 4),
            (run / "safe_commit", 4),
        ):
            _jsonl(directory / "per_sample.jsonl", count)
        _jsonl(run / "val_audit" / "predictions.jsonl", 4)
        _jsonl(run / "safe_commit" / "predictions.jsonl", 4)

    _json(
        run_root / "full_weight5_three_seed_20260726.json",
        {"run_count": 2, "protocol": {"test_assets_read": False}},
    )
    for name in (
        "full_weight5_three_seed_aoi_bootstrap_20260726.json",
        "full_weight5_three_seed_safe_commit_20260726.json",
        "full_weight5_three_seed_safe_commit_aoi_bootstrap_20260726.json",
    ):
        _json(
            run_root / name,
            {"run_count": 2, "test_assets_read": False},
        )
    for filename, variant in (
        ("changemamba_always_commit_val.json", "always_commit"),
        ("changemamba_safe_commit_val.json", "safe_commit"),
    ):
        _json(
            bundle_root / filename,
            {
                "experiment_id": "sn7_changemamba_commit_policy",
                "variant": variant,
                "split": "val",
                "seeds": ["7", "8"],
                "unit_count": 2,
            },
        )

    result = finalize(run_root, bundle_root, seeds=seeds)

    assert result["status"] == "complete"
    assert result["test_assets_read"] is False
    assert result["artifact_count"] == 24
