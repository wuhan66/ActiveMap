import json
from pathlib import Path

import pytest
import yaml

from scripts.audit_paper_result_bundles import audit_result_bundles
from scripts.build_paper_result_bundle import build_result_bundle
from scripts.render_paper_result_tables import render_result_tables


def _registry(path: Path, *, test_policy: str = "validation_only") -> Path:
    payload = {
        "protocol": {
            "muno21_budgets": [3.0],
            "bootstrap_unit": "task_or_aoi",
            "bootstrap_replicates": 200,
            "bootstrap_seed": 11,
            "confidence_level": 0.95,
        },
        "required_metrics": {
            "updater": ["score"],
            "selector": ["score"],
            "agent": ["score"],
            "writeback": ["score"],
        },
        "primary_metrics": {
            "updater": ["score"],
            "selector": ["score"],
            "agent": ["score"],
            "rl": ["score"],
            "writeback": ["score"],
        },
        "experiments": [
            {
                "id": "updater",
                "family": "updater",
                "test_policy": test_policy,
                "seeds": [7, 8],
            }
        ],
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _observations(path: Path, *, mismatched_units: bool = False) -> Path:
    rows = []
    values = {"7": {"a": 0.2, "b": 0.6}, "8": {"a": 0.4, "b": 0.8}}
    for seed, units in values.items():
        for unit_id, score in units.items():
            if mismatched_units and seed == "8" and unit_id == "b":
                unit_id = "c"
            rows.append(
                {
                    "schema_version": "activemap-paper-observation-v1",
                    "experiment_id": "updater",
                    "variant": None,
                    "budget": None,
                    "split": "val",
                    "seed": seed,
                    "observation_id": f"{seed}-{unit_id}",
                    "unit_id": unit_id,
                    "metrics": {"score": score},
                }
            )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_build_bundle_uses_seed_means_and_paired_group_bootstrap(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.yaml")
    observations = _observations(tmp_path / "observations.jsonl")
    bundle = build_result_bundle(
        registry,
        [observations],
        experiment_id="updater",
        variant=None,
        budget=None,
    )

    assert bundle["sample_count"] == 4
    assert bundle["unit_count"] == 2
    assert bundle["rows_by_seed"] == {"7": 2, "8": 2}
    assert bundle["metrics"]["score"]["mean"] == pytest.approx(0.5)
    assert bundle["metrics"]["score"]["std"] == pytest.approx(2**0.5 / 10)
    assert bundle["metrics"]["score"]["seed_means"] == pytest.approx(
        {"7": 0.4, "8": 0.6}
    )
    low, high = bundle["metrics"]["score"]["ci95"]
    assert 0.3 <= low <= 0.5 <= high <= 0.7

    bundles = tmp_path / "bundles"
    bundles.mkdir()
    (bundles / "updater.json").write_text(
        json.dumps(bundle, indent=2), encoding="utf-8"
    )
    report = audit_result_bundles(registry, bundles)
    assert report["paper_tables_complete"] is True
    manifest = render_result_tables(registry, bundles, tmp_path / "tables")
    assert manifest["complete"] is True
    assert (tmp_path / "tables/updater.csv").is_file()
    markdown = (tmp_path / "tables/updater.md").read_text(encoding="utf-8")
    assert "0.5000 +/- 0.1414" in markdown
    assert len(manifest["tables"]) == 2


def test_build_bundle_rejects_unpaired_bootstrap_units(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.yaml")
    observations = _observations(tmp_path / "observations.jsonl", mismatched_units=True)
    with pytest.raises(ValueError, match="unit ids must be identical"):
        build_result_bundle(
            registry,
            [observations],
            experiment_id="updater",
            variant=None,
            budget=None,
        )


def test_test_bundle_requires_completed_frozen_ledger(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.yaml", test_policy="frozen_once")
    observations = _observations(tmp_path / "observations.jsonl")
    rows = [json.loads(line) for line in observations.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["split"] = "test"
    observations.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="frozen test ledger"):
        build_result_bundle(
            registry,
            [observations],
            experiment_id="updater",
            variant=None,
            budget=None,
        )


def test_table_rendering_refuses_incomplete_result_set(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.yaml")
    bundles = tmp_path / "empty"
    bundles.mkdir()
    with pytest.raises(ValueError, match="incomplete"):
        render_result_tables(registry, bundles, tmp_path / "tables")
