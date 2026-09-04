import hashlib
import json
from pathlib import Path

import yaml

from scripts.audit_paper_result_bundles import _expected_cells, audit_result_bundles


def test_repository_registry_expands_every_paper_cell(tmp_path: Path) -> None:
    registry = yaml.safe_load(
        Path("configs/experiments/paper_registry.yaml").read_text(encoding="utf-8")
    )
    expected_cell_count = len(_expected_cells(registry))
    report = audit_result_bundles(
        Path("configs/experiments/paper_registry.yaml"), tmp_path / "empty"
    )
    assert report["contract_valid"] is True
    assert report["paper_tables_complete"] is False
    assert report["expected_cell_count"] == expected_cell_count
    assert report["observed_cell_count"] == 0
    assert len(report["missing_cells"]) == expected_cell_count
    stochastic_cells = [
        cell
        for cell in _expected_cells(registry)
        if cell["family"] in {"agent", "writeback"}
    ]
    assert stochastic_cells
    assert all(cell["seeds"] for cell in stochastic_cells)
    assert ("20260821", "20260822", "20260823") in {
        tuple(cell["seeds"]) for cell in stochastic_cells
    }


def test_complete_validation_bundle_passes_contract(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("seed: 1\n", encoding="utf-8")
    registry = {
        "protocol": {
            "muno21_budgets": [3.0],
            "bootstrap_unit": "task_or_aoi",
            "bootstrap_replicates": 100,
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
                "id": "validation_rl",
                "family": "rl",
                "role": "gated_intervention",
                "test_policy": "validation_only",
                "seeds": [7],
            }
        ],
    }
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    registry_hash = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    bundles = tmp_path / "bundles"
    bundles.mkdir()
    observations = tmp_path / "observations.jsonl"
    observations.write_text("{}\n", encoding="utf-8")
    observation_hash = hashlib.sha256(observations.read_bytes()).hexdigest()
    (bundles / "result.json").write_text(
        json.dumps(
            {
                "schema_version": "activemap-paper-result-v1",
                "experiment_id": "validation_rl",
                "variant": None,
                "budget": 3.0,
                "split": "val",
                "seeds": [7],
                "sample_count": 10,
                "unit_count": 5,
                "bootstrap_unit": "task_or_aoi",
                "bootstrap_replicates": 100,
                "confidence_level": 0.95,
                "registry_sha256": registry_hash,
                "source_observations": [
                    {"path": str(observations), "sha256": observation_hash}
                ],
                "metrics": {"score": {"mean": 0.5, "std": 0.1, "ci95": [0.4, 0.6]}},
            }
        ),
        encoding="utf-8",
    )
    report = audit_result_bundles(registry_path, bundles)
    assert report["contract_valid"] is True
    assert report["paper_tables_complete"] is True
    assert report["complete_cell_count"] == 1
