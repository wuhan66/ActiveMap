import json
from pathlib import Path

from scripts.assert_training_ready import DATASET_GATES, training_readiness


def test_repository_training_readiness_is_blocked_without_data(tmp_path: Path) -> None:
    result = training_readiness(
        Path("configs/experiments/paper_registry.yaml"), tmp_path / "empty"
    )
    assert result["ready"] is False
    assert set(result["failed_gates"]) == DATASET_GATES


def test_training_readiness_accepts_all_dataset_gates(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    paths = {
        "sn7_audit": storage / "processed/sn7_v1/updater_v4_cap20/audit.json",
        "muno21_audit": storage / "processed/muno21_v2/updater/audit.json",
        "inria_audit": storage / "processed/inria_v1/segmentation/audit.json",
        "manual_qc_approval": storage / "logs/dataset_preparation/QC_APPROVED.json",
    }
    for artifact_id, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"approved": True} if artifact_id == "manual_qc_approval" else {"passed": True}
        path.write_text(json.dumps(payload), encoding="utf-8")

    result = training_readiness(Path("configs/experiments/paper_registry.yaml"), storage)
    assert result["ready"] is True
    assert result["failed_gates"] == []
