import hashlib
import json
from pathlib import Path

import pytest

from scripts.export_sn7_step0_frozen_test_tables import export
from scripts.export_sn7_step0_paper_tables import (
    CONTROLLER_METRICS,
    WRITEBACK_METRICS,
)


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _summary() -> dict:
    rows = []
    for seed in (1, 2, 3):
        for variant_index, variant in enumerate(("notool", "forced", "benefit")):
            row = {"seed": seed, "variant": variant}
            for metric_index, (metric, _) in enumerate(CONTROLLER_METRICS):
                row[metric] = 0.1 * (metric_index + 1) + 0.01 * variant_index
            rows.append(row)
    return {
        "seeds": [1, 2, 3],
        "per_seed": rows,
        "comparisons": {"benefit_vs_notool": {}, "benefit_vs_forced": {}},
        "protocol": {"split": "test", "test_assets_read": True},
    }


def _comparison(offset: float) -> dict:
    candidate = {
        metric: 0.2 + 0.01 * index
        for index, (metric, _) in enumerate(WRITEBACK_METRICS)
    }
    baseline = {metric: value + offset for metric, value in candidate.items()}
    return {
        "model_seeds": [1, 2, 3],
        "baseline": baseline,
        "candidate": candidate,
        "paired_delta": {
            metric: {
                "delta": candidate[metric] - baseline[metric],
                "ci95_low": -0.02,
                "ci95_high": 0.02,
            }
            for metric, _ in WRITEBACK_METRICS
        },
        "split": "test",
        "test_assets_read": True,
    }


def _fixture(tmp_path: Path, *, promotion_passed: bool = False) -> tuple[Path, ...]:
    registry = tmp_path / "registry.yaml"
    registry.write_text("schema_version: test\n", encoding="utf-8")
    ledger = _write(
        tmp_path / "ledger.json",
        {
            "schema_version": "activemap-frozen-test-access-v1",
            "status": "complete",
            "returncode": 0,
            "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
            "command": ["bash", "scripts/run_sn7_step0_frozen_test_v2.sh"],
        },
    )
    run_root = tmp_path / "run"
    _write(
        run_root / "COMPLETE.json",
        {
            "schema_version": "sn7-step0-frozen-test-complete-v2",
            "test_assets_read": True,
            "promotion_passed": promotion_passed,
        },
    )
    _write(run_root / "three_policy_summary.json", _summary())
    _write(run_root / "benefit_vs_notool.json", _comparison(-0.01))
    _write(run_root / "benefit_vs_forced.json", _comparison(0.01))
    _write(
        run_root / "writeback_promotion.json",
        {
            "promote": promotion_passed,
            "claim_boundary": "Three-seed frozen-test outcome.",
            "test_assets_read": True,
        },
    )
    return registry, ledger, run_root, tmp_path / "tables"


def test_export_reports_a_non_promoted_frozen_test(tmp_path: Path) -> None:
    registry, ledger, run_root, output = _fixture(tmp_path)
    manifest = export(registry, ledger, run_root, output)

    assert manifest["split"] == "test"
    assert manifest["test_assets_read"] is True
    assert manifest["promotion_passed"] is False
    assert manifest["reporting_policy"] == "report_regardless_of_promotion_outcome"
    assert manifest["seeds"] == [1, 2, 3]
    assert len(manifest["sources"]["ledger"]["sha256"]) == 64
    assert "One-time frozen test" in (output / "controller_table.md").read_text(
        encoding="utf-8"
    )


def test_export_rejects_incomplete_access_ledger(tmp_path: Path) -> None:
    registry, ledger, run_root, output = _fixture(tmp_path)
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["status"] = "started"
    _write(ledger, payload)

    with pytest.raises(ValueError, match="not complete and successful"):
        export(registry, ledger, run_root, output)


def test_export_rejects_non_test_evidence(tmp_path: Path) -> None:
    registry, ledger, run_root, output = _fixture(tmp_path)
    summary_path = run_root / "three_policy_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["protocol"] = {"split": "val", "test_assets_read": False}
    _write(summary_path, summary)

    with pytest.raises(ValueError, match="consistent frozen-test evidence"):
        export(registry, ledger, run_root, output)


def test_export_rejects_promotion_disagreement(tmp_path: Path) -> None:
    registry, ledger, run_root, output = _fixture(tmp_path)
    complete_path = run_root / "COMPLETE.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["promotion_passed"] = True
    _write(complete_path, complete)

    with pytest.raises(ValueError, match="promotion artifacts disagree"):
        export(registry, ledger, run_root, output)
