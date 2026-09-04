from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.import_sn7_frozen_test_into_iclr import import_result


SEEDS = [20260730, 20260731, 20260801]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _bundle(root: Path, *, reporting_policy: str = "report_regardless_of_promotion_outcome") -> tuple[Path, Path, Path]:
    export_dir = root / "sealed"
    paper_dir = root / "paper"
    operation_path = root / "operation_slices.json"
    _write_json(
        export_dir / "manifest.json",
        {
            "schema_version": "sn7-step0-frozen-test-tables-v1",
            "split": "test",
            "test_assets_read": True,
            "seeds": SEEDS,
            "promotion_passed": False,
            "reporting_policy": reporting_policy,
            "protocol_recovery": "v3_cap20_launcher_only",
            "metric_reporter_sha256": "a" * 64,
            "compatibility_adapter_sha256": "b" * 64,
        },
    )
    controller_rows = []
    for index, variant in enumerate(("notool", "forced", "benefit")):
        row: dict[str, object] = {"method": variant, "variant": variant}
        for metric in (
            "terminal_accuracy",
            "false_edit_rate",
            "missed_edit_rate",
            "mean_quality_cost_utility",
            "mean_tool_calls",
        ):
            row[f"{metric}_mean"] = 0.1 + index * 0.01
            row[f"{metric}_std"] = 0.001
        controller_rows.append(row)
    _write_csv(export_dir / "controller_table.csv", controller_rows)

    writeback_rows = []
    for index, variant in enumerate(("notool", "forced", "benefit")):
        row = {"method": variant, "variant": variant}
        for metric in (
            "raster_iou_auc",
            "false_edit_auc",
            "missed_edit_auc",
            "spent_cost_auc",
            "episode_utility_v2_balanced_auc",
            "episode_utility_v2_safety_auc",
        ):
            row[metric] = 0.2 + index * 0.01
        writeback_rows.append(row)
    _write_csv(export_dir / "writeback_table.csv", writeback_rows)

    fields = (
        "raster_iou_auc",
        "false_edit_auc",
        "missed_edit_auc",
        "spent_cost_auc",
        "episode_utility_v2_safety_auc",
    )
    comparison = {
        field: {"delta": 0.01, "ci95_low": -0.01, "ci95_high": 0.03}
        for field in fields
    }
    _write_json(
        export_dir / "paired_intervals.json",
        {
            "controller": {},
            "writeback": {
                "benefit_vs_notool": comparison,
                "benefit_vs_forced": comparison,
            },
        },
    )

    operations = {}
    for operation in ("KEEP", "ADD", "DELETE", "RESHAPE"):
        operations[operation] = {
            "support_per_seed": 20,
            "aoi_count": 3,
            "candidate_minus_reference_mean": {
                "map_quality_after": 0.01,
                "false_edit": -0.01,
                "missed_edit": 0.0,
                "spent_cost": 0.02,
            },
        }
    _write_json(
        operation_path,
        {
            "schema_version": "sn7-frozen-test-operation-slices-v1",
            "split": "test",
            "test_assets_read": True,
            "model_seeds": SEEDS,
            "operations": operations,
        },
    )
    _write_json(
        paper_dir / "numeric_evidence.json",
        {"schema_version": "ai-research-writing/numeric-evidence-v2", "entries": []},
    )
    return export_dir, paper_dir, operation_path


def test_import_renders_failed_result_without_hiding_it(tmp_path: Path) -> None:
    export_dir, paper_dir, operation_path = _bundle(tmp_path)
    receipt = import_result(
        export_dir,
        paper_dir,
        operation_slices_path=operation_path,
    )

    generated = paper_dir / "evidence" / "generated"
    macros = (generated / "sn7_frozen_result_macros.tex").read_text(encoding="utf-8")
    tables = (generated / "sn7_frozen_result_tables.tex").read_text(encoding="utf-8")
    numeric = json.loads((paper_dir / "numeric_evidence.json").read_text(encoding="utf-8"))

    assert "promotion gate failed" in macros
    assert "\\textbf{failed}" in tables
    assert "RESHAPE" in tables
    assert receipt["promotion_passed"] is False
    assert any(
        row["source"] == "evidence/generated/sn7_frozen_result_snapshot.json"
        for row in numeric["entries"]
    )
    assert any(
        row["selector"]["pointer"] == "/controller_rows/0/terminal_accuracy_mean"
        and row["value"] == pytest.approx(0.1)
        for row in numeric["entries"]
    )


def test_import_rejects_non_sign_agnostic_bundle(tmp_path: Path) -> None:
    export_dir, paper_dir, operation_path = _bundle(
        tmp_path,
        reporting_policy="report_only_when_promoted",
    )
    with pytest.raises(ValueError, match="sign-agnostic"):
        import_result(
            export_dir,
            paper_dir,
            operation_slices_path=operation_path,
        )


def test_reimport_replaces_only_legacy_sn7_numeric_entries(tmp_path: Path) -> None:
    export_dir, paper_dir, operation_path = _bundle(tmp_path)
    _write_json(
        paper_dir / "numeric_evidence.json",
        {
            "schema_version": "ai-research-writing/numeric-evidence-v2",
            "entries": [
                {
                    "value": 0.9,
                    "source": "evidence/evidence_snapshot.json",
                    "selector": {"kind": "json-pointer", "pointer": "/sn7/obsolete"},
                    "aggregate": "identity",
                    "representations": ["raw"],
                },
                {
                    "value": 0.8,
                    "source": "evidence/evidence_snapshot.json",
                    "selector": {"kind": "json-pointer", "pointer": "/muno21/retained"},
                    "aggregate": "identity",
                    "representations": ["raw"],
                },
                {
                    "value": 0.7,
                    "source": "evidence/generated/sn7_frozen_result_snapshot.json",
                    "selector": {"kind": "json-pointer", "pointer": "/obsolete"},
                    "aggregate": "identity",
                    "representations": ["raw"],
                },
            ],
        },
    )

    import_result(export_dir, paper_dir, operation_slices_path=operation_path)
    numeric = json.loads((paper_dir / "numeric_evidence.json").read_text(encoding="utf-8"))
    entries = numeric["entries"]

    assert any(
        row["source"] == "evidence/evidence_snapshot.json"
        and row["selector"]["pointer"] == "/muno21/retained"
        for row in entries
    )
    assert not any(
        row["source"] == "evidence/evidence_snapshot.json"
        and row["selector"]["pointer"].startswith("/sn7/")
        for row in entries
    )
    assert not any(
        row["source"] == "evidence/generated/sn7_frozen_result_snapshot.json"
        and row["selector"]["pointer"] == "/obsolete"
        for row in entries
    )
    assert sum(
        row["source"] == "evidence/generated/sn7_frozen_result_snapshot.json"
        for row in entries
    ) > 1
