import json
from pathlib import Path

import pytest

from scripts.render_paired_rollout_reports import render


def _report(path: Path, comparison_id: str) -> Path:
    metric = {
        "baseline": {"mean": 0.1, "ci95": [0.0, 0.2]},
        "candidate": {"mean": 0.2, "ci95": [0.1, 0.3]},
        "delta_candidate_minus_baseline": {"mean": 0.1, "ci95": [0.05, 0.15]},
    }
    payload = {
        "schema_version": "activemap-paired-rollout-report-v1",
        "comparison_id": comparison_id,
        "claim": "claim",
        "budgets": [1.5, 3.0, 4.5],
        "by_budget": {
            "3": {
                "quality_cost_utility": metric,
                "false_edit_rate": metric,
                "mean_cost": metric,
            }
        },
        "quality_cost_auc": metric,
        "gates": {"quality_cost_safety_non_dominated": True},
        "all_required_gates_passed": True,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_renders_hashed_csv_and_markdown_claim_matrix(tmp_path: Path) -> None:
    reports = [_report(tmp_path / f"{name}.json", name) for name in ("a", "b")]
    output = tmp_path / "tables"
    manifest = render(reports, output)

    assert manifest["report_count"] == 2
    assert (output / "paired_claim_matrix.csv").read_text(encoding="utf-8").count(
        "true"
    ) == 4
    assert "| a | true |" in (output / "paired_claim_matrix.md").read_text(
        encoding="utf-8"
    )


def test_renderer_refuses_duplicate_comparison_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate paired comparison"):
        render(
            [_report(tmp_path / "a.json", "same"), _report(tmp_path / "b.json", "same")],
            tmp_path / "tables",
        )
