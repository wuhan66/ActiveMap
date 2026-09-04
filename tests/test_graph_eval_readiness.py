from pathlib import Path

from scripts.audit_graph_eval_readiness import PINNED_COMMIT, audit_graph_eval


def test_graph_eval_readiness_reports_missing_infrastructure_without_test_access(
    tmp_path: Path,
) -> None:
    report = audit_graph_eval(tmp_path / "project", tmp_path / "storage")
    assert report["ready"] is False
    assert report["infrastructure_ready"] is False
    assert report["test_assets_read"] is False
    assert report["third_party_source_imported"] is False
    assert report["pinned_commit"] == PINNED_COMMIT


def test_official_metric_runner_records_per_budget_provenance() -> None:
    runner = Path("scripts/run_muno21_official_graph_metrics.sh").read_text(
        encoding="utf-8"
    )
    assert "official_metric_logs" in runner
    assert "apls.exit_code.txt" not in runner
    assert 'run_metric "$budget_output" apls' in runner
    assert 'run_metric "$budget_output" pixel_f1' in runner
    assert 'run_metric "$budget_output" error_rate' in runner
    assert 'export PYTHONPATH="${OFFICIAL}/python:' in runner
    assert 'export PATH="$(dirname "$GO_BIN"):${PATH}"' in runner
    assert "export GOPROXY=off" in runner
    assert "muno21-official-graph-metric-run-v1" in runner
    assert "official_metric_run_manifest.json" in runner
