import csv
import json

from scripts.export_active_catalog_paper_table import METRICS, export


def aggregate_fixture():
    intervals = {
        metric: {
            "observed_fixed_seed_mean": 0.2,
            "ci95_low": 0.1,
            "ci95_high": 0.3,
        }
        for metric in METRICS
    }
    intervals["selection_macro_f1_delta_vs_always_stop"] = {
        "observed_fixed_seed_mean": 0.1,
        "ci95_low": 0.01,
        "ci95_high": 0.2,
    }
    intervals["false_call_rate"]["observed_fixed_seed_mean"] = 0.05
    baselines = {
        name: {metric: 0.05 for metric in METRICS}
        for name in ("always_stop", "cheapest", "clear_per_cost", "random")
    }
    return {
        "schema_version": "active-catalog-fixed-seed-aoi-bootstrap-v1",
        "seed_count": 3,
        "aoi_count": 9,
        "shared_resample_indices_across_model_seeds": True,
        "fixed_seed_mean_intervals": intervals,
        "baseline_metrics": baselines,
        "test_assets_read": False,
    }


def test_export_writes_complete_table_and_promotion_gate(tmp_path):
    source = tmp_path / "aggregate.json"
    source.write_text(json.dumps(aggregate_fixture()), encoding="utf-8")
    output = tmp_path / "paper"
    result = export(source, output)
    assert result["promotion_passed"] is True
    with (output / "main_results.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["method"] for row in rows] == [
        "ActiveMap (Qwen3-VL-4B)",
        "always_stop",
        "cheapest",
        "clear_per_cost",
        "random",
    ]
    gate = json.loads((output / "promotion_gate.json").read_text(encoding="utf-8"))
    assert gate["passed"] is True
    assert "[0.1000, 0.3000]" in (output / "main_results.md").read_text(encoding="utf-8")
