from scripts.build_sn7_controller_uncertainty_paper_assets import build_rows


def _metric(delta: float) -> dict[str, float]:
    return {
        "delta": delta,
        "ci95_low": delta - 0.01,
        "ci95_high": delta + 0.01,
    }


def test_build_rows_marks_strict_dominance() -> None:
    quantiles = {}
    active = {
        "raster_iou_auc": 0.8,
        "false_edit_auc": 0.1,
        "spent_cost_auc": 0.3,
    }
    for label in ("q05", "q15", "q30"):
        quantiles[label] = {
            "controller": {"tool_call_episode_rate": 0.2},
            "writeback": {
                "uncertainty": {
                    "raster_iou_auc": 0.7,
                    "false_edit_auc": 0.2,
                    "spent_cost_auc": 0.4,
                },
                "active": active,
            },
            "paired_writeback": {
                "active_vs_uncertainty": {
                    "raster_iou_auc": _metric(0.1),
                    "false_edit_auc": _metric(-0.1),
                    "spent_cost_auc": _metric(-0.1),
                }
            },
        }
    rows = build_rows({"rows": [{"severity": 4, "quantiles": quantiles}]})

    assert len(rows) == 3
    assert rows[0]["quality_ci_positive"] is True
    assert rows[0]["false_edit_ci_negative"] is True
    assert rows[0]["cost_ci_negative"] is True
