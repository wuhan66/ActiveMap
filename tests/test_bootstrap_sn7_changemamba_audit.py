from scripts.bootstrap_sn7_changemamba_audit import cluster_bootstrap


def _row(aoi: str, edit: str, delta: float):
    return {
        "aoi_id": aoi,
        "target_edit": edit,
        "prior_map_iou": 0.5,
        "committed_map_iou": 0.5 + delta,
        "map_iou_delta": delta,
        "change_iou": 0.7,
        "operation_correct": True,
        "false_edit": False,
        "missed_edit": False,
        "wrong_edit": False,
    }


def test_cluster_bootstrap_reports_positive_delta_interval():
    rows = [
        _row("a", "KEEP", 0.1),
        _row("a", "ADD", 0.2),
        _row("b", "KEEP", 0.3),
        _row("b", "DELETE", 0.4),
    ]

    result = cluster_bootstrap(rows, draws=200, seed=7)

    assert result["map_iou_delta"]["observed"] == 0.25
    assert result["map_iou_delta"]["ci_low"] > 0.0


def test_cluster_bootstrap_resamples_draws_with_missing_denominators():
    rows = [
        _row("stable", "KEEP", 0.1),
        _row("update", "ADD", 0.2),
    ]

    result = cluster_bootstrap(rows, draws=200, seed=11)

    assert result["false_edit_rate"]["observed"] == 0.0
    assert result["missed_edit_rate"]["observed"] == 0.0
