from scripts.bootstrap_sn7_changemamba_safe_commit import (
    cluster_bootstrap,
    grouped_statistics,
    paired_from_grouped,
    paired_metrics,
    seed_mean_cluster_bootstrap,
)


def _rows(split, aoi_count):
    rows = []
    for aoi in range(aoi_count):
        for target, prediction, confidence, delta in (
            ("ADD", "ADD", 0.9, 0.4),
            ("KEEP", "ADD", 0.2, -0.4),
            ("ADD", "KEEP", 0.8, 0.0),
            ("KEEP", "KEEP", 0.8, 0.0),
        ):
            rows.append(
                {
                    "sample_id": f"{split}-{len(rows)}",
                    "aoi_id": f"aoi-{aoi}",
                    "target_edit": target,
                    "predicted_edit": prediction,
                    "confidence": confidence,
                    "prior_map_iou": 0.5,
                    "committed_map_iou": 0.5 + delta,
                }
            )
    return rows


def test_paired_metrics_uses_saved_safe_commit_acceptance():
    rows = _rows("val", 3)
    for row in rows:
        row["safe_commit_accepted"] = (
            row["predicted_edit"] != "KEEP" and row["confidence"] > 0.5
        )

    result = paired_metrics(rows)

    assert result["safe_minus_always"]["false_edit_rate"] < 0.0
    assert result["safe_minus_always"]["map_iou_delta"] > 0.0


def test_cluster_bootstrap_reports_paired_delta_interval():
    rows = _rows("val", 5)
    for row in rows:
        row["safe_commit_accepted"] = (
            row["predicted_edit"] != "KEEP" and row["confidence"] > 0.5
        )

    result = cluster_bootstrap(rows, draws=100, seed=7)

    delta = result["safe_minus_always"]["map_iou_delta"]
    assert delta["observed"] > 0.0
    assert delta["ci_low"] > 0.0


def test_sufficient_statistics_match_direct_metrics():
    rows = _rows("val", 5)
    for row in rows:
        row["safe_commit_accepted"] = (
            row["predicted_edit"] != "KEEP" and row["confidence"] > 0.5
        )

    direct = paired_metrics(rows)
    grouped = paired_from_grouped(
        grouped_statistics(rows),
        sorted({row["aoi_id"] for row in rows}),
    )

    assert grouped == direct


def test_seed_mean_bootstrap_uses_shared_aoi_draws():
    first = _rows("val", 5)
    second = _rows("val", 5)
    for rows in (first, second):
        for row in rows:
            row["safe_commit_accepted"] = (
                row["predicted_edit"] != "KEEP" and row["confidence"] > 0.5
            )

    result = seed_mean_cluster_bootstrap(
        [first, second], draws=100, seed=11
    )

    delta = result["safe_minus_always"]["map_iou_delta"]
    assert delta["observed"] > 0.0
    assert delta["ci_low"] > 0.0
