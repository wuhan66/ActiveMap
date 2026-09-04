import pytest

pytest.importorskip("matplotlib")

from scripts.plot_sn7_controller_prior_corruption import plot


def test_plot_writes_six_panel_figure_and_manifest(tmp_path):
    rows = []
    for severity in (0, 4, 8, 16):
        rows.append(
            {
                "severity": severity,
                "writeback": {
                    variant: {
                        "raster_iou_auc": 0.5,
                        "false_edit_auc": 0.1,
                        "missed_edit_auc": 0.2,
                        "spent_cost_auc": 1.0,
                        "episode_utility_v2_balanced_auc": 0.3,
                    }
                    for variant in ("notool", "forced", "benefit")
                },
                "controller": {
                    variant: {"tool_call_episode_rate": 0.1}
                    for variant in ("notool", "forced", "benefit")
                },
            }
        )

    plot(
        {
            "schema_version": "sn7-controller-prior-corruption-summary-v1",
            "rows": rows,
        },
        tmp_path,
    )

    assert (tmp_path / "controller_prior_corruption.png").is_file()
    assert (tmp_path / "controller_prior_corruption.pdf").is_file()
    assert (tmp_path / "controller_prior_corruption.svg").is_file()
    assert (tmp_path / "plot_manifest.json").is_file()
