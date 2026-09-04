from scripts.build_sn7_controller_corruption_paper_assets import (
    _write_latex,
    _write_markdown,
    build_payload,
)


def _comparison(iou=0.2, false_edit=-0.2, missed=0.0, cost=-0.1, utility=0.3):
    def value(delta):
        radius = 0.01
        return {
            "delta": delta,
            "ci95_low": delta - radius,
            "ci95_high": delta + radius,
        }

    return {
        "raster_iou_auc": value(iou),
        "false_edit_auc": value(false_edit),
        "missed_edit_auc": value(missed),
        "spent_cost_auc": value(cost),
        "episode_utility_v2_balanced_auc": value(utility),
    }


def test_joint_claim_gate_passes_only_with_quality_safety_cost_and_audits(tmp_path):
    rows = []
    for severity in (0, 4, 8, 16):
        rows.append(
            {
                "severity": severity,
                "controller": {
                    "notool": {"tool_call_episode_rate": 0.0},
                    "forced": {"tool_call_episode_rate": 0.5},
                    "benefit": {"tool_call_episode_rate": 0.25},
                },
                "writeback": {
                    "notool": {"raster_iou_auc": 0.2, "false_edit_auc": 0.4},
                    "forced": {"raster_iou_auc": 0.4, "false_edit_auc": 0.2},
                    "benefit": {"raster_iou_auc": 0.4, "false_edit_auc": 0.2},
                },
                "paired_writeback": {
                    "benefit_vs_notool": _comparison(),
                    "benefit_vs_forced": _comparison(
                        iou=0.0,
                        false_edit=0.0,
                        cost=-0.1,
                    ),
                },
            }
        )
    payload = build_payload(
        {"rows": rows},
        {"paired_delta": _comparison()},
        {"status": "pass", "forbidden_hits": []},
    )

    assert payload["claim_gate"]["passed"] is True
    assert payload["rows"][1]["forced_quality_recovery_fraction"] == 1.0

    markdown = tmp_path / "table.md"
    latex = tmp_path / "table.tex"
    _write_markdown(payload["rows"], markdown)
    _write_latex(payload["rows"], latex)
    assert "+0.200000 [+0.190000, +0.210000]" in markdown.read_text()
    assert "\\begin{tabular}" in latex.read_text()


def test_joint_claim_gate_rejects_nonnegative_false_edit_interval():
    comparison = _comparison(false_edit=0.0)
    rows = [
        {
            "severity": 4,
            "controller": {
                "notool": {"tool_call_episode_rate": 0.0},
                "forced": {"tool_call_episode_rate": 0.5},
                "benefit": {"tool_call_episode_rate": 0.25},
            },
            "writeback": {
                "notool": {"raster_iou_auc": 0.2, "false_edit_auc": 0.4},
                "forced": {"raster_iou_auc": 0.4, "false_edit_auc": 0.2},
                "benefit": {"raster_iou_auc": 0.4, "false_edit_auc": 0.2},
            },
            "paired_writeback": {
                "benefit_vs_notool": comparison,
                "benefit_vs_forced": _comparison(iou=0.0, false_edit=0.0),
            },
        }
    ]

    payload = build_payload(
        {"rows": rows},
        {"paired_delta": _comparison()},
        {"status": "pass", "forbidden_hits": []},
    )

    assert payload["claim_gate"]["passed"] is False
