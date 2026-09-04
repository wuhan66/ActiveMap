from __future__ import annotations

import json

import pytest

from scripts.evaluate_nonkeep_edit_recovery import (
    OPERATIONS,
    load_rows,
    promotion_gate,
    summarize_comparison,
    validate_inputs,
)


def _row(
    *,
    task_id: str,
    aoi_id: str,
    target: str,
    after: float,
    tool_called: bool,
    test_assets_read: bool = False,
) -> dict:
    return {
        "task_id": task_id,
        "aoi_id": aoi_id,
        "budget": 1.5,
        "target": f"COMMIT:{target}",
        "effective_operation": target,
        "prior_raster_iou": 0.5,
        "raster_iou": after,
        "episode_utility_v2_balanced": after - 0.5,
        "false_edit": False,
        "missed_edit": False,
        "wrong_edit": False,
        "terminal_correct": True,
        "writeback_changed": True,
        "semantic_tool_called": tool_called,
        "spent_cost": 0.3 if tool_called else 0.0,
        "split": "val",
        "test_assets_read": test_assets_read,
    }


def _write(path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_nonkeep_audit_requires_paired_support_and_promotes_only_safe_gain(tmp_path) -> None:
    inputs = {}
    for seed in ("1", "2", "3"):
        inputs[seed] = {}
        for policy, after, tool_called in (("reference", 0.55, False), ("candidate", 0.70, True)):
            rows = []
            for aoi in ("aoi_a", "aoi_b"):
                for operation in OPERATIONS:
                    rows.append(
                        _row(
                            task_id=f"{aoi}-{operation}",
                            aoi_id=aoi,
                            target=operation,
                            after=after,
                            tool_called=tool_called,
                        )
                    )
            path = tmp_path / f"{seed}_{policy}.jsonl"
            _write(path, rows)
            inputs[seed][policy], _ = load_rows(path)

    seeds, _, aois = validate_inputs(
        inputs,
        reference_policy="reference",
        candidate_policy="candidate",
        expected_seed_count=3,
    )
    comparisons = {
        name: summarize_comparison(
            inputs,
            seeds=seeds,
            reference_policy="reference",
            candidate_policy="candidate",
            aoi_ids=aois,
            operation_filter=None if name == "ALL_NONKEEP" else name,
            repetitions=100,
            bootstrap_seed=4,
        )
        for name in ("ALL_NONKEEP", *OPERATIONS)
    }
    gate = promotion_gate(
        comparisons,
        seed_count=3,
        aoi_count=2,
        minimum_aois=2,
        minimum_rows_per_operation_per_seed=1,
    )

    delta = comparisons["ALL_NONKEEP"]["candidate_minus_reference"]
    assert delta["map_quality_gain"]["observed_delta"] == pytest.approx(0.15)
    assert delta["false_edit_rate"]["ci95_high"] == 0.0
    assert delta["missed_edit_rate"]["ci95_high"] == 0.0
    assert gate["passes_strict_posthoc_diagnostic_gate"] is True


def test_nonkeep_audit_rejects_test_provenance(tmp_path) -> None:
    path = tmp_path / "invalid.jsonl"
    _write(
        path,
        [
            _row(
                task_id="x",
                aoi_id="aoi_a",
                target="ADD",
                after=0.6,
                tool_called=False,
                test_assets_read=True,
            )
        ],
    )
    with pytest.raises(ValueError, match="validation-only provenance"):
        load_rows(path)
