from __future__ import annotations

import json
import math
from pathlib import Path

from scripts.aggregate_sn7_v5_factorial_writebacks import (
    FORCED_ACQUISITION_POLICY,
    POLICIES,
    aggregate,
)


def _row(task: str, *, aoi: str, target: str, final: float) -> dict:
    operation = target.removeprefix("COMMIT:")
    return {
        "task_id": task,
        "aoi_id": aoi,
        "budget": 1.5,
        "target": target,
        "split": "val",
        "test_assets_read": False,
        "raster_iou": final,
        "prior_raster_iou": 0.5,
        "spent_cost": 0.2,
        "writeback_changed": True,
        "effective_operation": operation,
        "vector_delta_topology_valid": True,
        "semantic_tool_called": True,
        "episode_utility_v2_balanced": final - 0.5,
    }


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_three_seed_factorial_aggregates_all_real_edit_slices(tmp_path: Path) -> None:
    records: dict[int, dict[str, Path]] = {}
    targets = ("COMMIT:ADD", "COMMIT:DELETE", "COMMIT:RESHAPE")
    for seed in (20260817, 20260818, 20260819):
        records[seed] = {}
        for policy in POLICIES:
            final = {
                "direct_commit": 0.50,
                "direct_safe_commit": 0.50,
                "selected_commit": 0.55,
                "selected_safe_commit": 0.60,
            }[policy]
            rows = [
                _row(
                    f"{aoi}-{operation.lower()}",
                    aoi=aoi,
                    target=f"COMMIT:{operation}",
                    final=final,
                )
                for aoi in ("aoi-a", "aoi-b")
                for operation in (target.removeprefix("COMMIT:") for target in targets)
            ]
            records[seed][policy] = _write(tmp_path / f"{seed}_{policy}.jsonl", rows)

    result = aggregate(records, repetitions=100, bootstrap_seed=7)

    selection = result["selection_factor"]["paired_delta"]["final_map_quality"]
    assert math.isclose(selection["delta"], 0.1)
    assert selection["ci95_low"] > 0.0
    assert set(result["operation_slices"]) == {"ADD", "DELETE", "RESHAPE"}
    assert result["promotion"]["eligible_for_extension_claim"] is False
    assert result["promotion"]["forced_acquisition_cost_control"] == "not_evaluated"


def test_forced_acquisition_control_requires_lower_selected_cost(tmp_path: Path) -> None:
    records: dict[int, dict[str, Path]] = {}
    for seed in (20260817, 20260818, 20260819):
        records[seed] = {}
        for policy in POLICIES + (FORCED_ACQUISITION_POLICY,):
            rows = []
            for operation in ("ADD", "DELETE", "RESHAPE"):
                row = _row(
                    f"shared-{operation.lower()}",
                    aoi="aoi-a",
                    target=f"COMMIT:{operation}",
                    final=0.60 if policy == "selected_safe_commit" else 0.50,
                )
                row["spent_cost"] = 0.10 if policy == "selected_safe_commit" else 0.40
                rows.append(row)
            records[seed][policy] = _write(tmp_path / f"{seed}_{policy}.jsonl", rows)

    result = aggregate(records, repetitions=100, bootstrap_seed=7)

    control = result["promotion"]["forced_acquisition_cost_control"]
    assert control["baseline"] == FORCED_ACQUISITION_POLICY
    assert control["candidate"] == "selected_safe_commit"
    assert control["paired_delta"]["additional_cost"]["ci95_high"] < 0.0


def test_promotion_requires_nonpositive_selection_missed_edit_interval(tmp_path: Path) -> None:
    """The executable V5 gate cannot promote quality gains bought by deferral."""
    records: dict[int, dict[str, Path]] = {}
    for seed in (20260817, 20260818, 20260819):
        records[seed] = {}
        for policy in POLICIES + (FORCED_ACQUISITION_POLICY,):
            rows = []
            for operation in ("ADD", "DELETE", "RESHAPE"):
                final = 0.60 if policy in {"selected_commit", "selected_safe_commit"} else 0.50
                row = _row(
                    f"shared-{operation.lower()}",
                    aoi="aoi-a",
                    target=f"COMMIT:{operation}",
                    final=final,
                )
                row["spent_cost"] = 0.10 if policy == "selected_safe_commit" else 0.40
                if policy == "selected_safe_commit" and operation == "ADD":
                    row["effective_operation"] = "KEEP"
                    row["writeback_changed"] = False
                rows.append(row)
            records[seed][policy] = _write(tmp_path / f"{seed}_{policy}.jsonl", rows)

    result = aggregate(records, repetitions=100, bootstrap_seed=7)

    promotion = result["promotion"]
    assert promotion["checks"]["selection_final_map_quality_lower_positive"] is True
    assert promotion["checks"]["selection_false_edit_upper_nonpositive"] is True
    assert promotion["checks"]["selection_missed_edit_upper_nonpositive"] is False
    assert promotion["checks"]["selected_cost_upper_strictly_below_forced"] is True
    assert promotion["eligible_for_extension_claim"] is False
