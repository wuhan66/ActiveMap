from __future__ import annotations

import pytest

from activemap.features import EVIDENCE_DIM, HYPOTHESIS_DIM, STATE_DIM
from activemap.selector_records import SelectorSample
from scripts.audit_nonkeep_candidate_headroom import (
    deduplicate_source_episodes,
    record,
    summarize,
)


def _sample(*, target: str, aoi_id: str, improved: bool) -> SelectorSample:
    anchor = {
        "final_raster_iou": 0.50,
        "false_edit": False,
        "missed_edit": True,
    }
    candidate = {
        "final_raster_iou": 0.70 if improved else 0.50,
        "false_edit": False,
        "missed_edit": False if improved else True,
    }
    return SelectorSample.model_validate(
        {
            "sample_id": f"{aoi_id}-{target}-{improved}",
            "split": "val",
            "edit_type": "KEEP",
            "hypothesis_features": [0.0] * HYPOTHESIS_DIM,
            "state_features": [0.0] * STATE_DIM,
            "evidence_ids": ["anchor", "candidate"],
            "evidence_features": [[0.0] * EVIDENCE_DIM] * 2,
            "evidence_costs": [0.1, 0.2],
            "false_edit_risks": [0.1, 0.2],
            "oracle_utilities": [0.0, 0.1],
            "metadata": {
                "gt_edit": target,
                "aoi_id": aoi_id,
                "source_episode": f"episode-{aoi_id}",
                "initial_evidence_id": "anchor",
                "executable_outcomes": {"anchor": anchor, "candidate": candidate},
                "evidence_predictions": {
                    "anchor": {"gated_edit": "KEEP"},
                    "candidate": {"gated_edit": target if improved else "KEEP"},
                },
            },
        }
    )


def test_headroom_preflight_passes_only_when_every_edit_has_safe_opportunity() -> None:
    rows = []
    for target in ("ADD", "DELETE", "RESHAPE"):
        for aoi_id in ("aoi_a", "aoi_b"):
            item = record(_sample(target=target, aoi_id=aoi_id, improved=True), headroom_epsilon=1e-6)
            assert item is not None
            rows.append(item)
    result = summarize(
        rows,
        minimum_rows=2,
        minimum_aois=2,
        headroom_epsilon=1e-6,
        bootstrap_draws=100,
        bootstrap_seed=4,
    )
    assert result["gate"]["passes_candidate_recovery_preflight"] is True
    assert result["slices"]["DELETE"]["mean_safe_map_headroom"] == pytest.approx(0.2)
    assert result["slices"]["RESHAPE"]["missed_edit_recovery_fraction"] == 1.0


def test_headroom_preflight_rejects_aliased_candidate_outcomes() -> None:
    rows = []
    for target in ("ADD", "DELETE", "RESHAPE"):
        for aoi_id in ("aoi_a", "aoi_b"):
            item = record(_sample(target=target, aoi_id=aoi_id, improved=False), headroom_epsilon=1e-6)
            assert item is not None
            rows.append(item)
    result = summarize(
        rows,
        minimum_rows=2,
        minimum_aois=2,
        headroom_epsilon=1e-6,
        bootstrap_draws=100,
        bootstrap_seed=4,
    )
    assert result["gate"]["passes_candidate_recovery_preflight"] is False
    assert result["slices"]["ADD"]["candidate_outcome_diversity_fraction"] == 0.0


def test_headroom_audit_deduplicates_budget_step_replicas_from_one_episode() -> None:
    first = record(_sample(target="ADD", aoi_id="aoi_a", improved=True), headroom_epsilon=1e-6)
    second = record(_sample(target="ADD", aoi_id="aoi_a", improved=True), headroom_epsilon=1e-6)
    assert first is not None and second is not None
    rows, duplicates = deduplicate_source_episodes([first, second])
    assert len(rows) == 1
    assert duplicates == 1
