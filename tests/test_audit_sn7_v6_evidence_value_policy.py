from __future__ import annotations

import numpy as np

from activemap.models import EditOperation
from activemap.selector_records import SelectorSample
from scripts.audit_sn7_v6_evidence_value_policy import audit_row, summarize


def sample() -> SelectorSample:
    return SelectorSample(
        sample_id="sample",
        split="val",
        edit_type=EditOperation.ADD,
        hypothesis_features=[0.0] * 16,
        state_features=[0.0] * 8,
        evidence_ids=["a", "b"],
        evidence_features=[[0.0] * 13, [0.0] * 13],
        evidence_costs=[1.0, 1.0],
        false_edit_risks=[0.1, 0.2],
        oracle_utilities=[0.4, -0.2],
        stop_utility=0.0,
        metadata={
            "gt_edit": "ADD",
            "source_episode": "source",
            "aoi_id": "aoi",
            "oracle_step": 0,
            "executable_outcomes": {
                "a": {
                    "final_raster_iou": 0.8,
                    "false_edit": False,
                    "missed_edit": False,
                    "wrong_edit": False,
                },
                "b": {
                    "final_raster_iou": 0.1,
                    "false_edit": True,
                    "missed_edit": False,
                    "wrong_edit": True,
                },
            },
        },
    )


def test_policy_audit_reports_selected_candidate_and_stop() -> None:
    row = audit_row(sample(), np.asarray([0.7, 0.2]), stop_margin=0.3)
    assert row["acquire"] is True
    assert row["selected_evidence_id"] == "a"
    assert row["exact_oracle_candidate"] is True
    assert row["chosen_utility"] == 0.4

    stopped = audit_row(sample(), np.asarray([0.2, 0.1]), stop_margin=0.3)
    assert stopped["acquire"] is False
    assert stopped["chosen_utility"] == 0.0
    summary = summarize([row, stopped])
    assert summary["acquire_rate"] == 0.5
    assert summary["acquire_recall"] == 0.5
