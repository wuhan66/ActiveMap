import json

from activemap.models import EditOperation
from activemap.selector_records import SelectorSample
from scripts.audit_online_runtime_feature_distribution import audit


def sample(sample_id: str) -> SelectorSample:
    return SelectorSample(
        sample_id=sample_id,
        split="train",
        edit_type=EditOperation.ADD,
        hypothesis_features=[0.1] * 16,
        state_features=[0.2] * 8,
        evidence_ids=["candidate"],
        evidence_features=[[0.3] * 13],
        evidence_costs=[0.4],
        false_edit_risks=[0.1],
        oracle_utilities=[0.2],
        metadata={
            "selected_evidence_ids": ["direct"],
            "evidence_predictions": {"direct": {}, "candidate": {}},
        },
    )


def test_audit_compares_training_and_exact_runtime_inputs(tmp_path):
    training = tmp_path / "training.jsonl"
    training.write_text(sample("a").model_dump_json() + "\n", encoding="utf-8")
    runtime = tmp_path / "runtime.jsonl"
    runtime.write_text(
        json.dumps(
            {
                "policy": "active_selective_safe",
                "test_assets_read": False,
                "selector_invoked": True,
                "hypothesis_features": [0.2] * 16,
                "state_features": [0.3] * 8,
                "evidence_features": [[0.4] * 13],
                "evidence_costs": [0.5],
                "selected_evidence_ids": ["direct"],
                "candidate_count": 1,
                "total_evidence_count": 2,
                "source_catalog_candidate_count": 3,
                "runtime_grid_eligible_count": 1,
                "runtime_grid_excluded_count": 1,
                "candidate_minus_stop": -0.2,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = audit(
        training,
        runtime,
        policy="active_selective_safe",
        training_split="train",
        training_sample_limit=10,
        seed=1,
        runtime_step=None,
    )

    assert result["test_assets_read"] is False
    assert result["runtime_inputs"]["selector_invoked_record_count"] == 1
    assert result["support"]["runtime_grid_eligible_fraction"]["mean"] == 0.5
    assert result["support"]["runtime_positive_candidate_margin_rate"] == 0.0
