import json
from pathlib import Path

from activemap.agent.records import AgentBelief
from activemap.agent.tool_belief_data import ToolBeliefExample
from activemap.geo_tools.records import GeoToolName, GeoToolResult
from activemap.models import EditOperation
from scripts.audit_tool_belief_data import audit


def _belief(probabilities: list[float], confidence: float) -> AgentBelief:
    return AgentBelief(
        edit_probabilities=probabilities,
        confidence=confidence,
        uncertainty=0.5,
        geometry_delta=[0.0] * 8,
    )


def _write_split(path: Path, split: str) -> None:
    prior = _belief([0.7, 0.1, 0.1, 0.1], 0.8)
    target = _belief([0.5, 0.2, 0.2, 0.1], 0.65)
    rows = []
    for index, operation in enumerate(EditOperation):
        for tool, kind, belief, outputs, cost in (
            (
                GeoToolName.IMAGE_QUALITY,
                "observational_noop",
                prior,
                {"sharpness": 0.1 + index},
                0.03,
            ),
            (
                GeoToolName.TEMPORAL_CHANGE,
                "teacher_belief_update",
                target,
                {"changed_fraction": 0.1 + index / 10},
                0.15,
            ),
        ):
            key = f"{split}-{index}-{tool.value}"
            rows.append(
                ToolBeliefExample(
                    record_id=key,
                    episode_id=f"{split}-episode-{index}",
                    split=split,
                    evidence_id=f"evidence-{index}",
                    prior_belief=prior,
                    tool_result=GeoToolResult(
                        call_id=key,
                        tool=tool,
                        success=True,
                        outputs=outputs,
                        cost=cost,
                    ),
                    target_belief=belief,
                    gt_edit=operation,
                    metadata={"target_kind": kind, "test_assets_read": False},
                )
            )
    path.write_text(
        "".join(json.dumps(row.model_dump(mode="json")) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_grounded_data_audit_passes_complete_disjoint_corpus(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_split(train, "train")
    _write_split(val, "val")

    report = audit(
        train,
        val,
        expected_train=8,
        expected_val=8,
        minimum_success_rate=0.99,
    )

    assert report["passed"]
    assert report["episode_counts"]["overlap"] == 0
    assert report["noop_delta_max"]["probability_l1"] == 0.0
    assert report["semantic_probability_delta"]["mean"] > 0.0


def test_grounded_data_audit_fails_wrong_expected_size(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_split(train, "train")
    _write_split(val, "val")

    report = audit(
        train,
        val,
        expected_train=9,
        expected_val=8,
        minimum_success_rate=0.99,
    )

    assert not report["passed"]
    assert any("train record count" in item for item in report["failures"])
