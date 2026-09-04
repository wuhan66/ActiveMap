# ruff: noqa: E402

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader

from activemap.agent.records import AgentBelief
from activemap.agent.tool_belief_data import (
    ToolBeliefSequenceExample,
    ToolBeliefSequenceStep,
)
from activemap.agent.tool_belief_model import (
    ToolBeliefResidualNetwork,
    ToolPairBeliefResidualNetwork,
)
from activemap.geo_tools.records import GeoToolName, GeoToolResult
from activemap.models import EditOperation
from scripts.train_tool_belief_recurrent import (
    RecurrentLossWeights,
    ToolBeliefSequenceDataset,
    _safety_feasible,
    _selection_score,
    recurrent_losses,
)


def _belief(index: int, confidence: float) -> AgentBelief:
    probabilities = [0.02] * 4
    probabilities[index] = 0.94
    return AgentBelief(
        edit_probabilities=probabilities,
        confidence=confidence,
        uncertainty=0.1,
        geometry_delta=[0.0] * 8,
    )


def _sequence(operation: EditOperation) -> ToolBeliefSequenceExample:
    index = list(EditOperation).index(operation)
    initial = _belief(0, 0.8)
    target = _belief(index, 0.9)
    steps = []
    for step in range(3):
        steps.append(
            ToolBeliefSequenceStep(
                evidence_id=f"evidence-{operation.value}-{step}",
                quality_result=GeoToolResult(
                    call_id=f"quality-{operation.value}-{step}",
                    tool=GeoToolName.IMAGE_QUALITY,
                    success=True,
                    outputs={"sharpness": 0.1 + step},
                    cost=0.03,
                ),
                temporal_result=GeoToolResult(
                    call_id=f"temporal-{operation.value}-{step}",
                    tool=GeoToolName.TEMPORAL_CHANGE,
                    success=True,
                    outputs={"changed_fraction": 0.2 + step / 10},
                    cost=0.15,
                ),
                individual_target_belief=target,
                cumulative_target_belief=target,
            )
        )
    return ToolBeliefSequenceExample(
        sequence_id=f"sequence-{operation.value}",
        episode_id=f"episode-{operation.value}",
        split="train",
        initial_belief=initial,
        steps=steps,
        gt_edit=operation,
        metadata={"test_assets_read": False},
    )


def test_recurrent_loss_runs_all_steps_with_explicit_safety_terms(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    rows = [_sequence(EditOperation.KEEP), _sequence(EditOperation.ADD)]
    path.write_text(
        "".join(json.dumps(row.model_dump(mode="json")) + "\n" for row in rows),
        encoding="utf-8",
    )
    dataset = ToolBeliefSequenceDataset(path)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    model = ToolBeliefResidualNetwork(hidden_dim=16, dropout=0.0)

    total, components, predictions = recurrent_losses(
        batch,
        model,
        torch.ones(4),
        max_logit_delta=1.5,
        geometry_scale=0.15,
        loss_weights=RecurrentLossWeights(),
    )

    assert torch.isfinite(total)
    assert len(predictions) == 3
    assert set(components) == {
        "teacher_kl",
        "operation",
        "calibration",
        "geometry",
        "false_edit",
        "missed_edit",
        "noop_probability",
        "noop_confidence",
        "noop_geometry",
    }
    assert float(components["false_edit"]) > 0.0
    assert float(components["missed_edit"]) > 0.0


def test_selection_score_penalizes_recurrent_safety_regression() -> None:
    safe = {
        "step3_macro_f1": 0.6,
        "step3_false_edit_rate": 0.1,
        "baseline_false_edit_rate": 0.1,
        "step3_missed_edit_rate": 0.2,
        "baseline_missed_edit_rate": 0.2,
        "noop_probability": 0.001,
    }
    unsafe = {**safe, "step3_false_edit_rate": 0.3}

    assert _selection_score(safe, 0.02) > _selection_score(unsafe, 0.02)
    assert _safety_feasible(safe, 0.02)
    assert not _safety_feasible(unsafe, 0.02)


def test_paired_fusion_keeps_quality_noop_exact_by_construction(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    rows = [_sequence(EditOperation.KEEP), _sequence(EditOperation.ADD)]
    path.write_text(
        "".join(json.dumps(row.model_dump(mode="json")) + "\n" for row in rows),
        encoding="utf-8",
    )
    batch = next(iter(DataLoader(ToolBeliefSequenceDataset(path), batch_size=2)))
    model = ToolPairBeliefResidualNetwork(hidden_dim=16, dropout=0.0)

    total, components, predictions = recurrent_losses(
        batch,
        model,
        torch.ones(4),
        max_logit_delta=1.5,
        geometry_scale=0.15,
        fusion_mode="paired",
        loss_weights=RecurrentLossWeights(),
    )

    assert torch.isfinite(total)
    assert len(predictions) == 3
    assert float(components["noop_probability"]) == 0.0
    assert float(components["noop_confidence"]) == 0.0
    assert float(components["noop_geometry"]) == 0.0


def test_dataset_baseline_uses_explicit_recommended_edit(tmp_path: Path) -> None:
    row = _sequence(EditOperation.KEEP)
    row.initial_belief = row.initial_belief.model_copy(
        update={"recommended_edit": EditOperation.DELETE}
    )
    path = tmp_path / "train.jsonl"
    path.write_text(row.model_dump_json() + "\n", encoding="utf-8")

    dataset = ToolBeliefSequenceDataset(path)

    assert int(dataset.initial_prediction[0]) == list(EditOperation).index(
        EditOperation.DELETE
    )
    assert float(dataset.initial_recommendation[0, -1]) == 1.0
