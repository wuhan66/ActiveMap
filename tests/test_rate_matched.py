import json
import sys

import pytest

from activemap.agent.identifiers import public_evidence_id
from activemap.agent.records import AgentBelief, AgentCandidate, AgentObservation
from activemap.models import EditOperation
from activemap.policy.rate_matched import (
    RateMatchedSingleAcquirePolicy,
    achieved_rate,
    admits,
    candidate_score,
    fit_rate_calibration,
)
from activemap.selector_records import SelectorSample
from scripts import evaluate_rate_matched_baselines
from scripts.build_train_call_rate_receipt import build_receipt


def _sample(index: int, *, split: str = "train", aoi: str = "a") -> SelectorSample:
    evidence_ids = [f"e-{index}-0", f"e-{index}-1"]
    confidence = 0.1 + 0.1 * index
    return SelectorSample(
        sample_id=f"{split}-{index}",
        split=split,
        edit_type=EditOperation.KEEP,
        hypothesis_features=[0.0] * 12 + [0.05 * index, 0.65, 0.0, 0.0],
        state_features=[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        evidence_ids=evidence_ids,
        evidence_features=[
            [0.4 + 0.1 * index, 0.8, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 1.0],
            [0.2, 0.7, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 2.0],
        ],
        evidence_costs=[1.0, 2.0],
        false_edit_risks=[0.1, 0.1],
        # Deliberately non-observable fields: changing them must not change the policy.
        oracle_utilities=[1000.0 - index, -1000.0 + index],
        metadata={
            "source_episode": f"{split}-episode-{index}",
            "aoi_id": aoi,
            "budget": 3.0,
            "oracle_step": 0,
            "gt_edit": "KEEP",
            "evidence_predictions": {
                evidence_ids[0]: {
                    "edit_probabilities": [0.7, 0.2, 0.05, 0.05],
                    "confidence": confidence,
                    "geometry_delta": [0.1] + [0.0] * 7,
                },
                evidence_ids[1]: {
                    "edit_probabilities": [0.1, 0.4, 0.3, 0.2],
                    "confidence": 0.9,
                    "geometry_delta": [0.2] + [0.0] * 7,
                },
            },
        },
    )


def _observation(sample: SelectorSample) -> AgentObservation:
    return AgentObservation(
        task_id="task",
        split=sample.split,
        step=0,
        initial_budget=3.0,
        remaining_budget=3.0,
        spent_cost=0.0,
        selected_evidence_ids=[],
        belief=AgentBelief(
            edit_probabilities=[1.0, 0.0, 0.0, 0.0],
            confidence=0.7,
            uncertainty=0.2,
            recommended_edit=EditOperation.KEEP,
        ),
        candidates=[
            AgentCandidate(
                evidence_id=public_evidence_id(evidence_id),
                cost=cost,
                selector_score=1.0,
                features=features,
            )
            for evidence_id, cost, features in zip(
                sample.evidence_ids,
                sample.evidence_costs,
                sample.evidence_features,
                strict=True,
            )
        ],
    )


def test_train_calibration_has_exact_target_count_and_endpoints():
    samples = [_sample(index) for index in range(10)]
    calibration = fit_rate_calibration(samples, "random", 0.3)
    assert calibration.target_call_count == 3
    assert sum(admits(sample, calibration) for sample in samples) == 3
    assert achieved_rate(samples, calibration) == 0.3

    never = fit_rate_calibration(samples, "random", 0.0)
    always = fit_rate_calibration(samples, "random", 1.0)
    assert sum(admits(sample, never) for sample in samples) == 0
    assert sum(admits(sample, always) for sample in samples) == len(samples)


def test_scores_do_not_depend_on_oracle_utility_or_target_edit():
    original = _sample(2)
    changed = original.model_copy(
        update={
            "edit_type": EditOperation.RESHAPE,
            "oracle_utilities": [-9999.0, 9999.0],
            "metadata": {**original.metadata, "gt_edit": "RESHAPE"},
        }
    )
    for mode in ("random", "uncertainty", "low_confidence", "clear_per_cost", "cheap_positive", "minimum_entropy"):
        assert candidate_score(original, mode, 0) == candidate_score(changed, mode, 0)
        assert candidate_score(original, mode, 1) == candidate_score(changed, mode, 1)


def test_policy_exposes_only_public_evidence_and_acquires_once():
    sample = _sample(1)
    calibration = fit_rate_calibration([sample], "cheap_positive", 1.0)
    policy = RateMatchedSingleAcquirePolicy(sample, calibration)
    first = policy.act(_observation(sample))
    assert first.action.value == "ACQUIRE"
    assert first.evidence_id in {public_evidence_id(item) for item in sample.evidence_ids}

    second = policy.act(_observation(sample).model_copy(update={
        "selected_evidence_ids": [first.evidence_id],
    }))
    assert second.action.value == "REJECT"


def test_validation_runner_writes_test_free_receipt(tmp_path, monkeypatch):
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    train_path.write_text(
        "".join(_sample(index).model_dump_json() + "\n" for index in range(6)),
        encoding="utf-8",
    )
    val_path.write_text(
        "".join(
            _sample(index, split="val", aoi=f"aoi-{index % 2}").model_dump_json() + "\n"
            for index in range(6)
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "result"
    target_trace = tmp_path / "target_train.jsonl"
    target_trace.write_text(
        "".join(
            json.dumps(
                {
                    "split": "train",
                    "acquisitions": index % 2,
                    "test_assets_read": False,
                    "source_episode": f"train-episode-{index}",
                    "budget": 3.0,
                    "aoi_id": "a",
                }
            )
            + "\n"
            for index in range(6)
        ),
        encoding="utf-8",
    )
    rate_receipt = tmp_path / "rate_receipt.json"
    rate_receipt.write_text(json.dumps(build_receipt(target_trace)), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_rate_matched_baselines.py",
            str(train_path),
            str(val_path),
            str(output_dir),
            "--target-rate-receipt",
            str(rate_receipt),
            "--target-train-trace",
            str(target_trace),
            "--mode",
            "random",
            "--bootstrap-repetitions",
            "10",
        ],
    )

    evaluate_rate_matched_baselines.main()

    receipt = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert receipt["split"] == "val"
    assert receipt["test_assets_read"] is False
    assert receipt["protocol"]["calibration_split"] == "train"
    assert receipt["target_source"]["type"] == "predeclared_train_derived_rate"
    assert (output_dir / "rate_matched_random.jsonl").exists()


def test_validation_runner_rejects_trace_that_does_not_match_receipt(tmp_path, monkeypatch):
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    train_path.write_text(
        "".join(_sample(index).model_dump_json() + "\n" for index in range(2)),
        encoding="utf-8",
    )
    val_path.write_text(
        "".join(_sample(index, split="val").model_dump_json() + "\n" for index in range(2)),
        encoding="utf-8",
    )
    trace = tmp_path / "target_train.jsonl"
    trace.write_text(
        "".join(
            json.dumps(
                {
                    "split": "train",
                    "acquisitions": index % 2,
                    "test_assets_read": False,
                    "source_episode": f"train-episode-{index}",
                    "budget": 3.0,
                    "aoi_id": "a",
                }
            )
            + "\n"
            for index in range(2)
        ),
        encoding="utf-8",
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(build_receipt(trace)), encoding="utf-8")
    trace.write_text(trace.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_rate_matched_baselines.py",
            str(train_path),
            str(val_path),
            str(tmp_path / "result"),
            "--target-rate-receipt",
            str(receipt_path),
            "--target-train-trace",
            str(trace),
        ],
    )
    with pytest.raises(ValueError, match="does not match"):
        evaluate_rate_matched_baselines.main()


def test_validation_runner_requires_predeclared_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_rate_matched_baselines.py", "train.jsonl", "val.jsonl", str(tmp_path / "out")],
    )
    with pytest.raises(SystemExit):
        evaluate_rate_matched_baselines._parse_args()


def test_train_call_rate_receipt_rejects_validation_or_test_trace(tmp_path):
    trace = tmp_path / "bad.jsonl"
    trace.write_text(
        json.dumps({"split": "val", "acquisitions": 0, "test_assets_read": False}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="training rows only"):
        build_receipt(trace)


def test_train_call_rate_receipt_rejects_multi_acquisition_trace(tmp_path):
    trace = tmp_path / "bad.jsonl"
    trace.write_text(
        json.dumps({"split": "train", "acquisitions": 2, "test_assets_read": False}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="0 or 1 acquisition"):
        build_receipt(trace)
