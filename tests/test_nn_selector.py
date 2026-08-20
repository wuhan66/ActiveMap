from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from activemap.features import AblationSpec, EVIDENCE_DIM  # noqa: E402
from activemap.nn.selector import EvidenceSelector, SelectorConfig  # noqa: E402
from activemap.synthetic import (  # noqa: E402
    generate_selector_smoke_samples,
    write_selector_samples,
)
from activemap.training.data import (  # noqa: E402
    SelectorDataset,
    fit_selector_feature_normalizer,
)
from activemap.training.selector import (  # noqa: E402
    select_stop_margin,
    selector_loss_components,
    split_fit_calibration_samples,
    two_stage_selector_loss_components,
    train_selector,
)


def test_stop_margin_calibration_respects_safety_before_utility() -> None:
    margins = np.asarray([4.0, 3.0, 2.0, 1.0])
    deltas = np.asarray([0.1, -0.01, -0.01, 1.0])
    stops = np.zeros(4)
    target_acquire = np.asarray([True, False, False, True])

    unconstrained = select_stop_margin(margins, deltas, stops, target_acquire)
    constrained = select_stop_margin(
        margins,
        deltas,
        stops,
        target_acquire,
        max_false_call_rate=0.25,
        max_harmful_call_fraction=0.30,
        min_acquire_recall=0.10,
    )

    assert unconstrained["acquire_rate"] == 1.0
    assert constrained["acquire_rate"] == 0.25
    assert constrained["stop_margin"] == 3.5
    assert constrained["false_call_rate"] == 0.0
    assert constrained["harmful_call_fraction"] == 0.0
    assert constrained["acquire_recall"] == 0.5
    assert constrained["constraints_satisfied"] == 1.0


def test_stop_margin_calibration_falls_back_safely_until_feasible() -> None:
    result = select_stop_margin(
        np.asarray([2.0, 1.0]),
        np.asarray([-0.1, 1.0]),
        np.zeros(2),
        np.asarray([False, True]),
        max_false_call_rate=0.0,
        max_harmful_call_fraction=0.0,
        min_acquire_recall=1.0,
    )

    assert result["acquire_rate"] == 0.0
    assert result["constraints_satisfied"] == 0.0


def test_generic_and_conditioned_selectors_have_equal_capacity() -> None:
    conditioned = EvidenceSelector(SelectorConfig(condition_on_hypothesis=True))
    generic = EvidenceSelector(SelectorConfig(condition_on_hypothesis=False))
    assert conditioned.parameter_count() == generic.parameter_count()

    evidence = torch.randn(2, 5, EVIDENCE_DIM)
    hypothesis = torch.randn(2, 16)
    state = torch.randn(2, 8)
    mask = torch.tensor([[True, True, True, False, False], [True] * 5])
    logits = conditioned(evidence, hypothesis, state, mask)
    assert logits.shape == (2, 6)
    assert torch.isneginf(logits[0, 3:5]).all()

    no_stop = EvidenceSelector(SelectorConfig(allow_stop=False))
    assert no_stop(evidence, hypothesis, state).shape == (2, 5)


def test_selector_loss_handles_padding_and_utility_rescaling() -> None:
    logits = torch.tensor([[0.2, 1.1, -torch.inf, -0.3]], requires_grad=True)
    utilities = torch.tensor([[1.0, 3.0, -torch.inf, 2.0]])
    targets = torch.tensor([1])
    kwargs = {
        "regret_weight": 0.25,
        "listwise_weight": 0.5,
        "utility_temperature": 0.25,
        "stop_weight": 0.25,
        "include_stop": True,
    }
    first, first_components = selector_loss_components(
        logits, utilities, targets, **kwargs
    )
    second, _ = selector_loss_components(
        logits, utilities * 7.0 + 11.0, targets, **kwargs
    )
    first.backward()
    assert torch.isfinite(first)
    assert torch.isfinite(logits.grad[torch.isfinite(logits)]).all()
    assert torch.allclose(first, second, atol=1e-6)
    assert set(first_components) == {
        "imitation",
        "listwise",
        "expected_regret",
        "stop",
        "utility_regression",
    }


def test_selector_loss_without_stop_is_finite() -> None:
    logits = torch.tensor([[0.1, 0.3]], requires_grad=True)
    utilities = torch.tensor([[0.4, 0.8]])
    total, components = selector_loss_components(
        logits,
        utilities,
        torch.tensor([1]),
        regret_weight=0.25,
        listwise_weight=0.5,
        utility_temperature=0.25,
        stop_weight=0.25,
        include_stop=False,
    )
    total.backward()
    assert torch.isfinite(total)
    assert components["stop"].item() == 0.0


def test_two_stage_gate_is_separate_from_candidate_ranking() -> None:
    model = EvidenceSelector(
        SelectorConfig(hidden_dim=16, dropout=0.0, decision_mode="two_stage")
    )
    evidence = torch.randn(3, 4, EVIDENCE_DIM)
    hypothesis = torch.randn(3, 16)
    state = torch.randn(3, 8)
    mask = torch.tensor([[True, True, False, False], [True] * 4, [True] * 3 + [False]])
    evidence_logits, gate_logits = model.forward_components(
        evidence, hypothesis, state, mask
    )
    logits = model(evidence, hypothesis, state, mask)
    assert torch.allclose(logits[:, :-1], evidence_logits)
    assert torch.allclose(
        evidence_logits.max(dim=-1).values - logits[:, -1], gate_logits
    )


@pytest.mark.parametrize("targets", [torch.tensor([2, 2]), torch.tensor([0, 2])])
def test_two_stage_loss_is_finite_for_stop_and_acquire_batches(targets) -> None:
    evidence_logits = torch.tensor([[0.3, -0.2], [0.1, 0.4]], requires_grad=True)
    gate_logits = torch.tensor([-0.2, 0.3], requires_grad=True)
    utilities = torch.tensor([[0.2, -0.1, 0.0], [-0.2, -0.3, 0.0]])
    total, components = two_stage_selector_loss_components(
        evidence_logits,
        gate_logits,
        utilities,
        targets,
        regret_weight=0.5,
        listwise_weight=0.5,
        utility_temperature=0.1,
        acquire_weight=1.0,
        imitation_weight=1.0,
        utility_regression_weight=1.0,
        utility_scale=1.0,
        gate_utility_weight=0.5,
    )
    total.backward()
    assert torch.isfinite(total)
    assert torch.isfinite(gate_logits.grad).all()
    assert set(components) == {
        "gate_bce",
        "gate_utility",
        "rank_imitation",
        "listwise",
        "expected_regret",
        "utility_regression",
        "candidate_value_sign",
    }


def test_candidate_value_head_can_drive_candidate_decisions() -> None:
    model = EvidenceSelector(
        SelectorConfig(
            hidden_dim=16,
            dropout=0.0,
            decision_mode="two_stage",
            candidate_value_head=True,
            candidate_decision_mode="value",
        )
    )
    evidence = torch.randn(2, 3, EVIDENCE_DIM)
    hypothesis = torch.randn(2, 16)
    state = torch.randn(2, 8)
    rank, values, gate = model.forward_training_components(evidence, hypothesis, state)
    logits = model(evidence, hypothesis, state)
    assert values is not None
    assert rank.shape == values.shape == (2, 3)
    assert torch.allclose(logits[:, :-1], values)
    assert torch.allclose(values.max(dim=-1).values - logits[:, -1], gate)


def test_value_aware_terminal_gate_compares_best_value_with_zero() -> None:
    model = EvidenceSelector(
        SelectorConfig(
            hidden_dim=16,
            dropout=0.0,
            decision_mode="two_stage",
            candidate_value_head=True,
            candidate_decision_mode="value",
            terminal_gate_mode="value",
        )
    )
    evidence = torch.randn(2, 3, EVIDENCE_DIM)
    hypothesis = torch.randn(2, 16)
    state = torch.randn(2, 8)
    logits = model(evidence, hypothesis, state)
    assert torch.allclose(logits[:, -1], torch.zeros(2), atol=1e-6)


def test_two_stage_stop_delta_regression_uses_stop_as_zero_point() -> None:
    evidence_logits = torch.tensor([[0.2, -0.1]], requires_grad=True)
    gate_logits = torch.tensor([0.4], requires_grad=True)
    utilities = torch.tensor([[0.6, 0.3, 0.5]])
    targets = utilities.argmax(dim=-1)
    shared = {
        "regret_weight": 0.0,
        "listwise_weight": 0.0,
        "utility_temperature": 0.25,
        "acquire_weight": 1.0,
        "imitation_weight": 0.0,
        "utility_regression_weight": 1.0,
        "utility_scale": 1.0,
        "gate_utility_weight": 0.0,
    }
    _, absolute = two_stage_selector_loss_components(
        evidence_logits,
        gate_logits,
        utilities,
        targets,
        **shared,
        utility_regression_target="absolute",
    )
    _, stop_delta = two_stage_selector_loss_components(
        evidence_logits,
        gate_logits,
        utilities,
        targets,
        **shared,
        utility_regression_target="stop_delta",
    )
    assert absolute["utility_regression"] != stop_delta["utility_regression"]


def test_selector_training_writes_reproducible_artifacts(tmp_path: Path) -> None:
    samples_path = tmp_path / "samples.jsonl"
    write_selector_samples(
        generate_selector_smoke_samples(sample_count=80, candidate_count=5, seed=3),
        samples_path,
    )
    output_dir = tmp_path / "run"
    config = {
        "seed": 3,
        "data": {"samples": str(samples_path)},
        "model": {"hidden_dim": 16, "dropout": 0.0},
        "training": {
            "device": "cpu",
            "epochs": 2,
            "patience": 2,
            "batch_size": 16,
            "num_workers": 0,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "utility_loss_weight": 0.1,
            "calibrate_stop_margin": True,
            "grad_clip": 1.0,
        },
        "ablation": {
            "name": "test",
            "condition_on_hypothesis": True,
            "allow_stop": True,
            "false_edit_penalty": True,
        },
        "output_dir": str(output_dir),
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    summary = train_selector(config_path)
    assert summary["epochs_completed"] == 2
    assert (output_dir / "best.pt").is_file()
    assert (output_dir / "last.pt").is_file()
    assert (output_dir / "history.jsonl").is_file()
    assert (output_dir / "state.json").is_file()
    assert (output_dir / "metrics.json").is_file()
    checkpoint = torch.load(
        output_dir / "best.pt", map_location="cpu", weights_only=False
    )
    assert np.isfinite(checkpoint["stop_margin"])


def test_fit_calibration_split_has_no_episode_leakage() -> None:
    samples = generate_selector_smoke_samples(
        sample_count=80, candidate_count=5, seed=4
    )
    for index, sample in enumerate(samples):
        sample.metadata["source_episode"] = f"episode-{index // 4}"
    fit, calibration = split_fit_calibration_samples(
        samples, fraction=0.2, seed=9, group_key="source_episode"
    )
    fit_groups = {sample.metadata["source_episode"] for sample in fit}
    calibration_groups = {sample.metadata["source_episode"] for sample in calibration}
    assert fit and calibration
    assert fit_groups.isdisjoint(calibration_groups)
    assert len(fit) + len(calibration) == len(samples)


def test_feature_normalizer_is_fit_only_and_ablation_safe() -> None:
    samples = generate_selector_smoke_samples(
        sample_count=40, candidate_count=5, seed=8
    )
    normalizer = fit_selector_feature_normalizer(samples)
    dataset = SelectorDataset(samples, AblationSpec(), normalizer)
    hypotheses = np.stack(
        [dataset[index]["hypothesis"] for index in range(len(dataset))]
    )
    states = np.stack([dataset[index]["state"] for index in range(len(dataset))])
    evidence = np.concatenate(
        [dataset[index]["evidence"] for index in range(len(dataset))], axis=0
    )
    assert np.allclose(hypotheses.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(states.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(evidence.mean(axis=0), 0.0, atol=1e-5)

    ablated = SelectorDataset(
        samples,
        AblationSpec(drop_hypothesis_groups=("edit_type",)),
        normalizer,
    )[0]
    assert np.all(ablated["hypothesis"][:4] == 0.0)
