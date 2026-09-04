from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from activemap.features import AblationSpec, apply_ablation  # noqa: E402
from activemap.inference import SelectorPredictor  # noqa: E402
from activemap.nn.selector import EvidenceSelector, SelectorConfig  # noqa: E402
from activemap.synthetic import generate_selector_smoke_samples  # noqa: E402
from activemap.training.data import fit_selector_feature_normalizer  # noqa: E402


def test_selector_predictor_applies_checkpoint_normalizer(tmp_path: Path) -> None:
    samples = generate_selector_smoke_samples(sample_count=24, candidate_count=5, seed=6)
    sample = samples[0]
    normalizer = fit_selector_feature_normalizer(samples)
    config = SelectorConfig(hidden_dim=16, dropout=0.0, decision_mode="two_stage")
    model = EvidenceSelector(config).eval()
    checkpoint_path = tmp_path / "selector.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": config.as_dict(),
            "ablation": asdict(AblationSpec()),
            "stop_margin": 0.17,
            "feature_normalizer": normalizer.as_dict(),
        },
        checkpoint_path,
    )
    predictor = SelectorPredictor(checkpoint_path, device="cpu")
    actual = predictor.action_scores(sample)

    hypothesis, evidence, state = normalizer.transform(
        np.asarray(sample.hypothesis_features, dtype=np.float32),
        np.asarray(sample.evidence_features, dtype=np.float32),
        np.asarray(sample.state_features, dtype=np.float32),
    )
    hypothesis, evidence, state = apply_ablation(
        hypothesis, evidence, state, AblationSpec()
    )
    with torch.no_grad():
        expected = model(
            torch.from_numpy(evidence)[None],
            torch.from_numpy(hypothesis)[None],
            torch.from_numpy(state)[None],
        )[0].numpy()
    expected[-1] += 0.17
    assert np.allclose(actual, expected)

    overridden = SelectorPredictor(
        checkpoint_path, device="cpu", stop_margin_override=0.41
    )
    overridden_expected = expected.copy()
    overridden_expected[-1] += 0.41 - 0.17
    assert np.allclose(overridden.action_scores(sample), overridden_expected)
    assert overridden.checkpoint_stop_margin == 0.17
    assert overridden.stop_margin == 0.41
    assert overridden.stop_margin_source == "override"
