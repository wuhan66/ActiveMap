import numpy as np
import pytest
from pydantic import ValidationError

from activemap.features import AblationSpec, apply_ablation
from activemap.synthetic import generate_selector_smoke_samples


def test_synthetic_selector_samples_cover_operations() -> None:
    samples = generate_selector_smoke_samples(sample_count=40, candidate_count=6, seed=1)
    assert {sample.edit_type.value for sample in samples} == {
        "KEEP",
        "ADD",
        "DELETE",
        "RESHAPE",
    }
    assert {sample.split for sample in samples} == {"train", "val", "test"}


def test_ablation_masks_only_registered_groups() -> None:
    sample = generate_selector_smoke_samples(sample_count=4, candidate_count=3)[0]
    hypothesis, evidence, state = apply_ablation(
        np.asarray(sample.hypothesis_features),
        np.asarray(sample.evidence_features),
        np.asarray(sample.state_features),
        AblationSpec(
            name="test",
            drop_hypothesis_groups=("edit_type",),
            drop_evidence_groups=("time",),
        ),
    )
    assert np.all(hypothesis[:4] == 0)
    assert np.all(evidence[:, 2:4] == 0)
    assert state.shape == (8,)

    with pytest.raises(ValueError, match="unknown feature group"):
        apply_ablation(
            hypothesis,
            evidence,
            state,
            AblationSpec(drop_evidence_groups=("not-a-group",)),
        )


def test_selector_record_rejects_wrong_dimensions() -> None:
    sample = generate_selector_smoke_samples(sample_count=4, candidate_count=3)[0]
    payload = sample.model_dump()
    payload["state_features"] = [0.0]
    with pytest.raises(ValidationError, match="state_features"):
        type(sample).model_validate(payload)
