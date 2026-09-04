import numpy as np
import pytest
from affine import Affine

pytest.importorskip("shapely")

from activemap.models import (
    CandidateHypothesis,
    EditOperation,
    EditRecord,
    EpisodeRecord,
    GeoJSONGeometry,
)
from scripts.evaluate_online_persistent_map_maintenance import (
    SafeCommitGate,
    build_contiguous_chains,
    evaluate_chains,
)


def polygon(x0, x1):
    return GeoJSONGeometry(
        type="Polygon",
        coordinates=[[[x0, 0], [x1, 0], [x1, 1], [x0, 1], [x0, 0]]],
    )


def episode(identifier, timestamp, prior, target):
    return EpisodeRecord(
        episode_id=identifier,
        aoi_id="aoi",
        anchor_timestamp=timestamp,
        split="val",
        source_dataset="fixture",
        map_before="before.geojson",
        target_map="after.geojson",
        prior_geometry=prior,
        target_geometry=target,
        hypothesis=CandidateHypothesis(
            op=EditOperation.RESHAPE,
            object_id="object-1",
            geometry=target,
            source="fixture",
        ),
        evidence_catalog=[],
        gt_edit=EditRecord(
            op=EditOperation.RESHAPE,
            object_id="object-1",
            geometry=target,
        ),
        is_synthetic=False,
        derivation_version="fixture",
    )


class RecorderPredictor:
    def __init__(self):
        self.priors = []

    def predict(self, image, prior):
        self.priors.append(np.asarray(prior).copy())
        # ADD is the second operation in the project enum and changes the carry state.
        return {
            "mask_probability": np.ones_like(prior, dtype=np.float32),
            "edit_probabilities": np.asarray([0.01, 0.98, 0.005, 0.005]),
            "confidence": 0.99,
        }


def reader(_episode, prior_geometry):
    # The reader exposes the supplied carried geometry through the prior raster.
    prior_value = 0.0 if prior_geometry is None else 1.0
    prior = np.full((8, 8), prior_value, dtype=np.float32)
    return (
        np.zeros((3, 8, 8), dtype=np.float32),
        prior,
        np.ones((8, 8), dtype=np.float32),
        np.ones((8, 8), dtype=np.float32),
        Affine.identity(),
        "anchor",
    )


def test_online_carry_is_rerasterized_into_the_next_updater_input():
    first = episode("one", "2018_02", polygon(0, 1), polygon(0, 2))
    second = episode("two", "2018_03", polygon(0, 2), polygon(0, 3))
    chains = build_contiguous_chains([first, second], minimum_length=2, continuity_tolerance=1e-6)
    predictor = RecorderPredictor()

    records = evaluate_chains(
        predictor,
        chains,
        input_reader=reader,
        gate=SafeCommitGate(0.5, 0.99),
    )

    carry_second = [
        row for row in records
        if row["step"] == 1 and row["branch"] == "carry_always_commit"
    ][0]
    assert carry_second["prior_source"] == "carried_committed_state"
    assert carry_second["input_prior_hash"] != carry_second["canonical_prior_hash"]
    assert any(not row["input_prior_matches_canonical"] for row in records if row["step"] == 1)
    assert all(row["commit_accepted"] == row["writeback_changed"] for row in records)
    # Reset, always-carry, and safe-carry each invoke the predictor at each timestamp.
    assert len(predictor.priors) == 6
