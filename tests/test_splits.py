import pandas as pd
import pytest

from activemap.data.manifest import select_pilot_groups
from activemap.data.splits import assert_no_group_leakage, assign_group_splits


def test_group_splits_are_deterministic_and_leakage_safe() -> None:
    frame = pd.DataFrame(
        {
            "aoi_id": [f"aoi-{index}" for index in range(10) for _ in range(3)],
            "timestamp": [f"2019_0{month}" for _ in range(10) for month in range(1, 4)],
        }
    )
    first, first_groups = assign_group_splits(frame, seed=42)
    second, second_groups = assign_group_splits(frame, seed=42)
    assert first_groups == second_groups
    assert first["split"].tolist() == second["split"].tolist()
    assert_no_group_leakage(first)
    assert set(first["split"]) == {"train", "val", "test"}


def test_leakage_is_rejected() -> None:
    frame = pd.DataFrame({"aoi_id": ["same", "same"], "split": ["train", "test"]})
    with pytest.raises(ValueError, match="multiple splits"):
        assert_no_group_leakage(frame)


def test_pilot_selects_complete_groups_from_every_split() -> None:
    frame = pd.DataFrame(
        {
            "aoi_id": [
                *["train-a"] * 3,
                *["train-b"] * 3,
                *["val-a"] * 3,
                *["test-a"] * 3,
            ],
            "split": [*["train"] * 6, *["val"] * 3, *["test"] * 3],
            "timestamp": [f"2019_{index:02d}" for index in range(1, 4)] * 4,
            "has_udm": [False] * 3 + [True] * 3 + [True] * 3 + [True] * 3,
        }
    )
    pilot, selected = select_pilot_groups(
        frame, groups_per_split=1, min_observations=3, seed=4
    )
    assert set(selected) == {"train", "val", "test"}
    assert pilot["aoi_id"].nunique() == 3
    assert len(pilot) == 9
    assert pilot.groupby("aoi_id").size().eq(3).all()

    udm_pilot, udm_selected = select_pilot_groups(
        frame,
        groups_per_split=1,
        min_observations=3,
        seed=4,
        require_udm=True,
    )
    assert udm_selected["train"] == ["train-b"]
    assert udm_pilot.groupby("aoi_id")["has_udm"].any().all()
