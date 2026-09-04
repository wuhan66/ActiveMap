import json
from pathlib import Path

import numpy as np
import pytest

from activemap.data.navigation_map import (
    NavigationEvidence,
    NavigationMapEpisode,
    NavigationPose,
    write_navigation_map_jsonl,
)
from activemap.evaluation.navigation_rollout import (
    NavigationState,
    _footprint_mask,
    _pose_to_grid,
    always_commit,
    consistency_commit,
    evaluate_navigation_rollouts,
    frontier_nearest_policy,
    load_navigation_episodes,
    rollout_navigation_episode,
    unknown_coverage_policy,
    unknown_per_cost_policy,
    write_navigation_rollout_results,
)
from activemap.evaluation.navigation_visualization import (
    _active_crop_slices,
    render_navigation_comparison,
)


def _episode(tmp_path: Path) -> NavigationMapEpisode:
    target = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.float32)
    initial = np.full_like(target, 0.5)
    initial[0, 0] = 0.0
    first = np.full_like(target, 0.5)
    first[1:3, 1:3] = target[1:3, 1:3]
    second = np.full_like(target, 0.5)
    second[0:2, 0:2] = target[0:2, 0:2]
    for name, array in (
        ("target", target),
        ("initial", initial),
        ("first", first),
        ("second", second),
    ):
        np.save(tmp_path / f"{name}.npy", array)
    return NavigationMapEpisode(
        schema_version="activemap-navigation-map-episode-v1",
        episode_id="indoor-1",
        dataset="fixture",
        domain="indoor",
        split="val",
        map_representation="occupancy_grid",
        initial_map_path=str(tmp_path / "initial.npy"),
        target_map_path=str(tmp_path / "target.npy"),
        start_pose=NavigationPose(x=0.0, y=0.0, yaw=0.0),
        evidence=[
            NavigationEvidence(
                evidence_id="first",
                modality="occupancy_crop",
                path=str(tmp_path / "first.npy"),
                pose=NavigationPose(x=1.0, y=1.0, yaw=0.0),
                cost=1.0,
                timestamp=0,
                footprint_radius_pixels=1,
            ),
            NavigationEvidence(
                evidence_id="second",
                modality="occupancy_crop",
                path=str(tmp_path / "second.npy"),
                pose=NavigationPose(x=3.0, y=0.0, yaw=0.0),
                cost=3.0,
                timestamp=1,
                footprint_radius_pixels=1,
            ),
        ],
        budget=10.0,
        metadata={"pixels_per_meter": 1.0},
    )


def test_rollout_commits_map_before_the_next_step(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    result = rollout_navigation_episode(
        episode,
        policy=lambda state, candidates: candidates[0].evidence_id,
        policy_name="ordered",
    )
    assert len(result.steps) == 2
    assert result.steps[0].committed is True
    assert result.steps[1].map_quality_before == result.steps[0].map_quality_after
    assert result.final_map_quality > result.steps[0].map_quality_before


def test_rollout_carries_only_acquired_visual_state_to_next_decision(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    evidence = [
        row.model_copy(
            update={"rgb_path": f"frame-{index}.png", "depth_path": f"frame-{index}.npy"}
        )
        for index, row in enumerate(episode.evidence, 1)
    ]
    episode = episode.model_copy(
        update={
            "initial_rgb_path": "initial.png",
            "initial_depth_path": "initial.npy",
            "evidence": evidence,
        }
    )
    observed_states: list[tuple[str | None, str | None]] = []

    def policy(state: NavigationState, candidates: tuple[NavigationEvidence, ...]) -> str:
        observed_states.append((state.visual_rgb_path, state.visual_depth_path))
        return candidates[0].evidence_id

    rollout_navigation_episode(episode, policy=policy, policy_name="visual-state")

    assert observed_states == [("initial.png", "initial.npy"), ("frame-1.png", "frame-1.npy")]


def test_deferred_observation_does_not_change_the_persistent_map(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    result = rollout_navigation_episode(
        episode,
        policy=lambda state, candidates: candidates[0].evidence_id,
        policy_name="defer",
        commit_rule=lambda state, evidence, proposed: False,
        max_steps=1,
    )
    assert result.steps[0].committed is False
    assert result.steps[0].committed_cell_count == 0
    assert result.steps[0].map_quality_before == result.steps[0].map_quality_after


def test_navigation_rollout_loader_and_writer_lock_test_and_record_protocol(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    index = tmp_path / "val.jsonl"
    write_navigation_map_jsonl([episode], index)
    loaded = load_navigation_episodes(index, split="val")
    summaries, results = evaluate_navigation_rollouts(
        loaded, policy_names=("stop", "nearest", "cheapest"), max_steps=1
    )
    output = tmp_path / "rollout"
    write_navigation_rollout_results(
        output,
        summaries=summaries,
        results=results,
        index_path=index,
        split="val",
        max_steps=1,
    )
    payload = json.loads((output / "summary.json").read_text())
    assert payload["protocol"]["observation_access"] == "after ACQUIRE only"
    assert len((output / "traces.jsonl").read_text().splitlines()) == 3
    with pytest.raises(FileExistsError):
        write_navigation_rollout_results(
            output,
            summaries=summaries,
            results=results,
            index_path=index,
            split="val",
            max_steps=1,
        )
    test_episode = episode.model_copy(update={"split": "test", "test_assets_read": True})
    test_index = tmp_path / "test.jsonl"
    write_navigation_map_jsonl([test_episode], test_index)
    with pytest.raises(PermissionError, match="locked"):
        load_navigation_episodes(test_index, split="test")


def test_always_commit_is_a_baseline_without_ground_truth_access(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    result = rollout_navigation_episode(
        episode,
        policy=lambda state, candidates: candidates[0].evidence_id,
        policy_name="always",
        commit_rule=always_commit,
        max_steps=1,
    )
    assert result.steps[0].committed is True


def test_frontier_policy_uses_committed_map_and_candidate_pose_only(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    state = NavigationState(
        committed_map=np.load(episode.initial_map_path),
        pose=episode.start_pose,
        remaining_budget=episode.budget,
        acquired_evidence_ids=(),
        step=0,
        pixels_per_meter=1.0,
    )
    choice = frontier_nearest_policy(state, tuple(episode.evidence))
    assert choice in {row.evidence_id for row in episode.evidence}


def test_unknown_coverage_controls_use_only_known_footprints(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    state = NavigationState(
        committed_map=np.load(episode.initial_map_path),
        pose=episode.start_pose,
        remaining_budget=episode.budget,
        acquired_evidence_ids=(),
        step=0,
        pixels_per_meter=1.0,
    )
    for policy in (unknown_coverage_policy, unknown_per_cost_policy):
        assert policy(state, tuple(episode.evidence)) in {"first", "second"}


def test_world_coordinates_map_to_offset_occupancy_grid(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    state = NavigationState(
        committed_map=np.full((8, 8), 0.5, dtype=np.float32),
        pose=NavigationPose(x=13.0, y=17.0, yaw=0.0),
        remaining_budget=episode.budget,
        acquired_evidence_ids=(),
        step=0,
        pixels_per_meter=1.0,
        grid_x_min=10.0,
        grid_z_max=20.0,
    )
    evidence = episode.evidence[0].model_copy(
        update={"pose": NavigationPose(x=13.0, y=17.0, yaw=0.0)}
    )

    assert np.array_equal(_pose_to_grid(state, evidence.pose), np.array([3.0, 3.0]))
    assert _footprint_mask(state, evidence)[3, 3]
    assert unknown_coverage_policy(state, (evidence,)) == evidence.evidence_id


def test_consistency_commit_rejects_conflicting_observation(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    initial = np.load(episode.initial_map_path)
    initial[1, 1] = 0.0
    np.save(episode.initial_map_path, initial)
    first = np.load(episode.evidence[0].path)
    first[0, 0] = 0.0
    np.save(episode.evidence[0].path, first)
    result = rollout_navigation_episode(
        episode,
        policy=lambda state, candidates: "first",
        policy_name="conflict",
        commit_rule=lambda state, evidence, proposal: consistency_commit(
            state, evidence, proposal, max_conflict_rate=0.0
        ),
        max_steps=1,
    )
    assert result.steps[0].committed is False


def test_navigation_visualization_replays_committed_trace(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    result = rollout_navigation_episode(
        episode,
        policy=lambda state, candidates: "first",
        policy_name="render",
        max_steps=1,
    )
    output = tmp_path / "comparison.png"
    render_navigation_comparison(
        episode,
        labeled_results=[("Selected control", result)],
        output_path=output,
        panel_width=80,
    )
    assert output.is_file()


def test_navigation_visualization_crops_to_reference_extent() -> None:
    target = np.full((20, 20), 0.5, dtype=np.float32)
    valid = np.zeros((20, 20), dtype=bool)
    target[8:10, 9:11] = 0.0
    valid[8:10, 9:11] = True

    rows, cols = _active_crop_slices(target, valid, padding=1)

    assert (rows.start, rows.stop, cols.start, cols.stop) == (7, 11, 8, 12)


def test_budget_override_changes_rollout_budget_without_mutating_episode(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    result = rollout_navigation_episode(
        episode,
        policy=lambda state, candidates: "first",
        policy_name="short_budget",
        max_steps=1,
        budget_override=1.5,
    )
    assert result.spent_cost == pytest.approx(2**0.5)
    assert episode.budget == 10.0
    with pytest.raises(ValueError, match="budget_override"):
        rollout_navigation_episode(
            episode,
            policy=lambda state, candidates: "first",
            policy_name="invalid_budget",
            budget_override=0.0,
        )
