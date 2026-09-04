"""Persistent occupancy-map rollouts for indoor/outdoor navigation episodes.

Policies select evidence using only the committed map state, current pose,
budget, and declared candidate metadata.  Occupancy observations are loaded
only after acquisition.  Consequently, a committed map is the actual input to
the following decision rather than a post-hoc scoring artifact.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from activemap.data.navigation_map import NavigationEvidence, NavigationMapEpisode, NavigationPose

UNKNOWN = np.float32(0.5)
NavigationPolicy = Callable[["NavigationState", tuple[NavigationEvidence, ...]], str | None]
CommitRule = Callable[["NavigationState", NavigationEvidence, np.ndarray], bool]


@dataclass(frozen=True)
class NavigationState:
    """Observable controller state. Ground truth is deliberately absent."""

    committed_map: np.ndarray
    pose: NavigationPose
    remaining_budget: float
    acquired_evidence_ids: tuple[str, ...]
    step: int
    pixels_per_meter: float
    grid_x_min: float = 0.0
    grid_z_max: float = 0.0
    visual_rgb_path: str | None = None
    visual_depth_path: str | None = None


@dataclass(frozen=True)
class NavigationRolloutStep:
    step: int
    evidence_id: str
    motion_cost: float
    committed: bool
    committed_cell_count: int
    remaining_budget: float
    map_quality_before: float
    map_quality_after: float
    coverage_before: float
    coverage_after: float


@dataclass(frozen=True)
class NavigationRolloutResult:
    episode_id: str
    policy: str
    stop_reason: str
    spent_cost: float
    final_map_quality: float
    final_coverage: float
    false_free_rate: float
    false_block_rate: float
    steps: tuple[NavigationRolloutStep, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def always_stop_policy(state: NavigationState, candidates: tuple[NavigationEvidence, ...]) -> None:
    """A no-acquisition control."""
    del state, candidates
    return None


def nearest_policy(
    state: NavigationState, candidates: tuple[NavigationEvidence, ...]
) -> str | None:
    """Choose the nearest affordable declared viewpoint without reading its observation."""
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (_motion_cost(state.pose, row), row.evidence_id),
    ).evidence_id


def cheapest_policy(
    state: NavigationState, candidates: tuple[NavigationEvidence, ...]
) -> str | None:
    """Choose the lowest declared acquisition cost, with stable tie-breaking."""
    del state
    if not candidates:
        return None
    return min(candidates, key=lambda row: (row.cost, row.evidence_id)).evidence_id


def _frontier_cells(committed_map: np.ndarray) -> np.ndarray:
    known_free = committed_map == 0.0
    unknown = committed_map == UNKNOWN
    unknown_neighbor = np.zeros_like(unknown)
    unknown_neighbor[1:, :] |= unknown[:-1, :]
    unknown_neighbor[:-1, :] |= unknown[1:, :]
    unknown_neighbor[:, 1:] |= unknown[:, :-1]
    unknown_neighbor[:, :-1] |= unknown[:, 1:]
    return np.argwhere(known_free & unknown_neighbor)


def frontier_nearest_policy(
    state: NavigationState,
    candidates: tuple[NavigationEvidence, ...],
) -> str | None:
    """Choose a candidate nearest the current map frontier without reading observations."""
    frontiers = _frontier_cells(state.committed_map)
    positioned = [row for row in candidates if row.pose is not None]
    if not positioned:
        return nearest_policy(state, candidates)
    if not len(frontiers):
        return nearest_policy(state, tuple(positioned))

    def distance_to_frontier(row: NavigationEvidence) -> tuple[float, str]:
        assert row.pose is not None
        point = _pose_to_grid(state, row.pose)
        distance = float(np.min(np.sum((frontiers - point) ** 2, axis=1)))
        return distance, row.evidence_id

    return min(positioned, key=distance_to_frontier).evidence_id


def _pose_to_grid(state: NavigationState, pose: NavigationPose) -> np.ndarray:
    """Map a world-space navigation pose to row/column grid coordinates."""
    # Legacy navigation fixtures predate explicit grid origins and stored pose
    # coordinates directly as row/column coordinates.
    if state.grid_x_min == 0.0 and state.grid_z_max == 0.0:
        return np.asarray(
            (pose.y * state.pixels_per_meter, pose.x * state.pixels_per_meter),
            dtype=np.float32,
        )
    return np.asarray(
        (
            (state.grid_z_max - pose.y) * state.pixels_per_meter,
            (pose.x - state.grid_x_min) * state.pixels_per_meter,
        ),
        dtype=np.float32,
    )


def _footprint_mask(state: NavigationState, evidence: NavigationEvidence) -> np.ndarray:
    """Return a declared acquisition footprint without opening its observation."""
    mask = np.zeros(state.committed_map.shape, dtype=bool)
    if evidence.pose is None or evidence.footprint_radius_pixels is None:
        return mask
    center_row, center_col = np.rint(_pose_to_grid(state, evidence.pose)).astype(int)
    radius = evidence.footprint_radius_pixels
    row_start, row_stop = (
        max(0, center_row - radius),
        min(state.committed_map.shape[0], center_row + radius + 1),
    )
    col_start, col_stop = (
        max(0, center_col - radius),
        min(state.committed_map.shape[1], center_col + radius + 1),
    )
    mask[row_start:row_stop, col_start:col_stop] = True
    return mask


def _unknown_footprint_count(state: NavigationState, evidence: NavigationEvidence) -> int:
    """Count currently unknown cells in a declared camera/sensor footprint.

    The footprint is geometry metadata made available before acquisition.  This
    function intentionally never opens ``evidence.path`` or uses target-map
    content, so it remains a valid pre-observation control.
    """
    return int(np.sum((state.committed_map == UNKNOWN) & _footprint_mask(state, evidence)))


def unknown_coverage_policy(
    state: NavigationState,
    candidates: tuple[NavigationEvidence, ...],
) -> str | None:
    """Choose the declared view that covers the most unknown map area."""
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            _unknown_footprint_count(state, row),
            -_motion_cost(state.pose, row),
            row.evidence_id,
        ),
    ).evidence_id


def unknown_per_cost_policy(
    state: NavigationState,
    candidates: tuple[NavigationEvidence, ...],
) -> str | None:
    """Choose the greatest predicted unknown-area gain per unit motion cost."""
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            _unknown_footprint_count(state, row) / _motion_cost(state.pose, row),
            _unknown_footprint_count(state, row),
            row.evidence_id,
        ),
    ).evidence_id


POLICIES: dict[str, NavigationPolicy] = {
    "stop": always_stop_policy,
    "nearest": nearest_policy,
    "cheapest": cheapest_policy,
    "frontier_nearest": frontier_nearest_policy,
    "unknown_coverage": unknown_coverage_policy,
    "unknown_per_cost": unknown_per_cost_policy,
}


def always_commit(
    state: NavigationState,
    evidence: NavigationEvidence,
    proposed_map: np.ndarray,
) -> bool:
    """A transparent baseline that accepts every acquired local observation."""
    del state, evidence, proposed_map
    return True


def consistency_commit(
    state: NavigationState,
    evidence: NavigationEvidence,
    proposed_map: np.ndarray,
    *,
    max_conflict_rate: float = 0.02,
) -> bool:
    """Commit only observations that agree with the already written overlap.

    This is a deliberately simple Safe-Commit control: it can reject only when
    a newly acquired observation conflicts with previously committed cells. It
    cannot certify a first observation, so noisy-sensor results must report its
    missed-update trade-off alongside false-free/false-block safety.
    """
    if not 0.0 <= max_conflict_rate <= 1.0:
        raise ValueError("max_conflict_rate must be in [0, 1]")
    overlap = (state.committed_map != UNKNOWN) & _footprint_mask(state, evidence)
    if not np.any(overlap):
        return True
    # Values at known cells that would change are the only evidence conflict
    # signal available before deciding whether to write the proposal.
    changed_overlap = overlap & (state.committed_map != proposed_map)
    return float(np.mean(changed_overlap[overlap])) <= max_conflict_rate


COMMIT_RULES: dict[str, CommitRule] = {
    "always": always_commit,
    "consistency": consistency_commit,
}


def _load_occupancy(path: str, *, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    array = np.load(path)
    if array.ndim != 2:
        raise ValueError(f"occupancy artifact must be 2D: {path}")
    normalized = np.asarray(array, dtype=np.float32)
    if expected_shape is not None and normalized.shape != expected_shape:
        raise ValueError(
            f"occupancy shape mismatch for {path}: {normalized.shape} != {expected_shape}"
        )
    if not np.isin(normalized, (0.0, UNKNOWN, 1.0)).all():
        raise ValueError(f"occupancy artifact has invalid encoding: {path}")
    return normalized


def _valid_mask(episode: NavigationMapEpisode, shape: tuple[int, int]) -> np.ndarray:
    raw_path = episode.metadata.get("valid_space_path")
    if not isinstance(raw_path, str):
        return np.ones(shape, dtype=bool)
    valid = np.load(raw_path)
    if valid.shape != shape:
        raise ValueError(f"valid-space shape mismatch for {episode.episode_id}")
    return valid > 0


def _map_quality(committed_map: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    return float(np.mean((committed_map[valid] == target[valid]).astype(np.float32)))


def _coverage(committed_map: np.ndarray, valid: np.ndarray) -> float:
    return float(np.mean((committed_map[valid] != UNKNOWN).astype(np.float32)))


def _false_rates(
    committed_map: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, float]:
    target_occupied = valid & (target == 1.0)
    target_free = valid & (target == 0.0)
    false_free = float(np.sum(target_occupied & (committed_map == 0.0))) / max(
        int(np.sum(target_occupied)), 1
    )
    false_block = float(np.sum(target_free & (committed_map == 1.0))) / max(
        int(np.sum(target_free)), 1
    )
    return false_free, false_block


def _merge_local_observation(
    committed_map: np.ndarray,
    observation: np.ndarray,
) -> tuple[np.ndarray, int]:
    proposed = committed_map.copy()
    revealed = observation != UNKNOWN
    changed = revealed & (proposed != observation)
    proposed[revealed] = observation[revealed]
    return proposed, int(np.sum(changed))


def _motion_cost(current_pose: NavigationPose, evidence: NavigationEvidence) -> float:
    if evidence.pose is None:
        return float(evidence.cost)
    distance = float(np.hypot(evidence.pose.x - current_pose.x, evidence.pose.y - current_pose.y))
    return max(distance, 1.0)


def _pixels_per_meter(episode: NavigationMapEpisode) -> float:
    raw_value = episode.metadata.get("pixels_per_meter", 1.0)
    if not isinstance(raw_value, int | float) or raw_value <= 0.0:
        raise ValueError(f"invalid pixels_per_meter metadata for {episode.episode_id}")
    return float(raw_value)


def _grid_origin(episode: NavigationMapEpisode) -> tuple[float, float]:
    """Read the occupancy-grid world origin, preserving legacy zero-origin maps."""
    x_min = episode.metadata.get("grid_x_min", 0.0)
    z_max = episode.metadata.get("grid_z_max", 0.0)
    if not isinstance(x_min, int | float) or not isinstance(z_max, int | float):
        raise ValueError(f"invalid occupancy-grid origin metadata for {episode.episode_id}")
    return float(x_min), float(z_max)


def rollout_navigation_episode(
    episode: NavigationMapEpisode,
    *,
    policy: NavigationPolicy,
    policy_name: str,
    commit_rule: CommitRule = always_commit,
    max_steps: int | None = None,
    budget_override: float | None = None,
) -> NavigationRolloutResult:
    """Execute one persistent occupancy-map episode with an auditable trace."""
    if episode.map_representation != "occupancy_grid":
        raise ValueError(f"only occupancy_grid episodes are supported: {episode.episode_id}")
    target = _load_occupancy(episode.target_map_path)
    committed = _load_occupancy(episode.initial_map_path, expected_shape=target.shape)
    valid = _valid_mask(episode, target.shape)
    pixels_per_meter = _pixels_per_meter(episode)
    grid_x_min, grid_z_max = _grid_origin(episode)
    budget = episode.budget if budget_override is None else budget_override
    if budget <= 0.0:
        raise ValueError("budget_override must be positive")
    available = {row.evidence_id: row for row in episode.evidence}
    state = NavigationState(
        committed_map=committed,
        pose=episode.start_pose,
        remaining_budget=budget,
        acquired_evidence_ids=(),
        step=0,
        pixels_per_meter=pixels_per_meter,
        grid_x_min=grid_x_min,
        grid_z_max=grid_z_max,
        visual_rgb_path=episode.initial_rgb_path,
        visual_depth_path=episode.initial_depth_path,
    )
    trace: list[NavigationRolloutStep] = []
    stop_reason = "candidate_exhausted"
    limit = max_steps if max_steps is not None else len(available)

    while available and len(trace) < limit:
        affordable = tuple(
            row
            for row in available.values()
            if _motion_cost(state.pose, row) <= state.remaining_budget
        )
        if not affordable:
            stop_reason = "budget_exhausted"
            break
        selected_id = policy(state, affordable)
        if selected_id is None:
            stop_reason = "policy_stop"
            break
        if selected_id not in {row.evidence_id for row in affordable}:
            raise ValueError(
                f"policy selected unavailable evidence {selected_id} in {episode.episode_id}"
            )
        evidence = available.pop(selected_id)
        motion_cost = _motion_cost(state.pose, evidence)
        observation = _load_occupancy(evidence.path, expected_shape=target.shape)
        proposed, changed_cells = _merge_local_observation(state.committed_map, observation)
        quality_before = _map_quality(state.committed_map, target, valid)
        coverage_before = _coverage(state.committed_map, valid)
        committed_now = bool(commit_rule(state, evidence, proposed))
        next_map = proposed if committed_now else state.committed_map
        next_pose = evidence.pose if evidence.pose is not None else state.pose
        next_state = NavigationState(
            committed_map=next_map,
            pose=next_pose,
            remaining_budget=max(state.remaining_budget - motion_cost, 0.0),
            acquired_evidence_ids=(*state.acquired_evidence_ids, evidence.evidence_id),
            step=state.step + 1,
            pixels_per_meter=state.pixels_per_meter,
            grid_x_min=state.grid_x_min,
            grid_z_max=state.grid_z_max,
            visual_rgb_path=evidence.rgb_path or state.visual_rgb_path,
            visual_depth_path=evidence.depth_path or state.visual_depth_path,
        )
        trace.append(
            NavigationRolloutStep(
                step=next_state.step,
                evidence_id=evidence.evidence_id,
                motion_cost=motion_cost,
                committed=committed_now,
                committed_cell_count=changed_cells if committed_now else 0,
                remaining_budget=next_state.remaining_budget,
                map_quality_before=quality_before,
                map_quality_after=_map_quality(next_state.committed_map, target, valid),
                coverage_before=coverage_before,
                coverage_after=_coverage(next_state.committed_map, valid),
            )
        )
        state = next_state
    else:
        if len(trace) >= limit and available:
            stop_reason = "max_steps"

    false_free, false_block = _false_rates(state.committed_map, target, valid)
    return NavigationRolloutResult(
        episode_id=episode.episode_id,
        policy=policy_name,
        stop_reason=stop_reason,
        spent_cost=float(budget - state.remaining_budget),
        final_map_quality=_map_quality(state.committed_map, target, valid),
        final_coverage=_coverage(state.committed_map, valid),
        false_free_rate=false_free,
        false_block_rate=false_block,
        steps=tuple(trace),
    )


def load_navigation_episodes(
    index_path: Path, *, split: Literal["train", "val", "test"], allow_test: bool = False
) -> list[NavigationMapEpisode]:
    """Load a split-filtered index while preserving the test lock."""
    if split == "test" and not allow_test:
        raise PermissionError("test navigation episodes are locked")
    episodes: list[NavigationMapEpisode] = []
    for line_number, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        episode = NavigationMapEpisode.model_validate_json(line)
        if episode.split != split:
            raise ValueError(
                f"{index_path}:{line_number} has split={episode.split}, expected {split}"
            )
        episodes.append(episode)
    if not episodes:
        raise ValueError(f"no {split} navigation episodes in {index_path}")
    return episodes


def evaluate_navigation_rollouts(
    episodes: Iterable[NavigationMapEpisode],
    *,
    policy_names: tuple[str, ...],
    max_steps: int | None,
    commit_rule: CommitRule = always_commit,
    budget_override: float | None = None,
) -> tuple[list[dict[str, object]], list[NavigationRolloutResult]]:
    """Evaluate no-leakage controls over a shared navigation episode set."""
    invalid = sorted(set(policy_names) - set(POLICIES))
    if invalid:
        raise ValueError(f"unknown navigation policies: {invalid}")
    rows: list[NavigationRolloutResult] = []
    cached = list(episodes)
    if not cached:
        raise ValueError("navigation evaluation requires at least one episode")
    for policy_name in policy_names:
        for episode in cached:
            rows.append(
                rollout_navigation_episode(
                    episode,
                    policy=POLICIES[policy_name],
                    policy_name=policy_name,
                    commit_rule=commit_rule,
                    max_steps=max_steps,
                    budget_override=budget_override,
                )
            )
    return summarize_navigation_results(rows), rows


def summarize_navigation_results(
    results: Iterable[NavigationRolloutResult],
) -> list[dict[str, object]]:
    """Aggregate built-in and externally supplied navigation policies identically."""
    rows = list(results)
    if not rows:
        raise ValueError("navigation result summary requires at least one rollout")
    summaries: list[dict[str, object]] = []
    for policy_name in sorted({row.policy for row in rows}):
        selected = [row for row in rows if row.policy == policy_name]
        summaries.append(
            {
                "policy": policy_name,
                "episode_count": len(selected),
                "mean_final_map_quality": float(
                    np.mean([row.final_map_quality for row in selected])
                ),
                "mean_final_coverage": float(np.mean([row.final_coverage for row in selected])),
                "mean_spent_cost": float(np.mean([row.spent_cost for row in selected])),
                "mean_steps": float(np.mean([len(row.steps) for row in selected])),
                "mean_commit_rate": float(
                    np.mean(
                        [
                            float(np.mean([step.committed for step in row.steps]))
                            if row.steps
                            else 0.0
                            for row in selected
                        ]
                    )
                ),
                "mean_false_free_rate": float(np.mean([row.false_free_rate for row in selected])),
                "mean_false_block_rate": float(np.mean([row.false_block_rate for row in selected])),
            }
        )
    return summaries


def write_navigation_rollout_results(
    output_dir: Path,
    *,
    summaries: list[dict[str, object]],
    results: list[NavigationRolloutResult],
    index_path: Path,
    split: str,
    max_steps: int | None,
    commit_rule_name: str = "always",
    budget_override: float | None = None,
) -> None:
    """Write one immutable run directory with summaries and per-step traces."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "activemap-navigation-rollout-v1",
                "protocol": {
                    "index_path": str(index_path.resolve()),
                    "split": split,
                    "test_assets_read": False,
                    "max_steps": max_steps,
                    "budget_override": budget_override,
                    "commit_rule": commit_rule_name,
                    "policy_input": "committed_map, pose, budget, candidate metadata only",
                    "observation_access": "after ACQUIRE only",
                    "state_transition": (
                        "COMMIT map becomes next controller state; DEFER preserves map"
                    ),
                },
                "results": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (output_dir / "traces.jsonl").open("x", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.as_dict(), separators=(",", ":")) + "\n")
