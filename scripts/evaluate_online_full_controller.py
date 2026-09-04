#!/usr/bin/env python3
# ruff: noqa: E402
"""Train/validation-only true sequential rollout for the frozen ActiveMap controller.

This evaluator closes the distinction between a carried-prior updater audit and
an actual online controller experiment.  At every chronological step and for
every policy it rerasterizes that policy's *executed* vector map into every
candidate image, reruns the frozen updater, rebuilds the observable selector
state, executes the frozen selector/tool/belief/terminal controller, and then
performs a typed vector writeback.  Target geometry is used only after the
terminal action to compute metrics; it is deliberately absent from the runtime
``SelectorSample`` passed to the controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from activemap.agent.active_catalog_tool_gate import SelectiveToolPolicy
from activemap.agent.environment import MapMaintenanceEnv, rollout_agent_policy
from activemap.agent.evidence_value_head import load_evidence_value_predictor
from activemap.agent.identifiers import public_evidence_id
from activemap.agent.post_tool_action_adapter import PostToolActionAdapterPredictor
from activemap.agent.records import AgentAction, AgentActionType, AgentObservation
from activemap.agent.tool_belief_model import (
    PairedToolBeliefUpdater,
    SequentialPairedToolBeliefUpdater,
)
from activemap.agent.tools import GreedyAgentPolicy
from activemap.agent.writeback import EvidenceMaskPrediction, evaluate_typed_writeback
from activemap.evaluation.episode_utility import UTILITY_PROFILES
from activemap.features import HYPOTHESIS_DIM, ONLINE_OBSERVABLE_STATE_CONTRACT, STATE_DIM
from activemap.geometry import geometry_features
from activemap.inference import SelectorPredictor, UpdaterPredictor
from activemap.models import EditOperation, EpisodeRecord
from activemap.oracle.updater_counterfactual import (
    MASK_FEATURE_NAMES,
    MASK_FEATURES_SCHEMA,
    _evidence_features,
    _month_index,
    _target_free_mask_features,
    load_episodes,
)
from activemap.selector_records import SelectorSample
from scripts.audit_true_sequential_rollout import audit as audit_true_sequential_rollout
from scripts.evaluate_active_catalog_closed_loop_baselines import (
    PostToolAdaptedPolicy,
    make_environment,
)
from scripts.evaluate_online_persistent_map_maintenance import (
    SafeCommitGate,
    _apply_delta,
    _episode_geometry,
    _operation_errors,
    _parse_asset_root_maps,
    _remap_episode_assets,
    _transform,
    build_contiguous_chains,
)

DEFAULT_POLICIES = (
    "always_stop",
    "direct_current_hypothesis",
    "active_forced",
    "active_selective",
)
SAFE_COMMIT_POLICIES = frozenset(
    {
        "direct_current_hypothesis_safe",
        "active_forced_safe",
        "active_selective_safe",
    }
)
VALID_POLICIES = (*DEFAULT_POLICIES, *sorted(SAFE_COMMIT_POLICIES))
RUNTIME_ORACLE_CONTRACT = "carried-state-runtime-oracle-v1"


def _base_policy_name(policy_name: str) -> str:
    """Map a Safe Commit variant to its identical pre-commit policy."""

    return policy_name.removesuffix("_safe")


@dataclass
class RecurrentSafeCommitMemory:
    """Observable state for suppressing unsupported immediate write retries."""

    last_step: int | None = None
    last_operation: EditOperation | None = None
    last_accepted: bool | None = None
    last_confidence: float | None = None
    last_prior_hash: str | None = None
    last_auxiliary_evidence_ids: frozenset[str] = field(default_factory=frozenset)
    last_tool_calls: int = 0


def decide_safe_commit(
    metrics: dict[str, Any],
    gate: SafeCommitGate | None,
    memory: RecurrentSafeCommitMemory | None,
    *,
    operation: EditOperation,
    step: int,
    prior_hash: str,
    selected_evidence_ids: list[str],
    direct_evidence_id: str,
    tool_calls: int,
    retry_confidence_margin: float,
) -> tuple[bool, dict[str, Any]]:
    """Apply the stateless gate, then require escalation for an immediate retry."""

    changed = bool(metrics["writeback_changed"])
    confidence = float(metrics["fused_confidence"])
    base_accepted = True if gate is None else gate.accepts(metrics)
    auxiliary_ids = frozenset(selected_evidence_ids) - {direct_evidence_id}
    retry = bool(
        memory is not None
        and changed
        and memory.last_step is not None
        and step == memory.last_step + 1
        and memory.last_accepted is False
        and memory.last_operation == operation
        and memory.last_prior_hash == prior_hash
    )
    previous_confidence = memory.last_confidence if retry and memory is not None else None
    required_confidence = (
        min(1.0, float(previous_confidence) + retry_confidence_margin)
        if previous_confidence is not None
        else None
    )
    new_auxiliary_evidence = bool(
        retry
        and memory is not None
        and auxiliary_ids.difference(memory.last_auxiliary_evidence_ids)
    )
    additional_tool_calls = bool(
        retry and memory is not None and tool_calls > memory.last_tool_calls
    )
    confidence_escalated = bool(
        retry
        and required_confidence is not None
        and confidence >= required_confidence
    )
    evidence_escalated = bool(
        new_auxiliary_evidence or additional_tool_calls or confidence_escalated
    )
    retry_blocked = bool(base_accepted and retry and not evidence_escalated)
    accepted = bool(base_accepted and not retry_blocked)
    if not changed:
        reason = "no_executable_delta"
    elif not base_accepted:
        reason = "base_gate_rejected"
    elif retry_blocked:
        reason = "retry_hysteresis_blocked"
    elif retry:
        reason = "retry_evidence_escalated"
    else:
        reason = "base_gate_accepted"

    if memory is not None and changed:
        memory.last_step = step
        memory.last_operation = operation
        memory.last_accepted = accepted
        memory.last_confidence = confidence
        memory.last_prior_hash = prior_hash
        memory.last_auxiliary_evidence_ids = auxiliary_ids
        memory.last_tool_calls = tool_calls

    return accepted, {
        "safe_commit_base_accepted": bool(base_accepted),
        "safe_commit_decision_reason": reason,
        "safe_commit_immediate_retry": retry,
        "safe_commit_retry_blocked": retry_blocked,
        "safe_commit_evidence_escalated": evidence_escalated,
        "safe_commit_new_auxiliary_evidence": new_auxiliary_evidence,
        "safe_commit_additional_tool_calls": additional_tool_calls,
        "safe_commit_confidence_escalated": confidence_escalated,
        "safe_commit_previous_confidence": previous_confidence,
        "safe_commit_required_confidence": required_confidence,
        "safe_commit_auxiliary_evidence_ids": sorted(auxiliary_ids),
    }


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _geometry_sha256(geometry: Any) -> str:
    """Return a full digest because the sequential audit rejects short aliases."""

    payload = b"empty" if geometry is None or geometry.is_empty else geometry.wkb
    return hashlib.sha256(payload).hexdigest()


def _sha256_prediction(evidence_id: str, prediction: dict[str, Any]) -> str:
    digest = hashlib.sha256(evidence_id.encode())
    for key in ("mask_probability", "edit_probabilities", "confidence", "geometry_delta"):
        value = prediction[key]
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        else:
            digest.update(repr(float(value)).encode())
    return digest.hexdigest()


def _resolve_registered_path(raw_path: str, *, storage_root: Path, project_root: Path) -> Path:
    return Path(
        raw_path.replace("${STORAGE_ROOT}", str(storage_root)).replace(
            "${PROJECT_ROOT}", str(project_root)
        )
    )


@dataclass(frozen=True)
class FrozenControllerAssets:
    updater: Path
    updater_threshold: float
    delta_margin: float
    selector: Path
    selector_stop_margin: float
    tool_gate: Path
    tool_gate_summary: Path
    tool_belief: Path
    post_tool_adapter: Path
    hashes: dict[str, str]


def load_frozen_controller_assets(
    registry_path: Path,
    controller_seed: int,
    *,
    storage_root: Path,
    project_root: Path,
) -> FrozenControllerAssets:
    """Resolve one registry-declared controller and fail closed on hash drift."""

    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or registry.get("test_assets_read") is not False:
        raise ValueError("online rollout requires a validation-only frozen registry")
    shared = registry.get("shared_artifacts", [])
    updater_entry = next((entry for entry in shared if entry.get("id") == "frozen_updater"), None)
    seed_assets = registry.get("seed_artifacts", {}).get(str(controller_seed))
    if not isinstance(updater_entry, dict) or not isinstance(seed_assets, dict):
        raise ValueError("registry lacks frozen updater or requested controller seed")
    protocol = registry.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("registry lacks writeback protocol")
    try:
        updater_threshold = float(protocol["updater_threshold"])
        delta_margin = float(protocol["delta_margin"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("registry writeback protocol is malformed") from error
    if not 0.0 < updater_threshold < 1.0:
        raise ValueError("registry updater threshold must be between zero and one")
    if (
        delta_margin < 0.0
        or updater_threshold - delta_margin < 0.0
        or updater_threshold + delta_margin > 1.0
    ):
        raise ValueError("registry delta margin is incompatible with updater threshold")

    resolved: dict[str, tuple[Path, str]] = {}
    entries = {"updater": updater_entry}
    for name in ("selector", "tool_gate", "tool_belief", "post_tool_adapter"):
        entry = seed_assets.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"registry seed {controller_seed} lacks {name}")
        entries[name] = entry
    for name, entry in entries.items():
        if not isinstance(entry.get("path"), str) or not isinstance(entry.get("sha256"), str):
            raise ValueError(f"registry {name} entry is malformed")
        path = _resolve_registered_path(
            entry["path"], storage_root=storage_root, project_root=project_root
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256_path(path)
        if actual != entry["sha256"]:
            raise ValueError(f"registry hash mismatch for {name}: {path}")
        resolved[name] = (path, actual)
    tool_gate_summary = resolved["tool_gate"][0].with_name("summary.json")
    if not tool_gate_summary.is_file():
        raise FileNotFoundError(f"tool gate summary is missing: {tool_gate_summary}")
    hashes = {name: value[1] for name, value in resolved.items()}
    hashes["tool_gate_summary"] = _sha256_path(tool_gate_summary)
    return FrozenControllerAssets(
        updater=resolved["updater"][0],
        updater_threshold=updater_threshold,
        delta_margin=delta_margin,
        selector=resolved["selector"][0],
        selector_stop_margin=float(seed_assets["selector"]["stop_margin"]),
        tool_gate=resolved["tool_gate"][0],
        tool_gate_summary=tool_gate_summary,
        tool_belief=resolved["tool_belief"][0],
        post_tool_adapter=resolved["post_tool_adapter"][0],
        hashes=hashes,
    )


def require_registry_writeback_protocol(
    *,
    threshold: float,
    delta_margin: float,
    assets: FrozenControllerAssets,
) -> None:
    """Keep online execution on the writeback contract used for supervision."""

    if not np.isclose(threshold, assets.updater_threshold, rtol=0.0, atol=1e-9):
        raise ValueError(
            "online writeback threshold must match the frozen registry: "
            f"expected {assets.updater_threshold}, got {threshold}"
        )
    if not np.isclose(delta_margin, assets.delta_margin, rtol=0.0, atol=1e-9):
        raise ValueError(
            "online writeback delta margin must match the frozen registry: "
            f"expected {assets.delta_margin}, got {delta_margin}"
        )


@dataclass(frozen=True)
class RuntimeCandidate:
    evidence_id: str
    image: np.ndarray
    prior: np.ndarray
    valid: np.ndarray
    transform: Any
    raster_shape: tuple[int, int]
    prediction: dict[str, Any]
    clear_fraction: float


@dataclass(frozen=True)
class RuntimeCatalog:
    sample: SelectorSample
    candidates: dict[str, RuntimeCandidate]
    direct_evidence_id: str
    excluded_future_evidence_ids: tuple[str, ...]
    prior_hash: str
    candidate_receipt_hash: str
    direct_receipt_hash: str


def _shares_runtime_grid(reference: RuntimeCandidate, candidate: RuntimeCandidate) -> bool:
    """Whether two predictions can be fused into one executable writeback."""

    return (
        reference.raster_shape == candidate.raster_shape
        and reference.prior.shape == candidate.prior.shape
        and reference.transform == candidate.transform
    )


def _predicted_operation(prediction: dict[str, Any]) -> EditOperation:
    gated = prediction.get("gated_edit")
    if gated is not None:
        return EditOperation(str(gated))
    return list(EditOperation)[
        int(np.argmax(np.asarray(prediction["edit_probabilities"], dtype=np.float64)))
    ]


def _prediction_entropy(probabilities: np.ndarray) -> float:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-8, None)
    values /= values.sum()
    return float(-np.sum(values * np.log(values)) / np.log(len(EditOperation)))


def build_runtime_selector_sample(
    episode: EpisodeRecord,
    prior_geometry: Any,
    candidates: Iterable[RuntimeCandidate],
    *,
    budget: float,
    excluded_future_evidence_ids: tuple[str, ...] = (),
    causal_evidence_only: bool = False,
) -> tuple[SelectorSample, dict[str, RuntimeCandidate], str]:
    """Build target-free frozen-selector features from fresh carried-prior outputs."""

    rows = list(candidates)
    if not rows:
        raise ValueError("a runtime catalog requires at least one candidate")
    by_id = {item.evidence_id: item for item in rows}
    if len(by_id) != len(rows):
        raise ValueError("runtime evidence ids must be unique")

    item_by_id = {item.evidence_id: item for item in episode.evidence_catalog}
    if set(item_by_id) != set(by_id):
        raise ValueError("runtime candidates do not match episode evidence catalog")
    probabilities = np.asarray(
        [
            by_id[item.evidence_id].prediction["edit_probabilities"]
            for item in episode.evidence_catalog
        ],
        dtype=np.float32,
    )
    confidences = np.asarray(
        [
            float(by_id[item.evidence_id].prediction["confidence"])
            for item in episode.evidence_catalog
        ],
        dtype=np.float32,
    )
    geometry_deltas = np.asarray(
        [by_id[item.evidence_id].prediction["geometry_delta"] for item in episode.evidence_catalog],
        dtype=np.float32,
    )
    costs = np.asarray(
        [
            item.cost + 0.25 * (1.0 - by_id[item.evidence_id].clear_fraction)
            for item in episode.evidence_catalog
        ],
        dtype=np.float32,
    )
    initial_index = min(
        range(len(costs)),
        key=lambda index: (float(costs[index]), -float(confidences[index]), index),
    )
    direct = by_id[episode.evidence_catalog[initial_index].evidence_id]
    initial_probability = probabilities[initial_index]
    entropy = _prediction_entropy(initial_probability)
    hypothesis = np.zeros(HYPOTHESIS_DIM, dtype=np.float32)
    hypothesis[:4] = initial_probability
    feature_geometry = prior_geometry
    if feature_geometry is None or feature_geometry.is_empty:
        feature_geometry = _episode_geometry(episode.hypothesis.geometry)
    if feature_geometry is not None and not feature_geometry.is_empty:
        hypothesis[4:12] = geometry_features(feature_geometry)
    hypothesis[12] = entropy
    fused_confidence = float(0.5 * confidences[initial_index] + 0.5)
    hypothesis[13] = fused_confidence
    hypothesis[14] = float(np.mean([np.mean(row.prior >= 0.5) for row in rows]))
    hypothesis[15] = float(min(len(rows) / 32.0, 1.0))
    state = np.zeros(STATE_DIM, dtype=np.float32)
    state[0] = 1.0
    state[2:6] = initial_probability
    state[6] = 1.0 / max(len(rows), 1)
    # Legacy states stored target-derived executable gain here.  The runtime
    # controller must use a feature available before a map edit is evaluated.
    state[7] = fused_confidence
    anchor_timestamp = episode.anchor_timestamp or episode.evidence_catalog[-1].timestamp
    temporal_distances = [
        abs(_month_index(item.timestamp) - _month_index(anchor_timestamp))
        for item in episode.evidence_catalog
    ]
    evidence_features = [
        _evidence_features(
            item,
            anchor_timestamp=anchor_timestamp,
            max_temporal_distance=max(temporal_distances, default=1),
            raster_shape=by_id[item.evidence_id].raster_shape,
            hypothesis_uncertainty=entropy,
            normalized_cost=float(cost / max(float(np.max(costs)), 1e-6)),
            observed_clear_fraction=by_id[item.evidence_id].clear_fraction,
        )
        for item, cost in zip(episode.evidence_catalog, costs, strict=True)
    ]
    evidence_predictions = {
        item.evidence_id: {
            "edit_probabilities": probabilities[index].tolist(),
            "gated_edit": _predicted_operation(by_id[item.evidence_id].prediction).value,
            "confidence": float(confidences[index]),
            "geometry_delta": geometry_deltas[index].tolist(),
            "mask_features": _target_free_mask_features(
                np.asarray(
                    by_id[item.evidence_id].prediction["mask_probability"],
                    dtype=np.float32,
                ),
                by_id[item.evidence_id].prior,
                by_id[item.evidence_id].valid,
            ),
        }
        for index, item in enumerate(episode.evidence_catalog)
    }
    candidate_indices = [
        index
        for index, item in enumerate(episode.evidence_catalog)
        if index != initial_index and _shares_runtime_grid(direct, by_id[item.evidence_id])
    ]
    excluded_misaligned_ids = [
        item.evidence_id
        for index, item in enumerate(episode.evidence_catalog)
        if index != initial_index and index not in candidate_indices
    ]
    # The environment needs a terminal label for its internal reward bookkeeping,
    # but this value is intentionally constant and never exposes the real target.
    sample = SelectorSample(
        sample_id=episode.episode_id,
        split=episode.split,
        edit_type=_predicted_operation(
            by_id[episode.evidence_catalog[initial_index].evidence_id].prediction
        ),
        hypothesis_features=hypothesis.tolist(),
        state_features=state.tolist(),
        evidence_ids=[episode.evidence_catalog[index].evidence_id for index in candidate_indices],
        evidence_features=[evidence_features[index] for index in candidate_indices],
        evidence_costs=[float(costs[index]) for index in candidate_indices],
        false_edit_risks=[
            float(1.0 - probabilities[index].max()) for index in candidate_indices
        ],
        oracle_utilities=[0.0] * len(candidate_indices),
        stop_utility=0.0,
        false_edit_penalty_weight=0.0,
        metadata={
            "source_episode": episode.episode_id,
            "aoi_id": episode.aoi_id,
            "gt_edit": EditOperation.KEEP.value,
            "initial_evidence_id": episode.evidence_catalog[initial_index].evidence_id,
            "selected_evidence_ids": [
                episode.evidence_catalog[initial_index].evidence_id
            ],
            "preacquired_spent_cost": 0.0,
            "preacquired_evidence_penalty": 0.0,
            "budget": float(budget),
            "cost_weight": 0.0,
            "utility_mode": "proxy",
            "runtime_target_free": True,
            "runtime_temporally_causal": causal_evidence_only,
            "runtime_excluded_future_evidence_ids": list(
                excluded_future_evidence_ids
            ),
            "online_state_contract": dict(ONLINE_OBSERVABLE_STATE_CONTRACT),
            "runtime_state7_source": "fused_belief_confidence",
            "runtime_grid_alignment": {
                "direct_evidence_id": direct.evidence_id,
                "eligible_evidence_ids": [
                    episode.evidence_catalog[index].evidence_id for index in candidate_indices
                ],
                "excluded_misaligned_evidence_ids": excluded_misaligned_ids,
            },
            "runtime_terminal_only": not candidate_indices,
            "evidence_predictions": evidence_predictions,
            "mask_feature_contract": {
                "schema_version": MASK_FEATURES_SCHEMA,
                "feature_names": list(MASK_FEATURE_NAMES),
                "target_free": True,
                "scope": "candidate_local_grid",
            },
            "operation_update_threshold": None,
        },
    )
    receipt = _sha256_payload(
        {
            "prior": _geometry_sha256(prior_geometry),
            "candidates": [
                {
                    "id": item.evidence_id,
                    "prediction": _sha256_prediction(item.evidence_id, item.prediction),
                }
                for item in rows
            ],
            "eligible_evidence_ids": [
                episode.evidence_catalog[index].evidence_id for index in candidate_indices
            ],
            "excluded_misaligned_evidence_ids": excluded_misaligned_ids,
            "excluded_future_evidence_ids": list(excluded_future_evidence_ids),
        }
    )
    return sample, by_id, receipt


def build_runtime_catalog(
    episode: EpisodeRecord,
    prior_geometry: Any,
    predictor: Any,
    *,
    image_size: int,
    budget: float,
    causal_evidence_only: bool = False,
) -> RuntimeCatalog:
    """Freshly infer every catalog image conditioned on one carried vector prior."""

    anchor_timestamp = episode.anchor_timestamp or episode.evidence_catalog[-1].timestamp
    anchor_month = _month_index(anchor_timestamp)
    excluded_future_ids = tuple(
        item.evidence_id
        for item in episode.evidence_catalog
        if causal_evidence_only and _month_index(item.timestamp) > anchor_month
    )
    runtime_items = [
        item
        for item in episode.evidence_catalog
        if item.evidence_id not in excluded_future_ids
    ]
    if not runtime_items:
        raise ValueError("causal runtime catalog has no observable evidence")
    runtime_episode = episode.model_copy(update={"evidence_catalog": runtime_items})
    candidates: list[RuntimeCandidate] = []
    for item in runtime_items:
        image, prior, _, valid, raster_shape = _runtime_candidate_inputs(
            runtime_episode,
            item,
            prior_geometry,
            image_size=image_size,
            image_channels=predictor.model.config.image_channels,
            temporal_pair_input=bool(
                getattr(predictor.model.config, "temporal_pair_input", False)
            ),
            target_geometry=None,
        )
        prediction = predictor.predict(image, prior)
        candidates.append(
            RuntimeCandidate(
                evidence_id=item.evidence_id,
                image=image,
                prior=prior,
                valid=valid,
                transform=_transform(item, image_size),
                raster_shape=raster_shape,
                prediction=prediction,
                clear_fraction=float(np.mean(valid >= 0.5)),
            )
        )
    sample, by_id, receipt = build_runtime_selector_sample(
        runtime_episode,
        prior_geometry,
        candidates,
        budget=budget,
        excluded_future_evidence_ids=excluded_future_ids,
        causal_evidence_only=causal_evidence_only,
    )
    direct_id = str(sample.metadata["initial_evidence_id"])
    return RuntimeCatalog(
        sample=sample,
        candidates=by_id,
        direct_evidence_id=direct_id,
        excluded_future_evidence_ids=excluded_future_ids,
        prior_hash=_geometry_sha256(prior_geometry),
        candidate_receipt_hash=receipt,
        direct_receipt_hash=_sha256_prediction(direct_id, by_id[direct_id].prediction),
    )


def _runtime_candidate_inputs(
    episode: EpisodeRecord,
    item: Any,
    prior_geometry: Any,
    *,
    image_size: int,
    image_channels: int,
    temporal_pair_input: bool,
    target_geometry: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    from activemap.oracle.updater_counterfactual import _read_candidate

    return _read_candidate(
        item,
        prior_geometry=prior_geometry,
        target_geometry=target_geometry,
        image_size=image_size,
        image_channels=image_channels,
        temporal_pair_input=temporal_pair_input,
        road_width_source_pixels=(
            float(episode.metadata["road_width_source_pixels"])
            if episode.metadata.get("road_width_source_pixels") is not None
            else None
        ),
    )


class DirectCurrentHypothesisPolicy:
    """No selector/tool baseline that writes the fresh direct hypothesis only."""

    def act(self, observation: AgentObservation) -> AgentAction:
        operation = observation.belief.predicted_edit
        if operation == EditOperation.KEEP:
            return AgentAction(action=AgentActionType.REJECT)
        return AgentAction(action=AgentActionType.COMMIT, edit=operation)


class AlwaysRejectPolicy:
    """True STOP baseline: preserve the carried map regardless of direct belief."""

    def act(self, _observation: AgentObservation) -> AgentAction:
        return AgentAction(action=AgentActionType.REJECT)


def _load_tool_gate(path: Path, summary_path: Path) -> tuple[Any, float]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") == "uncertainty-tool-gate-summary-v1":
        from activemap.agent.active_catalog_tool_gate import BeliefUncertaintyGate

        gate: Any = BeliefUncertaintyGate()
    else:
        from joblib import load

        gate = load(path)
    selected = summary.get("selected")
    if not isinstance(selected, dict) or "threshold" not in selected:
        raise ValueError("frozen tool-gate summary lacks selected threshold")
    return gate, float(selected["threshold"])


@dataclass(frozen=True)
class LoadedController:
    selector: Any
    paired_tool_belief: PairedToolBeliefUpdater
    post_tool_adapter: PostToolActionAdapterPredictor
    tool_gate: Any
    tool_gate_threshold: float
    registered_tool_gate_threshold: float


def require_online_observable_selector(selector: Any) -> None:
    if getattr(selector, "data_contract", None) != ONLINE_OBSERVABLE_STATE_CONTRACT:
        raise ValueError(
            "true online controller requires a selector trained with "
            "online-observable-state-v1"
        )


def load_controller(
    assets: FrozenControllerAssets,
    *,
    device: str,
    diagnostic_stop_margin_override: float | None = None,
    diagnostic_tool_gate_threshold_override: float | None = None,
) -> LoadedController:
    gate, registered_gate_threshold = _load_tool_gate(
        assets.tool_gate, assets.tool_gate_summary
    )
    gate_threshold = (
        registered_gate_threshold
        if diagnostic_tool_gate_threshold_override is None
        else diagnostic_tool_gate_threshold_override
    )
    stop_margin = (
        assets.selector_stop_margin
        if diagnostic_stop_margin_override is None
        else diagnostic_stop_margin_override
    )
    try:
        selector = load_evidence_value_predictor(
            str(assets.selector),
            device=device,
            stop_margin_override=stop_margin,
        )
    except ValueError as exc:
        if not str(exc).startswith("unsupported Evidence Value checkpoint protocol"):
            raise
        selector = SelectorPredictor(
            assets.selector,
            device=device,
            stop_margin_override=stop_margin,
        )
    require_online_observable_selector(selector)
    return LoadedController(
        selector=selector,
        paired_tool_belief=PairedToolBeliefUpdater.from_checkpoint(
            assets.tool_belief, device="cpu"
        ),
        post_tool_adapter=PostToolActionAdapterPredictor(assets.post_tool_adapter, device=device),
        tool_gate=gate,
        tool_gate_threshold=gate_threshold,
        registered_tool_gate_threshold=registered_gate_threshold,
    )


def _tool_registry(output_root: Path) -> Any:
    from activemap.geo_tools.raster import ImageQualityTool, TemporalChangeTool
    from activemap.geo_tools.registry import GeoToolRegistry

    registry = GeoToolRegistry()
    registry.register(ImageQualityTool())
    registry.register(TemporalChangeTool(output_root / "temporal_change"))
    return registry


def _policy_environment(
    catalog: RuntimeCatalog,
    episode: EpisodeRecord,
    controller: LoadedController,
    *,
    policy_name: str,
    max_candidates: int,
    tool_artifact_root: Path,
) -> tuple[MapMaintenanceEnv, Any]:
    base_policy_name = _base_policy_name(policy_name)
    if base_policy_name == "always_stop":
        environment = make_environment(
            catalog.sample,
            max_candidates,
            score_fn=controller.selector.action_scores,
        )
        return environment, AlwaysRejectPolicy()
    if base_policy_name == "direct_current_hypothesis":
        environment = make_environment(
            catalog.sample,
            max_candidates,
            score_fn=controller.selector.action_scores,
        )
        return environment, DirectCurrentHypothesisPolicy()
    if base_policy_name not in {"active_forced", "active_selective"}:
        raise ValueError(f"unknown online policy: {policy_name}")
    environment = make_environment(
        catalog.sample,
        max_candidates,
        score_fn=controller.selector.action_scores,
        episode=episode,
        tool_registry=_tool_registry(tool_artifact_root),
        tool_belief_updater=SequentialPairedToolBeliefUpdater(controller.paired_tool_belief),
        tool_out_size=256,
        asset_root_maps=[],
    )
    base: Any = PostToolAdaptedPolicy(GreedyAgentPolicy(), controller.post_tool_adapter)
    mode = "forced" if base_policy_name == "active_forced" else "selective"
    policy = SelectiveToolPolicy(
        base,
        mode=mode,
        gate=controller.tool_gate if mode == "selective" else None,
        threshold=controller.tool_gate_threshold,
    )
    return environment, policy


def _terminal_operation(trajectory: Any) -> EditOperation:
    terminal = trajectory.transitions[-1].action
    if terminal.action == AgentActionType.REJECT:
        return EditOperation.KEEP
    if terminal.action != AgentActionType.COMMIT or terminal.edit is None:
        raise ValueError("controller trajectory did not terminate with COMMIT or REJECT")
    return terminal.edit


def _writeback_from_trajectory(
    catalog: RuntimeCatalog,
    episode: EpisodeRecord,
    environment: MapMaintenanceEnv,
    trajectory: Any,
    prior_geometry: Any,
    *,
    threshold: float,
    delta_margin: float,
    min_delta_component_pixels: int,
    safe_gate: SafeCommitGate | None,
    safe_memory: RecurrentSafeCommitMemory | None,
    step: int,
    prior_hash: str,
    retry_confidence_margin: float,
) -> tuple[Any, dict[str, Any]]:
    selected_ids = list(environment.selected)
    if not selected_ids:
        raise RuntimeError("runtime environment must retain a direct evidence item")
    direct = catalog.candidates[catalog.direct_evidence_id]
    selected = [catalog.candidates[evidence_id] for evidence_id in selected_ids]
    if any(item.prior.shape != direct.prior.shape for item in selected):
        raise ValueError("selected runtime evidence does not share a raster grid")
    if any(item.transform != direct.transform for item in selected):
        raise ValueError("selected runtime evidence has inconsistent map transforms")
    operation = _terminal_operation(trajectory)
    direct_item = next(
        item for item in episode.evidence_catalog if item.evidence_id == catalog.direct_evidence_id
    )
    _, _, target, valid, _ = _runtime_candidate_inputs(
        episode,
        direct_item,
        prior_geometry,
        image_size=int(direct.prior.shape[0]),
        image_channels=int(direct.image.shape[0]),
        target_geometry=_episode_geometry(episode.target_geometry),
    )
    metrics = evaluate_typed_writeback(
        [
            EvidenceMaskPrediction(
                evidence_id=item.evidence_id,
                target_probability=np.asarray(item.prediction["mask_probability"]),
                confidence=float(item.prediction["confidence"]),
            )
            for item in selected
        ],
        operation=operation,
        prior=direct.prior,
        target=target,
        valid=valid,
        transform=direct.transform,
        threshold=threshold,
        delta_margin=delta_margin,
        min_delta_component_pixels=min_delta_component_pixels,
        return_artifacts=False,
    )
    accepted, safe_decision = decide_safe_commit(
        metrics,
        safe_gate,
        safe_memory,
        operation=operation,
        step=step,
        prior_hash=prior_hash,
        selected_evidence_ids=selected_ids,
        direct_evidence_id=catalog.direct_evidence_id,
        tool_calls=len(environment.tool_history),
        retry_confidence_margin=retry_confidence_margin,
    )
    effective = (
        EditOperation(str(metrics["effective_operation"])) if accepted else EditOperation.KEEP
    )
    next_state = (
        _apply_delta(
            prior_geometry,
            metrics["predicted_add_geometry"],
            metrics["predicted_remove_geometry"],
        )
        if accepted
        else prior_geometry
    )
    final_iou = float(metrics["raster_iou"] if accepted else metrics["prior_raster_iou"])
    false_edit, missed_edit, wrong_edit = _operation_errors(episode.gt_edit.op, effective)
    return next_state, {
        "controller_terminal_operation": operation.value,
        "effective_operation": effective.value,
        "writeback_accepted": bool(accepted),
        "safe_commit_rejected": bool(metrics["writeback_changed"]) and not accepted,
        "commit_accepted": bool(metrics["writeback_changed"]) and accepted,
        "selected_evidence_ids": selected_ids,
        "writeback_reference_evidence_id": catalog.direct_evidence_id,
        **safe_decision,
        "false_edit": bool(false_edit),
        "missed_edit": bool(missed_edit),
        "wrong_edit": bool(wrong_edit),
        "recovered_from_prior_error": bool(
            float(metrics["prior_raster_iou"]) < 1.0 - 1e-6
            and final_iou > float(metrics["prior_raster_iou"]) + 1e-6
        ),
        "final_raster_iou": final_iou,
        "executed_raster_iou_gain": final_iou - float(metrics["prior_raster_iou"]),
        "writeback_receipt_sha256": _sha256_payload(
            {
                key: value
                for key, value in metrics.items()
                if key not in {"predicted_add_geometry", "predicted_remove_geometry"}
            }
        ),
        **{
            key: value
            for key, value in metrics.items()
            if key not in {"predicted_add_geometry", "predicted_remove_geometry"}
        },
    }


def _runtime_oracle_outcome(
    catalog: RuntimeCatalog,
    episode: EpisodeRecord,
    selected_ids: list[str],
    operation: EditOperation,
    *,
    target: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    delta_margin: float,
    min_delta_component_pixels: int,
    safe_gate: SafeCommitGate | None,
    utility_profile: str,
) -> dict[str, Any]:
    """Evaluate one target-labeled counterfactual after target-free inference."""

    if utility_profile not in UTILITY_PROFILES:
        raise ValueError(f"unknown runtime oracle utility profile: {utility_profile}")
    direct = catalog.candidates[catalog.direct_evidence_id]
    selected = [catalog.candidates[evidence_id] for evidence_id in selected_ids]
    metrics = evaluate_typed_writeback(
        [
            EvidenceMaskPrediction(
                evidence_id=item.evidence_id,
                target_probability=np.asarray(item.prediction["mask_probability"]),
                confidence=float(item.prediction["confidence"]),
            )
            for item in selected
        ],
        operation=operation,
        prior=direct.prior,
        target=target,
        valid=valid,
        transform=direct.transform,
        threshold=threshold,
        delta_margin=delta_margin,
        min_delta_component_pixels=min_delta_component_pixels,
        return_artifacts=False,
    )
    accepted = safe_gate is None or safe_gate.accepts(metrics)
    effective = (
        EditOperation(str(metrics["effective_operation"]))
        if accepted
        else EditOperation.KEEP
    )
    final_iou = float(metrics["raster_iou"] if accepted else metrics["prior_raster_iou"])
    prior_iou = float(metrics["prior_raster_iou"])
    false_edit, missed_edit, wrong_edit = _operation_errors(
        episode.gt_edit.op, effective
    )
    profile = UTILITY_PROFILES[utility_profile]
    quality_gain = final_iou - prior_iou
    terminal_score = (
        quality_gain
        - profile.false_edit * float(false_edit)
        - profile.missed_edit * float(missed_edit)
        - profile.wrong_edit * float(wrong_edit)
    )
    return {
        "predicted_operation": operation.value,
        "raw_effective_operation": str(metrics["raw_effective_operation"]),
        "raw_writeback_changed": bool(metrics["raw_writeback_changed"]),
        "raw_added_pixels": int(metrics["raw_added_pixels"]),
        "raw_removed_pixels": int(metrics["raw_removed_pixels"]),
        "effective_operation": effective.value,
        "writeback_accepted": bool(accepted),
        "writeback_changed": bool(metrics["writeback_changed"]),
        "regularization_removed_all_delta": bool(
            metrics["regularization_removed_all_delta"]
        ),
        "raw_add_component_count": int(metrics["raw_add_component_count"]),
        "retained_add_component_count": int(metrics["retained_add_component_count"]),
        "raw_remove_component_count": int(metrics["raw_remove_component_count"]),
        "retained_remove_component_count": int(
            metrics["retained_remove_component_count"]
        ),
        "prior_raster_iou": prior_iou,
        "final_raster_iou": final_iou,
        "quality_gain": quality_gain,
        "false_edit": bool(false_edit),
        "missed_edit": bool(missed_edit),
        "wrong_edit": bool(wrong_edit),
        "terminal_score_before_cost": float(terminal_score),
    }


def build_carried_state_oracle_sample(
    catalog: RuntimeCatalog,
    episode: EpisodeRecord,
    prior_geometry: Any,
    *,
    output_split: str,
    policy_name: str,
    chain_index: int,
    step: int,
    budget: float,
    max_candidates: int,
    threshold: float,
    delta_margin: float,
    min_delta_component_pixels: int,
    safe_gate: SafeCommitGate | None,
    utility_profile: str = "balanced",
) -> SelectorSample | None:
    """Label exact carried-state features with one-step executable utility."""

    if episode.split == "test" or output_split not in {"train", "val"}:
        raise ValueError("runtime oracle accepts only train/validation provenance")
    sample = catalog.sample
    if not sample.evidence_ids:
        return None
    direct_item = next(
        item
        for item in episode.evidence_catalog
        if item.evidence_id == catalog.direct_evidence_id
    )
    direct = catalog.candidates[catalog.direct_evidence_id]
    _, _, target, valid, _ = _runtime_candidate_inputs(
        episode,
        direct_item,
        prior_geometry,
        image_size=int(direct.prior.shape[0]),
        image_channels=int(direct.image.shape[0]),
        target_geometry=_episode_geometry(episode.target_geometry),
    )

    def fresh_environment() -> MapMaintenanceEnv:
        return make_environment(
            sample,
            max_candidates,
            score_fn=lambda current: np.zeros(
                len(current.evidence_ids) + 1, dtype=np.float32
            ),
        )

    direct_environment = fresh_environment()
    direct_observation = direct_environment.reset()
    direct_operation = direct_observation.belief.predicted_edit
    direct_outcome = _runtime_oracle_outcome(
        catalog,
        episode,
        [catalog.direct_evidence_id],
        direct_operation,
        target=target,
        valid=valid,
        threshold=threshold,
        delta_margin=delta_margin,
        min_delta_component_pixels=min_delta_component_pixels,
        safe_gate=safe_gate,
        utility_profile=utility_profile,
    )
    direct_score = float(direct_outcome["terminal_score_before_cost"])
    profile = UTILITY_PROFILES[utility_profile]
    utilities: list[float] = []
    outcomes: dict[str, dict[str, Any]] = {}
    for evidence_id, evidence_cost in zip(
        sample.evidence_ids, sample.evidence_costs, strict=True
    ):
        environment = fresh_environment()
        environment.reset()
        transition = environment.step(
            AgentAction(
                action=AgentActionType.ACQUIRE,
                evidence_id=public_evidence_id(evidence_id),
            )
        )
        if transition.next_observation is None:
            raise RuntimeError("forced runtime-oracle acquisition terminated early")
        operation = transition.next_observation.belief.predicted_edit
        outcome = _runtime_oracle_outcome(
            catalog,
            episode,
            list(environment.selected),
            operation,
            target=target,
            valid=valid,
            threshold=threshold,
            delta_margin=delta_margin,
            min_delta_component_pixels=min_delta_component_pixels,
            safe_gate=safe_gate,
            utility_profile=utility_profile,
        )
        marginal_before_cost = (
            float(outcome["terminal_score_before_cost"]) - direct_score
        )
        cost_penalty = profile.cost * min(float(evidence_cost) / budget, 1.0)
        utility = marginal_before_cost - cost_penalty
        outcomes[evidence_id] = {
            **outcome,
            "quality_gain": (
                float(outcome["final_raster_iou"])
                - float(direct_outcome["final_raster_iou"])
            ),
            "terminal_score_before_cost": marginal_before_cost,
            "evidence_cost_penalty": float(cost_penalty),
        }
        utilities.append(float(utility))

    metadata = dict(sample.metadata)
    metadata.update(
        {
            "gt_edit": episode.gt_edit.op.value,
            "utility_mode": "executable",
            "utility_profile": utility_profile,
            "budget": float(budget),
            "cost_weight": 0.0,
            "false_edit_penalty_weight": 0.0,
            "writeback_threshold": float(threshold),
            "writeback_delta_margin": float(delta_margin),
            "executable_outcomes": outcomes,
            "runtime_oracle_contract": {
                "version": RUNTIME_ORACLE_CONTRACT,
                "policy": policy_name,
                "target_used_for_labels_only": True,
                "downstream": "belief_terminal_plus_safe_commit_without_tools",
                "chain_index": int(chain_index),
                "step": int(step),
                "direct_outcome": direct_outcome,
            },
            "test_assets_read": False,
        }
    )
    return sample.model_copy(
        update={
            "sample_id": (
                f"{episode.episode_id}__carried__{policy_name}"
                f"__c{chain_index:05d}__s{step:04d}"
            ),
            "split": output_split,
            "oracle_utilities": utilities,
            "stop_utility": 0.0,
            "false_edit_penalty_weight": 0.0,
            "metadata": metadata,
        }
    )


def evaluate_full_controller_chains(
    predictor: Any,
    controller: LoadedController,
    chains: list[list[EpisodeRecord]],
    *,
    policies: tuple[str, ...],
    image_size: int,
    budget: float,
    max_candidates: int,
    max_acquisitions: int,
    max_tool_calls: int,
    threshold: float,
    delta_margin: float,
    min_delta_component_pixels: int,
    tool_artifact_root: Path,
    safe_gate: SafeCommitGate | None,
    causal_evidence_only: bool = False,
    recurrent_safe_commit: bool = False,
    retry_confidence_margin: float = 0.05,
    runtime_selector_inputs: list[dict[str, Any]] | None = None,
    runtime_oracle_states: list[SelectorSample] | None = None,
    runtime_oracle_policy: str = "active_selective_safe",
    runtime_oracle_output_split: str = "train",
    runtime_oracle_utility_profile: str = "balanced",
) -> list[dict[str, Any]]:
    """Roll every policy over an independent carried state for each chain."""

    records: list[dict[str, Any]] = []
    for chain_index, chain in enumerate(chains):
        states = {name: _episode_geometry(chain[0].prior_geometry) for name in policies}
        safe_memories = {
            name: RecurrentSafeCommitMemory()
            for name in policies
            if recurrent_safe_commit and name in SAFE_COMMIT_POLICIES
        }
        for step, episode in enumerate(chain):
            for policy_name in policies:
                prior_geometry = states[policy_name]
                catalog = build_runtime_catalog(
                    episode,
                    prior_geometry,
                    predictor,
                    image_size=image_size,
                    budget=budget,
                    causal_evidence_only=causal_evidence_only,
                )
                environment, policy = _policy_environment(
                    catalog,
                    episode,
                    controller,
                    policy_name=policy_name,
                    max_candidates=max_candidates,
                    tool_artifact_root=(
                        tool_artifact_root
                        / policy_name
                        / f"chain_{chain_index:05d}"
                        / f"step_{step:04d}"
                    ),
                )
                if (
                    runtime_oracle_states is not None
                    and policy_name == runtime_oracle_policy
                ):
                    oracle_sample = build_carried_state_oracle_sample(
                        catalog,
                        episode,
                        prior_geometry,
                        output_split=runtime_oracle_output_split,
                        policy_name=policy_name,
                        chain_index=chain_index,
                        step=step,
                        budget=budget,
                        max_candidates=max_candidates,
                        threshold=threshold,
                        delta_margin=delta_margin,
                        min_delta_component_pixels=min_delta_component_pixels,
                        safe_gate=(
                            safe_gate if policy_name in SAFE_COMMIT_POLICIES else None
                        ),
                        utility_profile=runtime_oracle_utility_profile,
                    )
                    if oracle_sample is not None:
                        runtime_oracle_states.append(oracle_sample)
                if runtime_selector_inputs is not None:
                    runtime_selector_inputs.append(
                        snapshot_runtime_selector_input(
                            environment,
                            catalog,
                            policy_name=policy_name,
                            chain_index=chain_index,
                            step=step,
                            episode=episode,
                        )
                    )
                trajectory = rollout_agent_policy(
                    environment,
                    policy,
                    max_acquisitions=max_acquisitions,
                    max_tool_calls=(
                        0
                        if _base_policy_name(policy_name)
                        in {"always_stop", "direct_current_hypothesis"}
                        else max_tool_calls
                    ),
                    max_steps=3 * max_acquisitions + 1,
                )
                branch_gate = safe_gate if policy_name in SAFE_COMMIT_POLICIES else None
                next_state, writeback = _writeback_from_trajectory(
                    catalog,
                    episode,
                    environment,
                    trajectory,
                    prior_geometry,
                    threshold=threshold,
                    delta_margin=delta_margin,
                    min_delta_component_pixels=min_delta_component_pixels,
                    safe_gate=branch_gate,
                    safe_memory=safe_memories.get(policy_name),
                    step=step,
                    prior_hash=catalog.prior_hash,
                    retry_confidence_margin=retry_confidence_margin,
                )
                states[policy_name] = next_state
                terminal = trajectory.transitions[-1]
                initial_observation = trajectory.transitions[0].observation
                initial_best_score = (
                    max(
                        candidate.selector_score
                        for candidate in initial_observation.candidates
                    )
                    if initial_observation.candidates
                    else None
                )
                initial_stop_score = initial_observation.terminal_score
                records.append(
                    {
                        "policy": policy_name,
                        "chain_id": f"chain-{chain_index:05d}",
                        "step": step,
                        "task_id": episode.episode_id,
                        "aoi_id": str(episode.aoi_id),
                        "object_id": str(episode.hypothesis.object_id),
                        "timestamp": str(episode.anchor_timestamp),
                        "split": episode.split,
                        "target_edit": episode.gt_edit.op.value,
                        "test_assets_read": False,
                        "prior_source": "carried_executed_vector_state",
                        "prior_input_sha256": catalog.prior_hash,
                        "candidate_frontend_prior_sha256": catalog.prior_hash,
                        "direct_hypothesis_prior_sha256": catalog.prior_hash,
                        "executed_prior_sha256": _geometry_sha256(next_state),
                        "canonical_prior_sha256": _geometry_sha256(
                            _episode_geometry(episode.prior_geometry)
                        ),
                        "candidate_frontend_receipt_sha256": (catalog.candidate_receipt_hash),
                        "direct_hypothesis_receipt_sha256": catalog.direct_receipt_hash,
                        "target_labels_used_as_policy_input": False,
                        "runtime_selector_target_free": bool(
                            catalog.sample.metadata["runtime_target_free"]
                        ),
                        "runtime_eligible_candidate_count": len(catalog.sample.evidence_ids),
                        "runtime_excluded_future_evidence_count": len(
                            catalog.excluded_future_evidence_ids
                        ),
                        "runtime_excluded_future_evidence_ids": list(
                            catalog.excluded_future_evidence_ids
                        ),
                        "runtime_excluded_misaligned_candidate_count": len(
                            catalog.sample.metadata["runtime_grid_alignment"][
                                "excluded_misaligned_evidence_ids"
                            ]
                        ),
                        "runtime_initial_best_candidate_score": initial_best_score,
                        "runtime_initial_stop_score": initial_stop_score,
                        "runtime_initial_candidate_minus_stop": (
                            None
                            if initial_best_score is None or initial_stop_score is None
                            else float(initial_best_score - initial_stop_score)
                        ),
                        "acquisitions": int(trajectory.metadata["acquisition_count"]),
                        "tool_calls": int(trajectory.metadata["tool_call_count"]),
                        "uses_safe_commit": policy_name in SAFE_COMMIT_POLICIES,
                        "spent_cost": float(terminal.observation.spent_cost),
                        "tool_cost": float(sum(item.cost for item in environment.tool_history)),
                        "tool_history": [
                            item.model_dump(mode="json") for item in environment.tool_history
                        ],
                        "events": list(getattr(policy, "events", [])),
                        **writeback,
                    }
                )
    return records


def snapshot_runtime_selector_input(
    environment: MapMaintenanceEnv,
    catalog: RuntimeCatalog,
    *,
    policy_name: str,
    chain_index: int,
    step: int,
    episode: EpisodeRecord,
) -> dict[str, Any]:
    """Capture the exact post-environment sample presented to the selector."""

    sample = environment.current_sample()
    selector_invoked = bool(sample.evidence_ids)
    scores = (
        np.asarray(environment.score_fn(sample), dtype=np.float64)
        if selector_invoked
        else np.asarray([], dtype=np.float64)
    )
    best_candidate_score = (
        float(np.max(scores[:-1])) if selector_invoked and len(scores) > 1 else None
    )
    stop_score = float(scores[-1]) if selector_invoked and len(scores) > 1 else None
    excluded = catalog.sample.metadata["runtime_grid_alignment"][
        "excluded_misaligned_evidence_ids"
    ]
    return {
        "schema_version": "activemap-runtime-selector-input-v1",
        "policy": policy_name,
        "chain_id": f"chain-{chain_index:05d}",
        "step": step,
        "task_id": episode.episode_id,
        "aoi_id": str(episode.aoi_id),
        "split": episode.split,
        "test_assets_read": False,
        "selector_invoked": selector_invoked,
        "sample_id": sample.sample_id,
        "hypothesis_features": sample.hypothesis_features,
        "state_features": sample.state_features,
        "evidence_ids": sample.evidence_ids,
        "evidence_features": sample.evidence_features,
        "evidence_costs": sample.evidence_costs,
        "false_edit_risks": sample.false_edit_risks,
        "selected_evidence_ids": list(environment.selected),
        "candidate_count": len(sample.evidence_ids),
        "total_evidence_count": environment.total_evidence_count,
        "source_catalog_candidate_count": len(catalog.candidates),
        "runtime_grid_eligible_count": len(catalog.sample.evidence_ids),
        "runtime_grid_excluded_count": len(excluded),
        "initial_budget": environment.initial_budget,
        "remaining_budget": environment.remaining_budget,
        "selector_scores": scores.tolist(),
        "best_candidate_score": best_candidate_score,
        "stop_score": stop_score,
        "candidate_minus_stop": (
            None
            if best_candidate_score is None or stop_score is None
            else best_candidate_score - stop_score
        ),
    }


def _mean(rows: list[dict[str, Any]], name: str) -> float:
    return float(np.mean([float(row[name]) for row in rows])) if rows else 0.0


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_policy.setdefault(str(row["policy"]), []).append(row)
    result = {}
    for policy, rows in by_policy.items():
        final_by_chain: dict[str, dict[str, Any]] = {}
        for row in rows:
            final_by_chain[str(row["chain_id"])] = row
        final = list(final_by_chain.values())
        result[policy] = {
            "transition_count": len(rows),
            "chain_count": len(final),
            "mean_step_raster_iou": _mean(rows, "final_raster_iou"),
            "mean_final_chain_raster_iou": _mean(final, "final_raster_iou"),
            "false_edit_rate": _mean(rows, "false_edit"),
            "missed_edit_rate": _mean(rows, "missed_edit"),
            "wrong_edit_rate": _mean(rows, "wrong_edit"),
            "commit_rate": _mean(rows, "commit_accepted"),
            "safe_rejection_rate": _mean(rows, "safe_commit_rejected"),
            "safe_immediate_retry_rate": _mean(rows, "safe_commit_immediate_retry"),
            "safe_retry_block_rate": _mean(rows, "safe_commit_retry_blocked"),
            "recovery_rate_after_prior_error": _mean(rows, "recovered_from_prior_error"),
            "mean_acquisitions": _mean(rows, "acquisitions"),
            "mean_tool_calls": _mean(rows, "tool_calls"),
            "mean_spent_cost": _mean(rows, "spent_cost"),
            "mean_tool_cost": _mean(rows, "tool_cost"),
            "noncanonical_prior_rate": float(
                np.mean(
                    [row["prior_input_sha256"] != row["canonical_prior_sha256"] for row in rows]
                )
            ),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path)
    parser.add_argument("registry", type=Path)
    parser.add_argument("controller_seed", type=int)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--budget", type=float, default=3.0)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--max-acquisitions", type=int, default=2)
    parser.add_argument("--max-tool-calls", type=int, default=4)
    parser.add_argument("--minimum-chain-length", type=int, default=2)
    parser.add_argument("--continuity-tolerance", type=float, default=1e-6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--delta-margin", type=float, default=0.15)
    parser.add_argument("--min-delta-component-pixels", type=int, default=0)
    parser.add_argument("--safe-confidence-threshold", type=float)
    parser.add_argument("--safe-replay-iou-threshold", type=float, default=0.99)
    parser.add_argument("--causal-evidence-only", action="store_true")
    parser.add_argument("--recurrent-safe-commit", action="store_true")
    parser.add_argument("--retry-confidence-margin", type=float, default=0.05)
    parser.add_argument(
        "--diagnostic-stop-margin-override",
        type=float,
        help="Diagnostic override; outputs are marked as non-promotion evidence.",
    )
    parser.add_argument(
        "--diagnostic-tool-gate-threshold-override",
        type=float,
        help="Diagnostic override; outputs are marked as non-promotion evidence.",
    )
    parser.add_argument(
        "--dump-runtime-selector-inputs",
        action="store_true",
        help="Write exact post-environment selector inputs for validation diagnostics.",
    )
    parser.add_argument(
        "--dump-runtime-oracle-states",
        action="store_true",
        help="Label carried states after rollout-time target-free inference.",
    )
    parser.add_argument(
        "--runtime-oracle-policy",
        choices=VALID_POLICIES,
        default="active_selective_safe",
    )
    parser.add_argument(
        "--runtime-oracle-output-split",
        choices=("train", "val"),
    )
    parser.add_argument(
        "--runtime-oracle-utility-profile",
        choices=tuple(UTILITY_PROFILES),
        default="balanced",
    )
    parser.add_argument(
        "--policy", action="append", choices=VALID_POLICIES
    )
    parser.add_argument("--asset-root-map", action="append", default=[])
    parser.add_argument(
        "--chain-index",
        type=int,
        help="Evaluate one deterministic chronological-chain index after sorting.",
    )
    parser.add_argument("--max-chains", type=int)
    parser.add_argument("--chain-start-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    if args.episodes == args.registry or not args.episodes.is_file() or not args.registry.is_file():
        raise FileNotFoundError("episodes and registry must be existing distinct files")
    if args.max_candidates <= 0 or args.max_acquisitions < 0 or args.max_tool_calls < 0:
        raise ValueError("invalid controller limits")
    if not 0.0 <= args.retry_confidence_margin <= 1.0:
        raise ValueError("retry confidence margin must be between zero and one")
    if args.chain_start_index < 0:
        raise ValueError("chain start index must be non-negative")
    policies = tuple(args.policy or DEFAULT_POLICIES)
    if len(set(policies)) != len(policies):
        raise ValueError("duplicate policy requested")
    if len(policies) < 2:
        raise ValueError("the true sequential audit requires at least two policies")
    if args.dump_runtime_oracle_states and args.runtime_oracle_policy not in policies:
        raise ValueError("runtime oracle policy must be included in --policy")
    requires_safe_commit = any(policy in SAFE_COMMIT_POLICIES for policy in policies)
    if requires_safe_commit and args.safe_confidence_threshold is None:
        raise ValueError("Safe Commit policies require --safe-confidence-threshold")
    mappings = _parse_asset_root_maps(args.asset_root_map)
    episodes = _remap_episode_assets(
        [episode for episode in load_episodes(args.episodes) if episode.split == args.split],
        mappings,
    )
    chains = build_contiguous_chains(
        episodes,
        minimum_length=args.minimum_chain_length,
        continuity_tolerance=args.continuity_tolerance,
    )
    if args.chain_index is not None:
        if args.chain_start_index:
            raise ValueError("--chain-index and --chain-start-index are mutually exclusive")
        if not 0 <= args.chain_index < len(chains):
            raise ValueError(
                f"--chain-index must be in [0, {len(chains) - 1}], got {args.chain_index}"
            )
        chains = [chains[args.chain_index]]
    else:
        stop = (
            None
            if args.max_chains is None
            else args.chain_start_index + args.max_chains
        )
        chains = chains[args.chain_start_index:stop]
    if not chains:
        raise ValueError(f"no chronological {args.split} chains")
    assets = load_frozen_controller_assets(
        args.registry,
        args.controller_seed,
        storage_root=args.storage_root,
        project_root=args.project_root,
    )
    require_registry_writeback_protocol(
        threshold=args.threshold,
        delta_margin=args.delta_margin,
        assets=assets,
    )
    predictor = UpdaterPredictor(assets.updater, device=args.device)
    controller = load_controller(
        assets,
        device=args.device,
        diagnostic_stop_margin_override=args.diagnostic_stop_margin_override,
        diagnostic_tool_gate_threshold_override=(
            args.diagnostic_tool_gate_threshold_override
        ),
    )
    gate = (
        SafeCommitGate(
            confidence_threshold=args.safe_confidence_threshold,
            replay_iou_threshold=args.safe_replay_iou_threshold,
        )
        if any(policy in SAFE_COMMIT_POLICIES for policy in policies)
        else None
    )
    runtime_selector_inputs: list[dict[str, Any]] | None = (
        [] if args.dump_runtime_selector_inputs else None
    )
    runtime_oracle_states: list[SelectorSample] | None = (
        [] if args.dump_runtime_oracle_states else None
    )
    runtime_oracle_output_split = args.runtime_oracle_output_split or args.split
    records = evaluate_full_controller_chains(
        predictor,
        controller,
        chains,
        policies=policies,
        image_size=args.image_size,
        budget=args.budget,
        max_candidates=args.max_candidates,
        max_acquisitions=args.max_acquisitions,
        max_tool_calls=args.max_tool_calls,
        threshold=args.threshold,
        delta_margin=args.delta_margin,
        min_delta_component_pixels=args.min_delta_component_pixels,
        tool_artifact_root=args.output_dir / "tool_artifacts",
        safe_gate=gate,
        causal_evidence_only=args.causal_evidence_only,
        recurrent_safe_commit=args.recurrent_safe_commit,
        retry_confidence_margin=args.retry_confidence_margin,
        runtime_selector_inputs=runtime_selector_inputs,
        runtime_oracle_states=runtime_oracle_states,
        runtime_oracle_policy=args.runtime_oracle_policy,
        runtime_oracle_output_split=runtime_oracle_output_split,
        runtime_oracle_utility_profile=args.runtime_oracle_utility_profile,
    )
    trace = args.output_dir / "online_full_controller_traces.jsonl"
    trace.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in records),
        encoding="utf-8",
    )
    audit_path = args.output_dir / "true_sequential_rollout_audit.json"
    audit_path.write_text(
        json.dumps(
            audit_true_sequential_rollout(trace, expected_split=args.split), indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    runtime_selector_input_path = None
    if runtime_selector_inputs is not None:
        runtime_selector_input_path = args.output_dir / "runtime_selector_inputs.jsonl"
        runtime_selector_input_path.write_text(
            "".join(
                json.dumps(row, separators=(",", ":")) + "\n"
                for row in runtime_selector_inputs
            ),
            encoding="utf-8",
        )
    runtime_oracle_path = None
    runtime_oracle_summary = None
    if runtime_oracle_states is not None:
        runtime_oracle_path = args.output_dir / "runtime_oracle_states.jsonl"
        runtime_oracle_path.write_text(
            "".join(sample.model_dump_json() + "\n" for sample in runtime_oracle_states),
            encoding="utf-8",
        )
        candidate_count = sum(len(sample.evidence_ids) for sample in runtime_oracle_states)
        positive_candidate_count = sum(
            sum(float(value) > float(sample.stop_utility) for value in sample.oracle_utilities)
            for sample in runtime_oracle_states
        )
        runtime_oracle_summary = {
            "schema_version": RUNTIME_ORACLE_CONTRACT,
            "state_count": len(runtime_oracle_states),
            "candidate_count": candidate_count,
            "positive_candidate_count": positive_candidate_count,
            "positive_candidate_rate": (
                positive_candidate_count / candidate_count if candidate_count else 0.0
            ),
            "target_acquire_state_count": sum(
                max(sample.oracle_utilities, default=float("-inf"))
                > float(sample.stop_utility)
                for sample in runtime_oracle_states
            ),
            "target_operation_counts": dict(
                sorted(
                    Counter(
                        str(sample.metadata["gt_edit"])
                        for sample in runtime_oracle_states
                    ).items()
                )
            ),
            "policy": args.runtime_oracle_policy,
            "output_split": runtime_oracle_output_split,
            "utility_profile": args.runtime_oracle_utility_profile,
            "target_used_for_labels_only": True,
            "test_assets_read": False,
        }
        (args.output_dir / "runtime_oracle_summary.json").write_text(
            json.dumps(runtime_oracle_summary, indent=2) + "\n",
            encoding="utf-8",
        )
    summary = {
        "schema_version": "activemap-online-full-controller-v1",
        "split": args.split,
        "test_assets_read": False,
        "controller_seed": args.controller_seed,
        "policies": list(policies),
        "chain_count": len(chains),
        "transition_count_per_policy": sum(len(chain) for chain in chains),
        "metrics": summarize(records),
        "protocol": {
            "fresh_candidate_frontend_each_step": True,
            "causal_evidence_only": args.causal_evidence_only,
            "carried_vector_prior": True,
            "frozen_selector": True,
            "selective_or_forced_tools": True,
            "recurrent_tool_belief": True,
            "post_tool_terminal_adapter": True,
            "target_labels_used_as_policy_input": False,
            "runtime_selector_inputs_dumped": runtime_selector_inputs is not None,
            "runtime_oracle_states_dumped": runtime_oracle_states is not None,
            "runtime_oracle_contract": (
                None if runtime_oracle_summary is None else RUNTIME_ORACLE_CONTRACT
            ),
            "diagnostic_only": (
                args.split == "train"
                or args.diagnostic_stop_margin_override is not None
                or args.diagnostic_tool_gate_threshold_override is not None
            ),
            "selector_stop_margin": controller.selector.stop_margin,
            "registered_selector_stop_margin": assets.selector_stop_margin,
            "tool_gate_threshold": controller.tool_gate_threshold,
            "registered_tool_gate_threshold": (
                controller.registered_tool_gate_threshold
            ),
            "writeback_probability_threshold": args.threshold,
            "delta_margin": args.delta_margin,
            "min_delta_component_pixels": args.min_delta_component_pixels,
            "selected_chain_index": args.chain_index,
            "selected_chain_start_index": args.chain_start_index,
            "recurrent_safe_commit": args.recurrent_safe_commit,
            "retry_confidence_margin": args.retry_confidence_margin,
            "safe_commit": None
            if gate is None
            else {
                "confidence_threshold": gate.confidence_threshold,
                "replay_iou_threshold": gate.replay_iou_threshold,
                "require_topology": gate.require_topology,
            },
        },
        "assets": {
            "registry": {
                "path": str(args.registry.resolve()),
                "sha256": _sha256_path(args.registry),
            },
            "components": assets.hashes,
            "episodes": {
                "path": str(args.episodes.resolve()),
                "sha256": _sha256_path(args.episodes),
            },
            "trace": {"path": str(trace.resolve()), "sha256": _sha256_path(trace)},
            "audit": {"path": str(audit_path.resolve()), "sha256": _sha256_path(audit_path)},
            **(
                {
                    "runtime_selector_inputs": {
                        "path": str(runtime_selector_input_path.resolve()),
                        "sha256": _sha256_path(runtime_selector_input_path),
                    }
                }
                if runtime_selector_input_path is not None
                else {}
            ),
            **(
                {
                    "runtime_oracle_states": {
                        "path": str(runtime_oracle_path.resolve()),
                        "sha256": _sha256_path(runtime_oracle_path),
                        "summary": runtime_oracle_summary,
                    }
                }
                if runtime_oracle_path is not None
                else {}
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
