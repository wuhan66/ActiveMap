"""Native ActiveMap records for frozen ArgoTweak perception outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field, field_validator, model_validator

from activemap.data.structured_map import StructuredMapScene
from activemap.models import EditOperation, GeoJSONGeometry, StrictModel

CHANGED_OPERATIONS = (
    EditOperation.ADD,
    EditOperation.DELETE,
    EditOperation.RESHAPE,
)


def proposal_city_geometry(
    proposal: dict[str, Any], pose: dict[str, list[Any]]
) -> GeoJSONGeometry:
    """Convert an official frame-local map proposal into a city-frame polygon."""

    points = proposal.get("geometry")
    if not isinstance(points, list) or len(points) < 4:
        raise ValueError("ArgoTweak proposal lacks polygon-supporting geometry")
    if len(points) >= 30:
        left = points[10:20]
        right = points[20:30]
        ring = [*left, *reversed(right)]
    else:
        ring = list(points)
    rotation = pose.get("rotation")
    translation = pose.get("translation")
    if (
        not isinstance(rotation, list)
        or len(rotation) != 2
        or not isinstance(translation, list)
        or len(translation) < 2
    ):
        raise ValueError("invalid ArgoTweak city pose")
    city_ring = []
    for point in ring:
        x, y = float(point[0]), float(point[1])
        city_ring.append(
            [
                float(rotation[0][0]) * x
                + float(rotation[0][1]) * y
                + float(translation[0]),
                float(rotation[1][0]) * x
                + float(rotation[1][1]) * y
                + float(translation[1]),
            ]
        )
    if city_ring[0] != city_ring[-1]:
        city_ring.append(city_ring[0])
    return GeoJSONGeometry(type="Polygon", coordinates=[city_ring])


def proposal_operation_belief(proposals: list[dict[str, Any]]) -> list[float]:
    """Aggregate frozen detector proposals into a calibrated-shape operation belief.

    This is an adapter feature, not a learned calibration. Downstream training may
    replace it while retaining the same four-operation contract.
    """

    scores = {operation: 0.0 for operation in EditOperation}
    for proposal in proposals:
        operation = EditOperation(str(proposal["operation"]))
        confidence = proposal.get("confidence") or {}
        joint = confidence.get("joint")
        if joint is None:
            joint = float(confidence.get("object", 0.0)) * float(confidence.get("change", 0.0))
        scores[operation] += max(0.0, float(joint))
    if not proposals or sum(scores.values()) <= 0.0:
        scores[EditOperation.KEEP] = 1.0
    total = sum(scores.values())
    return [scores[operation] / total for operation in EditOperation]


def operation_set_utility(
    proposals: list[dict[str, Any]],
    target_counts: dict[str, int],
    *,
    evidence_cost: float,
    false_edit_weight: float,
    missed_edit_weight: float,
    cost_weight: float,
) -> float:
    """Score frame evidence using operation coverage, safety, and acquisition cost."""

    predicted = {
        EditOperation(str(row["operation"]))
        for row in proposals
        if str(row["operation"]) != EditOperation.KEEP.value
    }
    target = {
        operation
        for operation in CHANGED_OPERATIONS
        if int(target_counts.get(operation.value, 0)) > 0
    }
    true_positive = len(predicted & target)
    false_positive = len(predicted - target)
    false_negative = len(target - predicted)
    precision = true_positive / len(predicted) if predicted else float(not target)
    recall = true_positive / len(target) if target else float(not predicted)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(
        f1
        - false_edit_weight * false_positive
        - missed_edit_weight * false_negative
        - cost_weight * evidence_cost
    )


class ArgoTweakNativeEvidence(StrictModel):
    schema_version: Literal["activemap-argotweak-native-evidence-v1"] = (
        "activemap-argotweak-native-evidence-v1"
    )
    evidence_id: str
    timestamp: str
    camera_bundle_path: str
    cost: float = Field(gt=0.0)
    proposals: list[dict[str, Any]]
    operation_belief: list[float]
    target_operation_counts: dict[str, int]
    utility: float
    commit_ready_proposal_ids: list[str]
    blocked_proposals: dict[str, str]
    city_se3_egovehicle: dict[str, list[Any]]

    @field_validator("city_se3_egovehicle")
    @classmethod
    def pose_is_two_dimensional(cls, value: dict[str, list[Any]]) -> dict[str, list[Any]]:
        rotation = value.get("rotation")
        translation = value.get("translation")
        if (
            not isinstance(rotation, list)
            or len(rotation) != 2
            or any(not isinstance(row, list) or len(row) != 2 for row in rotation)
            or not isinstance(translation, list)
            or len(translation) < 2
        ):
            raise ValueError("city_se3_egovehicle requires 2x2 rotation and xy translation")
        return value

    @field_validator("operation_belief")
    @classmethod
    def belief_is_a_distribution(cls, value: list[float]) -> list[float]:
        if len(value) != len(EditOperation) or any(item < 0.0 for item in value):
            raise ValueError("operation belief must contain four non-negative values")
        if not math.isclose(sum(value), 1.0, abs_tol=1e-6):
            raise ValueError("operation belief must sum to one")
        return value


class ArgoTweakNativeEpisode(StrictModel):
    schema_version: Literal["activemap-argotweak-native-episode-v1"] = (
        "activemap-argotweak-native-episode-v1"
    )
    episode_id: str
    split: Literal["train", "val"]
    segment_id: str
    prior_map_path: str
    target_map_path: str
    evidence: list[ArgoTweakNativeEvidence]
    oracle_evidence_id: str
    oracle_utility: float
    frozen_perception: bool = True
    object_assignment_policy: Literal["fail_closed"] = "fail_closed"
    test_assets_read: Literal[False] = False

    @model_validator(mode="after")
    def oracle_references_available_evidence(self) -> ArgoTweakNativeEpisode:
        identifiers = {row.evidence_id for row in self.evidence}
        if not self.evidence or self.oracle_evidence_id not in identifiers:
            raise ValueError("oracle evidence must reference the episode catalog")
        return self


def _safe_commit_partition(
    proposals: list[dict[str, Any]], *, confidence_threshold: float
) -> tuple[list[str], dict[str, str]]:
    ready: list[str] = []
    blocked: dict[str, str] = {}
    for proposal in proposals:
        proposal_id = str(proposal["proposal_id"])
        operation = EditOperation(str(proposal["operation"]))
        confidence = float((proposal.get("confidence") or {}).get("joint", 0.0))
        if operation == EditOperation.KEEP:
            blocked[proposal_id] = "KEEP_IS_NOT_A_WRITE"
        elif confidence < confidence_threshold:
            blocked[proposal_id] = "BELOW_COMMIT_CONFIDENCE"
        elif operation in {EditOperation.DELETE, EditOperation.RESHAPE} and not proposal.get(
            "object_id"
        ):
            blocked[proposal_id] = "OBJECT_ASSIGNMENT_REQUIRED"
        else:
            ready.append(proposal_id)
    return ready, blocked


def build_argotweak_native_episodes(
    proposal_path: Path,
    scene_path: Path,
    output_path: Path,
    *,
    commit_confidence: float = 0.5,
    false_edit_weight: float = 0.5,
    missed_edit_weight: float = 0.5,
    cost_weight: float = 0.05,
) -> dict[str, Any]:
    """Join frozen detector proposals with seven-camera evidence catalogs."""

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite native episodes: {output_path}")
    if not 0.0 <= commit_confidence <= 1.0:
        raise ValueError("commit_confidence must be between zero and one")
    scenes = [
        StructuredMapScene.model_validate_json(line)
        for line in scene_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    proposal_rows = [
        json.loads(line)
        for line in proposal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    proposal_index = {
        (str(row["split"]), str(row["segment_id"]), str(row["timestamp"])): row
        for row in proposal_rows
    }
    episodes: list[ArgoTweakNativeEpisode] = []
    missing: list[str] = []
    blocked_reason_counts: dict[str, int] = {}
    for scene in scenes:
        if scene.split == "test":
            raise PermissionError("native ArgoTweak adapter does not read test scenes")
        split = cast(Literal["train", "val"], scene.split)
        evidence: list[ArgoTweakNativeEvidence] = []
        for observation in scene.observations:
            key = (scene.split, scene.aoi_id, observation.timestamp)
            row = proposal_index.get(key)
            if row is None:
                missing.append(":".join(key))
                continue
            target_counts = row.get("gt_operation_counts")
            if not isinstance(target_counts, dict):
                raise ValueError(
                    "proposal export lacks gt_operation_counts; regenerate with native exporter"
                )
            proposals = list(row.get("proposals") or [])
            utility = operation_set_utility(
                proposals,
                target_counts,
                evidence_cost=observation.cost,
                false_edit_weight=false_edit_weight,
                missed_edit_weight=missed_edit_weight,
                cost_weight=cost_weight,
            )
            ready, blocked = _safe_commit_partition(
                proposals, confidence_threshold=commit_confidence
            )
            for reason in blocked.values():
                blocked_reason_counts[reason] = blocked_reason_counts.get(reason, 0) + 1
            evidence.append(
                ArgoTweakNativeEvidence(
                    evidence_id=observation.observation_id,
                    timestamp=observation.timestamp,
                    camera_bundle_path=observation.path,
                    cost=observation.cost,
                    proposals=proposals,
                    operation_belief=proposal_operation_belief(proposals),
                    target_operation_counts={
                        operation.value: int(target_counts.get(operation.value, 0))
                        for operation in EditOperation
                    },
                    utility=utility,
                    commit_ready_proposal_ids=ready,
                    blocked_proposals=blocked,
                    city_se3_egovehicle=dict(row["city_se3_egovehicle"]),
                )
            )
        if not evidence:
            continue
        oracle = max(evidence, key=lambda item: (item.utility, item.evidence_id))
        episodes.append(
            ArgoTweakNativeEpisode(
                episode_id=f"argotweak-native:{scene.aoi_id}",
                split=split,
                segment_id=scene.aoi_id,
                prior_map_path=scene.prior_map_path,
                target_map_path=scene.target_map_path,
                evidence=evidence,
                oracle_evidence_id=oracle.evidence_id,
                oracle_utility=oracle.utility,
            )
        )
    if not episodes:
        raise ValueError("no aligned ArgoTweak native episodes were produced")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(row.model_dump_json() + "\n" for row in episodes), encoding="utf-8"
    )
    summary = {
        "schema_version": "activemap-argotweak-native-summary-v1",
        "episodes": len(episodes),
        "evidence_frames": sum(len(row.evidence) for row in episodes),
        "missing_alignment_count": len(missing),
        "missing_alignment_examples": missing[:20],
        "blocked_reason_counts": blocked_reason_counts,
        "commit_confidence": commit_confidence,
        "frozen_perception": True,
        "object_assignment_policy": "fail_closed",
        "test_assets_read": False,
    }
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
