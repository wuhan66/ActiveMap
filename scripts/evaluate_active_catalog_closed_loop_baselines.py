#!/usr/bin/env python3
"""Evaluate deterministic and oracle closed-loop baselines on identical val states."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from activemap.agent.active_catalog import (
    active_catalog_state_from_observation,
    observable_shortlist_scores,
    terminal_agent_action,
)
from activemap.agent.active_catalog_candidate_ranker import (
    CandidateUtilityRankerPredictor,
)
from activemap.agent.active_catalog_joint import (
    joint_transition_from_agent_transition,
)
from activemap.agent.active_catalog_tool_gate import (
    ForcedInitialSemanticToolPolicy,
    SelectiveInitialSemanticToolPolicy,
    SelectiveToolPolicy,
)
from activemap.agent.environment import MapMaintenanceEnv, rollout_agent_policy
from activemap.agent.evidence_value_head import (
    EvidenceValuePredictor,
    StructuredMapActionPredictor,
)
from activemap.agent.identifiers import public_task_id
from activemap.agent.post_tool_action_adapter import PostToolActionAdapterPredictor
from activemap.agent.records import AgentAction, AgentActionType, AgentObservation
from activemap.agent.tools import (
    CounterfactualBeliefUpdater,
    GreedyAgentPolicy,
    UncertaintyAwareAgentPolicy,
)
from activemap.inference import SelectorPredictor
from activemap.models import EditOperation, EpisodeRecord
from activemap.selector_records import SelectorSample
from activemap.updater_records import UpdaterSample, load_updater_samples
from scripts.evaluate_active_catalog_closed_loop import (
    grouped_bootstrap,
    load_episodes,
    load_initial_samples,
    metrics,
    trajectory_row,
)


def authorize_split(split: str, frozen_test: bool) -> bool:
    test_assets_read = split == "test"
    if test_assets_read:
        if not frozen_test:
            raise PermissionError("test evaluation requires --frozen-test")
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
    elif frozen_test:
        raise ValueError("--frozen-test is valid only for the test split")
    return test_assets_read


class AlwaysStopPolicy:
    def act(self, observation: AgentObservation) -> AgentAction:
        return terminal_agent_action(observation)


class CandidatePolicy:
    def __init__(self, score: Any) -> None:
        self.score = score

    def act(self, observation: AgentObservation) -> AgentAction:
        if not observation.candidates:
            return terminal_agent_action(observation)
        candidate = max(observation.candidates, key=self.score)
        return AgentAction(
            action=AgentActionType.ACQUIRE,
            evidence_id=candidate.evidence_id,
        )


class RandomCandidatePolicy:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def act(self, observation: AgentObservation) -> AgentAction:
        if not observation.candidates:
            return terminal_agent_action(observation)
        candidate = self.rng.choice(observation.candidates)
        return AgentAction(
            action=AgentActionType.ACQUIRE,
            evidence_id=candidate.evidence_id,
        )


class OraclePolicy:
    def __init__(self, environment: MapMaintenanceEnv) -> None:
        self.environment = environment

    def act(self, observation: AgentObservation) -> AgentAction:
        del observation
        return self.environment.oracle_action()


class StructuredActionPolicy:
    """Use the joint policy for acquisition and the terminal edit operation."""

    def __init__(
        self,
        predictor: StructuredMapActionPredictor,
        environment: MapMaintenanceEnv,
        post_tool_adapter: PostToolActionAdapterPredictor | None = None,
    ) -> None:
        self.predictor = predictor
        self.environment = environment
        self.post_tool_adapter = post_tool_adapter

    def act(self, observation: AgentObservation) -> AgentAction:
        if observation.candidates:
            best = max(observation.candidates, key=lambda item: item.selector_score)
            if (
                observation.terminal_score is None
                or best.selector_score > observation.terminal_score
            ):
                return AgentAction(
                    action=AgentActionType.ACQUIRE,
                    evidence_id=best.evidence_id,
                )
        # Tool calls can update belief without rebuilding a candidate shortlist.
        # Recompute the terminal head from the current deployable state instead
        # of relying on the score-function cache from an earlier observation.
        if observation.tool_history and self.post_tool_adapter is not None:
            edit = self.post_tool_adapter.predict(
                observation.belief,
                observation.tool_history,
            )
        else:
            _, edit = self.predictor.predict(self.environment.current_sample())
        if edit.value == "KEEP":
            return AgentAction(action=AgentActionType.REJECT)
        return AgentAction(action=AgentActionType.COMMIT, edit=edit)


class PostToolAdaptedPolicy:
    """Override a base policy's terminal edit after grounded tool execution."""

    def __init__(
        self,
        base_policy: Any,
        adapter: PostToolActionAdapterPredictor,
    ) -> None:
        self.base_policy = base_policy
        self.adapter = adapter

    def act(self, observation: AgentObservation) -> AgentAction:
        action = self.base_policy.act(observation)
        if (
            not observation.tool_history
            or action.action
            in {AgentActionType.ACQUIRE, AgentActionType.USE_TOOL}
        ):
            return action
        edit = self.adapter.predict(observation.belief, observation.tool_history)
        if edit == EditOperation.KEEP:
            return AgentAction(action=AgentActionType.REJECT)
        return AgentAction(action=AgentActionType.COMMIT, edit=edit)


class RankerOnlyPolicy:
    """Frozen observable ranker with its train-calibrated STOP margin."""

    def __init__(
        self,
        ranker: CandidateUtilityRankerPredictor,
        sample: SelectorSample,
        episode: EpisodeRecord,
    ) -> None:
        self.ranker = ranker
        self.sample = sample
        self.episode = episode

    def act(self, observation: AgentObservation) -> AgentAction:
        state = active_catalog_state_from_observation(
            observation,
            self.sample,
            self.episode,
            policy_snapshot="frozen_observable_ranker",
        )
        decision, evidence_id, _ = self.ranker.decide(state)
        if decision == "STOP":
            return terminal_agent_action(observation)
        if evidence_id is None:
            raise ValueError("ranker ACQUIRE decision did not name evidence")
        return AgentAction(
            action=AgentActionType.ACQUIRE,
            evidence_id=evidence_id,
        )


def stable_seed(sample: SelectorSample, seed: int) -> int:
    payload = (
        f"{sample.metadata['source_episode']}|{sample.metadata['budget']}|{seed}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def deterministic_sample(
    samples: list[SelectorSample], limit: int, seed: int
) -> list[SelectorSample]:
    if limit <= 0:
        raise ValueError("sample limit must be positive")
    if limit >= len(samples):
        return samples

    def key(sample: SelectorSample) -> tuple[bytes, str]:
        identity = (
            f"{sample.metadata['source_episode']}|"
            f"{float(sample.metadata['budget']):.8f}|{seed}"
        ).encode()
        return hashlib.sha256(identity).digest(), sample.sample_id

    return sorted(samples, key=key)[:limit]


def parse_learned_selector(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError(
            "learned selector must use LABEL=/path/to/checkpoint.pt"
        )
    if not label.replace("_", "").isalnum():
        raise argparse.ArgumentTypeError(
            "learned selector labels may contain letters, digits, and underscores"
        )
    return label, Path(raw_path)


def parse_learned_selector_stop_margin(value: str) -> tuple[str, float]:
    label, separator, raw_margin = value.partition("=")
    if not separator or not label or not raw_margin:
        raise argparse.ArgumentTypeError(
            "learned selector STOP margin must use LABEL=FLOAT"
        )
    if not label.replace("_", "").isalnum():
        raise argparse.ArgumentTypeError(
            "learned selector labels may contain letters, digits, and underscores"
        )
    try:
        return label, float(raw_margin)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("learned selector STOP margin must be numeric") from exc


def parse_root_map(value: str) -> tuple[str, str]:
    source, separator, target = value.partition("=")
    if not separator or not source or not target:
        raise argparse.ArgumentTypeError(
            "asset root maps must use SOURCE=TARGET"
        )
    return source.rstrip("/"), target.rstrip("/")


def remap_asset_path(path: str, root_maps: list[tuple[str, str]]) -> str:
    for source, target in root_maps:
        if path == source or path.startswith(source + "/"):
            return target + path[len(source) :]
    return path


def make_environment(
    sample: SelectorSample,
    max_candidates: int,
    *,
    score_fn: Any | None = None,
    episode: EpisodeRecord | None = None,
    tool_registry: Any | None = None,
    tool_belief_updater: Any | None = None,
    tool_out_size: int = 128,
    asset_root_maps: list[tuple[str, str]] | None = None,
    semantic_sample: UpdaterSample | None = None,
    semantic_thresholds: dict[str, float] | None = None,
) -> MapMaintenanceEnv:
    asset_paths = None
    tool_parameters = None
    tool_inputs_by_evidence = None
    if episode is not None and tool_registry is not None:
        root_maps = asset_root_maps or []
        asset_paths = {
            item.evidence_id: remap_asset_path(item.image_path, root_maps)
            for item in episode.evidence_catalog
        }
        tool_parameters = {
            item.evidence_id: {
                "pixel_window": [
                    item.region[0],
                    item.region[1],
                    item.region[2] - item.region[0],
                    item.region[3] - item.region[1],
                ],
                "out_size": [tool_out_size, tool_out_size],
            }
            for item in episode.evidence_catalog
        }
        if semantic_sample is not None:
            semantic_inputs = {
                "image_path": remap_asset_path(semantic_sample.image_path, root_maps),
                "prior_mask_path": remap_asset_path(
                    semantic_sample.prior_mask_path, root_maps
                ),
            }
            tool_inputs_by_evidence = {
                item.evidence_id: semantic_inputs
                for item in episode.evidence_catalog
            }
            tool_parameters = {
                item.evidence_id: {
                    "threshold": semantic_thresholds["current"],
                    "add_threshold": semantic_thresholds["add"],
                    "remove_threshold": semantic_thresholds["remove"],
                }
                for item in episode.evidence_catalog
            }
    return MapMaintenanceEnv(
        sample,
        budget=float(sample.metadata["budget"]),
        score_fn=(
            score_fn
            if score_fn is not None
            else lambda current: observable_shortlist_scores(
                current, max_candidates
            )
        ),
        belief_updater=CounterfactualBeliefUpdater(sample),
        top_k=max_candidates,
        public_identifiers=True,
        tool_registry=tool_registry,
        tool_belief_updater=tool_belief_updater,
        asset_paths=asset_paths,
        tool_parameters=tool_parameters,
        tool_inputs_by_evidence=tool_inputs_by_evidence,
    )


def policies(
    sample: SelectorSample,
    environment: MapMaintenanceEnv,
    seed: int,
    *,
    ranker: CandidateUtilityRankerPredictor | None = None,
    episode: EpisodeRecord | None = None,
) -> dict[str, Any]:
    result = {
        "always_stop": AlwaysStopPolicy(),
        "cheapest": CandidatePolicy(lambda item: -float(item.cost)),
        "clear_per_cost": CandidatePolicy(
            lambda item: float(item.features[0]) / max(float(item.cost), 1e-8)
        ),
        "uncertainty_gate": UncertaintyAwareAgentPolicy(),
        "random": RandomCandidatePolicy(stable_seed(sample, seed)),
        "shortlist_oracle_upper_bound": OraclePolicy(environment),
    }
    if ranker is not None:
        if episode is None:
            raise ValueError("ranker-only baseline requires the source episode")
        result["ranker_only"] = RankerOnlyPolicy(ranker, sample, episode)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("states", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--max-acquisitions", type=int, default=2)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--frozen-test", action="store_true")
    parser.add_argument("--ranker-checkpoint", type=Path)
    parser.add_argument("--episodes", type=Path)
    parser.add_argument(
        "--tool-mode",
        choices=(
            "none",
            "forced",
            "selective",
            "semantic-forced",
            "semantic-selective",
        ),
        default="none",
    )
    parser.add_argument(
        "--belief-mode",
        choices=("recurrent", "frozen_prior"),
        default="recurrent",
    )
    parser.add_argument("--tool-belief-checkpoint", type=Path)
    parser.add_argument("--post-tool-action-adapter", type=Path)
    parser.add_argument("--tool-gate", type=Path)
    parser.add_argument("--tool-gate-summary", type=Path)
    parser.add_argument("--tool-artifact-root", type=Path)
    parser.add_argument("--tool-out-size", type=int, default=128)
    parser.add_argument("--semantic-updater-manifest", type=Path)
    parser.add_argument("--semantic-sam-road-repo", type=Path)
    parser.add_argument("--semantic-sam-road-config", type=Path)
    parser.add_argument("--semantic-source-checkpoint", type=Path)
    parser.add_argument("--semantic-sam-checkpoint", type=Path)
    parser.add_argument("--semantic-change-checkpoint", type=Path)
    parser.add_argument("--semantic-change-summary", type=Path)
    parser.add_argument("--semantic-device", default="cuda")
    parser.add_argument("--semantic-gate", type=Path)
    parser.add_argument("--semantic-gate-summary", type=Path)
    parser.add_argument(
        "--asset-root-map",
        action="append",
        type=parse_root_map,
        default=[],
    )
    parser.add_argument(
        "--learned-selector", action="append", type=parse_learned_selector, default=[]
    )
    parser.add_argument(
        "--evidence-value", action="append", type=parse_learned_selector, default=[]
    )
    parser.add_argument(
        "--structured-action",
        action="append",
        type=parse_learned_selector,
        default=[],
    )
    parser.add_argument(
        "--learned-selector-stop-margin",
        action="append",
        type=parse_learned_selector_stop_margin,
        default=[],
    )
    parser.add_argument(
        "--policy",
        action="append",
        default=[],
        help="Evaluate only the named policy; repeat for multiple policies.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--sample-seed",
        type=int,
        help="Select a deterministic hash sample instead of the first LIMIT rows.",
    )
    args = parser.parse_args()
    test_assets_read = authorize_split(args.split, args.frozen_test)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.max_candidates <= 0 or args.max_acquisitions < 0:
        raise ValueError("invalid closed-loop limits")
    if args.ranker_checkpoint is not None and args.episodes is None:
        raise ValueError("--ranker-checkpoint requires --episodes")
    if args.tool_mode != "none":
        if args.episodes is None or args.tool_artifact_root is None:
            raise ValueError(
                "tool modes require --episodes and --tool-artifact-root"
            )
        if args.tool_mode in {"forced", "selective"} and (
            args.tool_belief_checkpoint is None
        ):
            raise ValueError("weak tool modes require --tool-belief-checkpoint")
        if args.tool_mode == "selective" and (
            args.tool_gate is None or args.tool_gate_summary is None
        ):
            raise ValueError(
                "selective tool mode requires --tool-gate and --tool-gate-summary"
            )
        if args.tool_mode in {"semantic-forced", "semantic-selective"} and any(
            value is None
            for value in (
                args.semantic_updater_manifest,
                args.semantic_sam_road_repo,
                args.semantic_sam_road_config,
                args.semantic_source_checkpoint,
                args.semantic_sam_checkpoint,
                args.semantic_change_checkpoint,
                args.semantic_change_summary,
                args.post_tool_action_adapter,
            )
        ):
            raise ValueError(
                "semantic tool modes require updater, SAM-Road, change, "
                "and post-tool adapter checkpoints"
            )
        if args.tool_mode == "semantic-selective" and (
            args.semantic_gate is None or args.semantic_gate_summary is None
        ):
            raise ValueError(
                "semantic-selective requires --semantic-gate and "
                "--semantic-gate-summary"
            )
    elif args.post_tool_action_adapter is not None:
        raise ValueError("--post-tool-action-adapter requires a tool mode")
    if args.belief_mode == "frozen_prior" and args.tool_mode not in {
        "forced",
        "selective",
    }:
        raise ValueError("frozen_prior belief ablation requires a weak tool mode")
    if args.tool_out_size <= 0:
        raise ValueError("--tool-out-size must be positive")

    samples = load_initial_samples(
        args.states,
        split=args.split,
        limit=None if args.sample_seed is not None else args.limit,
    )
    if args.sample_seed is not None:
        if args.limit is None:
            raise ValueError("--sample-seed requires --limit")
        samples = deterministic_sample(samples, args.limit, args.sample_seed)
    ranker = (
        CandidateUtilityRankerPredictor(str(args.ranker_checkpoint), "cpu")
        if args.ranker_checkpoint is not None
        else None
    )
    episodes = (
        load_episodes(args.episodes, split=args.split)
        if args.episodes is not None
        else {}
    )
    tool_registry = None
    paired_tool_updater = None
    sequential_tool_updater_class = None
    tool_gate = None
    tool_gate_threshold = 0.0
    semantic_samples: dict[str, UpdaterSample] = {}
    semantic_thresholds = None
    semantic_gate = None
    semantic_gate_threshold = 0.0
    if args.tool_mode in {"forced", "selective"}:
        from activemap.agent.tool_belief_model import (
            FrozenPriorSequentialPairedToolBeliefUpdater,
            PairedToolBeliefUpdater,
            SequentialPairedToolBeliefUpdater,
        )
        from activemap.geo_tools.raster import ImageQualityTool, TemporalChangeTool
        from activemap.geo_tools.registry import GeoToolRegistry

        paired_tool_updater = PairedToolBeliefUpdater.from_checkpoint(
            args.tool_belief_checkpoint,
            device="cpu",
        )
        sequential_tool_updater_class = (
            SequentialPairedToolBeliefUpdater
            if args.belief_mode == "recurrent"
            else FrozenPriorSequentialPairedToolBeliefUpdater
        )
        tool_registry = GeoToolRegistry()
        tool_registry.register(ImageQualityTool())
        tool_registry.register(
            TemporalChangeTool(args.tool_artifact_root / "temporal_change")
        )
        if args.tool_mode == "selective":
            gate_summary = json.loads(
                args.tool_gate_summary.read_text(encoding="utf-8")
            )
            if gate_summary.get("schema_version") == "uncertainty-tool-gate-summary-v1":
                from activemap.agent.active_catalog_tool_gate import (
                    BeliefUncertaintyGate,
                )

                tool_gate = BeliefUncertaintyGate()
            else:
                from joblib import load

                tool_gate = load(args.tool_gate)
            tool_gate_threshold = float(gate_summary["selected"]["threshold"])
    elif args.tool_mode in {"semantic-forced", "semantic-selective"}:
        from activemap.geo_tools.model_segmentation import (
            MapConditionedSegmentationTool,
        )
        from activemap.geo_tools.registry import GeoToolRegistry

        change_summary = json.loads(
            args.semantic_change_summary.read_text(encoding="utf-8")
        )
        if change_summary.get("frozen_test_access") is not False:
            raise ValueError("semantic change summary does not freeze test access")
        selected = change_summary.get("selected_channel_thresholds")
        if not isinstance(selected, dict):
            raise ValueError("semantic change summary lacks selected thresholds")
        semantic_thresholds = {
            name: float(selected[name]) for name in ("current", "add", "remove")
        }
        semantic_samples = {
            public_task_id(f"{row.sample_id}__temporal"): row
            for row in load_updater_samples(args.semantic_updater_manifest)
            if row.split in {"train", "val"}
        }
        tool_registry = GeoToolRegistry()
        tool_registry.register(
            MapConditionedSegmentationTool.from_prior_conditioned_sam_road(
                args.semantic_sam_road_repo,
                args.semantic_sam_road_config,
                args.semantic_source_checkpoint,
                args.semantic_sam_checkpoint,
                args.semantic_change_checkpoint,
                args.tool_artifact_root / "raster_segment",
                device=args.semantic_device,
            )
        )
        if args.tool_mode == "semantic-selective":
            if args.semantic_gate.suffix.lower() == ".json":
                from activemap.agent.active_catalog_tool_gate import (
                    LinearProbabilityGate,
                )

                semantic_gate = LinearProbabilityGate.from_json(
                    args.semantic_gate
                )
            else:
                from joblib import load

                semantic_gate = load(args.semantic_gate)
            semantic_gate_summary = json.loads(
                args.semantic_gate_summary.read_text(encoding="utf-8")
            )
            semantic_gate_threshold = float(
                semantic_gate_summary["selected"]["threshold"]
            )
    learned_selector_paths = dict(args.learned_selector)
    if len(learned_selector_paths) != len(args.learned_selector):
        raise ValueError("duplicate learned-selector labels")
    evidence_value_paths = dict(args.evidence_value)
    if len(evidence_value_paths) != len(args.evidence_value):
        raise ValueError("duplicate evidence-value labels")
    structured_action_paths = dict(args.structured_action)
    if len(structured_action_paths) != len(args.structured_action):
        raise ValueError("duplicate structured-action labels")
    learned_selector_stop_margins = dict(args.learned_selector_stop_margin)
    if len(learned_selector_stop_margins) != len(args.learned_selector_stop_margin):
        raise ValueError("duplicate learned-selector STOP-margin labels")
    if unknown_margins := sorted(
        learned_selector_stop_margins.keys() - learned_selector_paths.keys()
    ):
        raise ValueError(
            "STOP-margin overrides lack learned selectors: " f"{unknown_margins}"
        )
    reserved = {
        "always_stop",
        "cheapest",
        "clear_per_cost",
        "uncertainty_gate",
        "random",
        "shortlist_oracle_upper_bound",
        "ranker_only",
    }
    custom_labels = (
        set(learned_selector_paths)
        | set(evidence_value_paths)
        | set(structured_action_paths)
    )
    label_groups = (
        set(learned_selector_paths),
        set(evidence_value_paths),
        set(structured_action_paths),
    )
    overlap = set()
    for index, group in enumerate(label_groups):
        for other in label_groups[index + 1 :]:
            overlap.update(group & other)
    if overlap:
        raise ValueError(f"duplicate custom policy labels: {overlap}")
    if collisions := sorted(reserved & custom_labels):
        raise ValueError(f"custom policy labels are reserved: {collisions}")
    available_policies = set(reserved) | custom_labels
    if unknown := sorted(set(args.policy) - available_policies):
        raise ValueError(f"unknown requested policies: {unknown}")
    learned_selectors = {
        label: SelectorPredictor(
            path,
            device=args.device,
            stop_margin_override=learned_selector_stop_margins.get(label),
        )
        for label, path in learned_selector_paths.items()
    }
    evidence_values = {
        label: EvidenceValuePredictor(str(path), device=args.device)
        for label, path in evidence_value_paths.items()
    }
    structured_actions = {
        label: StructuredMapActionPredictor(str(path), device=args.device)
        for label, path in structured_action_paths.items()
    }
    post_tool_adapter = (
        PostToolActionAdapterPredictor(
            str(args.post_tool_action_adapter),
            device=args.device,
        )
        if args.post_tool_action_adapter is not None
        else None
    )
    rows_by_policy: dict[str, list[dict[str, Any]]] = {}
    post_tool_states: list[SelectorSample] = []
    tool_result_rows: list[dict[str, Any]] = []
    joint_transition_rows = []
    for sample in tqdm(samples, desc="Closed-loop baselines"):
        policy_names = (
            "always_stop",
            "cheapest",
            "clear_per_cost",
            "uncertainty_gate",
            "random",
            "shortlist_oracle_upper_bound",
        )
        if ranker is not None:
            policy_names = (*policy_names, "ranker_only")
        policy_names = (
            *policy_names,
            *sorted(learned_selectors),
            *sorted(evidence_values),
            *sorted(structured_actions),
        )
        if args.policy:
            requested = set(args.policy)
            policy_names = tuple(name for name in policy_names if name in requested)
        for policy_name in policy_names:
            learned = learned_selectors.get(policy_name)
            value_predictor = evidence_values.get(policy_name)
            structured_predictor = structured_actions.get(policy_name)
            source_episode = str(sample.metadata["source_episode"])
            episode = episodes.get(source_episode)
            semantic_sample = (
                semantic_samples.get(public_task_id(source_episode))
                if args.tool_mode in {"semantic-forced", "semantic-selective"}
                else None
            )
            if (
                args.tool_mode in {"semantic-forced", "semantic-selective"}
                and semantic_sample is None
            ):
                raise ValueError(
                    f"missing semantic updater sample for {source_episode}"
                )
            environment = make_environment(
                sample,
                args.max_candidates,
                score_fn=(
                    learned.action_scores
                    if learned is not None
                    else value_predictor or structured_predictor
                ),
                episode=episode,
                tool_registry=tool_registry,
                tool_belief_updater=(
                    sequential_tool_updater_class(paired_tool_updater)
                    if paired_tool_updater is not None
                    and sequential_tool_updater_class is not None
                    else None
                ),
                tool_out_size=args.tool_out_size,
                asset_root_maps=args.asset_root_map,
                semantic_sample=semantic_sample,
                semantic_thresholds=semantic_thresholds,
            )
            policy = (
                StructuredActionPolicy(
                    structured_predictor,
                    environment,
                    post_tool_adapter=post_tool_adapter,
                )
                if structured_predictor is not None
                else GreedyAgentPolicy()
                if learned is not None or value_predictor is not None
                else policies(
                    sample,
                    environment,
                    args.seed,
                    ranker=ranker,
                    episode=episodes.get(source_episode),
                )[policy_name]
            )
            if post_tool_adapter is not None and structured_predictor is None:
                policy = PostToolAdaptedPolicy(policy, post_tool_adapter)
            if args.tool_mode in {"forced", "selective"}:
                policy = SelectiveToolPolicy(
                    policy,
                    mode=args.tool_mode,
                    gate=tool_gate,
                    threshold=tool_gate_threshold,
                )
            elif args.tool_mode == "semantic-forced":
                policy = ForcedInitialSemanticToolPolicy(policy)
            elif args.tool_mode == "semantic-selective":
                policy = SelectiveInitialSemanticToolPolicy(
                    policy,
                    environment.current_sample,
                    semantic_gate,
                    threshold=semantic_gate_threshold,
                )
            trajectory = rollout_agent_policy(
                environment,
                policy,
                max_acquisitions=args.max_acquisitions,
                max_tool_calls=(
                    0
                    if args.tool_mode == "none"
                    else 1
                    if args.tool_mode in {"semantic-forced", "semantic-selective"}
                    else 2 * args.max_acquisitions
                ),
                max_steps=(
                    args.max_acquisitions + 1
                    if args.tool_mode == "none"
                    else args.max_acquisitions + 2
                    if args.tool_mode in {"semantic-forced", "semantic-selective"}
                    else 3 * args.max_acquisitions + 1
                ),
            )
            if (
                learned is not None
                or value_predictor is not None
                or structured_predictor is not None
            ):
                policy_snapshot = (
                    str(learned_selector_paths[policy_name].resolve())
                    if learned is not None
                    else str(evidence_value_paths[policy_name].resolve())
                    if value_predictor is not None
                    else str(structured_action_paths[policy_name].resolve())
                )
                for transition in trajectory.transitions:
                    exported = joint_transition_from_agent_transition(
                        transition,
                        source_episode=source_episode,
                        aoi_id=str(sample.metadata["aoi_id"]),
                        policy_snapshot=policy_snapshot,
                        target_edit=EditOperation(str(sample.metadata["gt_edit"])),
                        selected_by_model=True,
                        operation_update_threshold=float(
                            sample.metadata.get("operation_update_threshold", 0.5)
                        ),
                    )
                    if exported is not None:
                        joint_transition_rows.append(exported)
            if args.tool_mode in {"semantic-forced", "semantic-selective"} and any(
                not result.success for result in environment.tool_history
            ):
                errors = [
                    result.error
                    for result in environment.tool_history
                    if not result.success
                ]
                raise RuntimeError(
                    f"semantic tool failed for {source_episode}: {errors}"
                )
            if (
                args.tool_mode != "none"
                and int(trajectory.metadata["tool_call_count"]) > 0
            ):
                tool_result_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "source_episode": source_episode,
                        "policy": policy_name,
                        "results": [
                            result.model_dump(mode="json")
                            for result in environment.tool_history
                        ],
                        "test_assets_read": test_assets_read,
                    }
                )
                replay = environment.current_sample()
                replay_metadata = {
                    **replay.metadata,
                    "oracle_step": environment.step_index,
                    "post_tool_terminal_replay": True,
                    "tool_call_count": int(
                        trajectory.metadata["tool_call_count"]
                    ),
                    "tool_mode": args.tool_mode,
                    "test_assets_read": test_assets_read,
                }
                post_tool_states.append(
                    replay.model_copy(
                        update={
                            "sample_id": (
                                f"{replay.sample_id}__{policy_name}"
                                f"__posttool_s{environment.step_index}"
                            ),
                            "metadata": replay_metadata,
                        }
                    )
                )
            rows_by_policy.setdefault(policy_name, []).append(
                trajectory_row(
                    sample,
                    environment,
                    trajectory,
                    policy_name=policy_name,
                    events=getattr(policy, "events", None),
                )
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tool_result_path = args.output_dir / "tool_results.jsonl"
    if tool_result_rows:
        tool_result_path.write_text(
            "".join(
                json.dumps(row, separators=(",", ":")) + "\n"
                for row in tool_result_rows
            ),
            encoding="utf-8",
        )
    post_tool_state_path = args.output_dir / "post_tool_terminal_states.jsonl"
    if post_tool_states:
        post_tool_state_path.write_text(
            "".join(row.model_dump_json() + "\n" for row in post_tool_states),
            encoding="utf-8",
        )
    joint_transition_path = args.output_dir / "joint_transitions.jsonl"
    if joint_transition_rows:
        joint_transition_path.write_text(
            "".join(row.model_dump_json() + "\n" for row in joint_transition_rows),
            encoding="utf-8",
        )
    summaries = {}
    for policy_name, rows in rows_by_policy.items():
        trace_path = args.output_dir / f"{policy_name}.jsonl"
        trace_path.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        summaries[policy_name] = {
            "metrics": metrics(rows),
            "aoi_bootstrap": (
                grouped_bootstrap(rows, args.bootstrap_repetitions, args.seed)
                if args.bootstrap_repetitions > 0
                else None
            ),
            "trace": str(trace_path.resolve()),
        }
    summary = {
        "schema_version": "active-catalog-closed-loop-baselines-v1",
        "sample_count": len(samples),
        "split": args.split,
        "post_tool_terminal_states": len(post_tool_states),
        "post_tool_terminal_state_path": (
            str(post_tool_state_path.resolve()) if post_tool_states else None
        ),
        "tool_result_trace": (
            str(tool_result_path.resolve()) if tool_result_rows else None
        ),
        "joint_transition_count": len(joint_transition_rows),
        "joint_transition_path": (
            str(joint_transition_path.resolve()) if joint_transition_rows else None
        ),
        "policies": summaries,
        "protocol": {
            "same_states_and_budgets": True,
            "same_frozen_belief_updater": True,
            "same_max_acquisitions": args.max_acquisitions,
            "tool_mode": args.tool_mode,
            "max_tool_calls": (
                0
                if args.tool_mode == "none"
                else 1
                if args.tool_mode in {"semantic-forced", "semantic-selective"}
                else 2 * args.max_acquisitions
            ),
            "explicit_geospatial_tool_calls": args.tool_mode != "none",
            "selective_tool_calling": args.tool_mode
            in {"selective", "semantic-selective"},
            "map_relative_semantic_tool": args.tool_mode
            in {"semantic-forced", "semantic-selective"},
            "semantic_selective_calling": args.tool_mode == "semantic-selective",
            "inputs": {
                "states": {
                    "path": str(args.states.resolve()),
                    "sha256": hashlib.sha256(
                        args.states.read_bytes()
                    ).hexdigest(),
                },
                "episodes": (
                    {
                        "path": str(args.episodes.resolve()),
                        "sha256": hashlib.sha256(
                            args.episodes.read_bytes()
                        ).hexdigest(),
                    }
                    if args.episodes is not None
                    else None
                ),
            },
            "semantic_gate": (
                {
                    "checkpoint": str(args.semantic_gate.resolve()),
                    "sha256": hashlib.sha256(
                        args.semantic_gate.read_bytes()
                    ).hexdigest(),
                    "summary": str(args.semantic_gate_summary.resolve()),
                    "summary_sha256": hashlib.sha256(
                        args.semantic_gate_summary.read_bytes()
                    ).hexdigest(),
                    "threshold": semantic_gate_threshold,
                }
                if args.tool_mode == "semantic-selective"
                else None
            ),
            "oracle_is_upper_bound_only": True,
            "ranker_only_uses_train_calibrated_safety_margin": ranker is not None,
            "learned_selectors": {
                label: {
                    "checkpoint": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "condition_on_hypothesis": learned_selectors[
                        label
                    ].ablation.condition_on_hypothesis,
                    "checkpoint_stop_margin": learned_selectors[
                        label
                    ].checkpoint_stop_margin,
                    "effective_stop_margin": learned_selectors[label].stop_margin,
                    "stop_margin_source": learned_selectors[label].stop_margin_source,
                }
                for label, path in learned_selector_paths.items()
            },
            "evidence_value_heads": {
                label: {
                    "checkpoint": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "safety_margin": evidence_values[label].safety_margin,
                    "unsafe_penalty": evidence_values[label].unsafe_penalty,
                    "missed_penalty": evidence_values[label].missed_penalty,
                }
                for label, path in evidence_value_paths.items()
            },
            "structured_action_policies": {
                label: {
                    "checkpoint": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "safety_margin": structured_actions[label].safety_margin,
                    "unsafe_penalty": structured_actions[label].unsafe_penalty,
                    "missed_penalty": structured_actions[label].missed_penalty,
                    "joint_terminal_edit": True,
                }
                for label, path in structured_action_paths.items()
            },
            "post_tool_action_adapter": (
                {
                    "checkpoint": str(args.post_tool_action_adapter.resolve()),
                    "sha256": hashlib.sha256(
                        args.post_tool_action_adapter.read_bytes()
                    ).hexdigest(),
                    "applies_only_after_tool_history": True,
                }
                if args.post_tool_action_adapter is not None
                else None
            ),
            "evaluated_policies": sorted(rows_by_policy),
            "selector_device": args.device,
            "deterministic_sample_seed": args.sample_seed,
            "belief_mode": args.belief_mode,
            "tool_belief_checkpoint": (
                {
                    "path": str(args.tool_belief_checkpoint.resolve()),
                    "sha256": hashlib.sha256(
                        args.tool_belief_checkpoint.read_bytes()
                    ).hexdigest(),
                    "reliability_gate": bool(
                        paired_tool_updater.model.reliability_gate
                    ),
                }
                if paired_tool_updater is not None
                else None
            ),
            "tool_gate": (
                {
                    "path": str(args.tool_gate.resolve()),
                    "sha256": hashlib.sha256(args.tool_gate.read_bytes()).hexdigest(),
                    "summary": str(args.tool_gate_summary.resolve()),
                }
                if args.tool_mode == "selective"
                else None
            ),
            "quality_gain_semantics": (
                "executable_terminal_score"
                if all(
                    str(sample.metadata.get("utility_mode", "proxy")) == "executable"
                    for sample in samples
                )
                else "frozen_teacher_proxy"
            ),
            "test_assets_read": test_assets_read,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
