#!/usr/bin/env python3
"""Run matched closed-loop baselines and executable writeback comparisons."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.assess_active_catalog_closed_loop_promotion import assess

DEPLOYMENT_POLICIES = (
    "always_stop",
    "clear_per_cost",
    "uncertainty_gate",
    "ranker_only",
)
WRITEBACK_POLICIES = (*DEPLOYMENT_POLICIES, "shortlist_oracle_upper_bound")
ALL_CLOSED_LOOP_POLICIES = (
    "always_stop",
    "cheapest",
    "clear_per_cost",
    "uncertainty_gate",
    "random",
    "shortlist_oracle_upper_bound",
    "ranker_only",
)


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    expected_output: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("states", type=Path)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("ranker_checkpoint", type=Path)
    parser.add_argument("updater_checkpoint", type=Path)
    parser.add_argument("candidate_closed_loop", type=Path)
    parser.add_argument("candidate_writeback", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--max-acquisitions", type=int, default=2)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--asset-root-map", action="append", default=[])
    parser.add_argument(
        "--learned-selector",
        action="append",
        default=[],
        metavar="LABEL=CHECKPOINT",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def learned_selector_paths(args: argparse.Namespace) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in getattr(args, "learned_selector", []):
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError("learned selectors must use LABEL=CHECKPOINT")
        if not label.replace("_", "").isalnum():
            raise ValueError("learned selector labels must be alphanumeric or underscore")
        if label in result:
            raise ValueError(f"duplicate learned selector label: {label}")
        result[label] = Path(raw_path)
    return result


def policy_sets(args: argparse.Namespace) -> tuple[tuple[str, ...], tuple[str, ...]]:
    labels = tuple(sorted(learned_selector_paths(args)))
    deployment = (*DEPLOYMENT_POLICIES, *labels)
    writeback = (*deployment, "shortlist_oracle_upper_bound")
    return deployment, writeback


def build_stages(args: argparse.Namespace) -> list[Stage]:
    selectors = learned_selector_paths(args)
    deployment_policies, writeback_policies = policy_sets(args)
    baseline_root = args.output_root / "closed_loop_baselines"
    baseline_command = [
        args.python,
        "scripts/evaluate_active_catalog_closed_loop_baselines.py",
        str(args.states),
        str(baseline_root),
        "--episodes",
        str(args.episodes),
        "--ranker-checkpoint",
        str(args.ranker_checkpoint),
        "--max-candidates",
        str(args.max_candidates),
        "--max-acquisitions",
        str(args.max_acquisitions),
        "--bootstrap-repetitions",
        str(args.bootstrap_repetitions),
        "--seed",
        str(args.seed),
    ]
    if args.limit is not None:
        baseline_command.extend(["--limit", str(args.limit)])
    for label, checkpoint in selectors.items():
        baseline_command.extend(
            ["--learned-selector", f"{label}={checkpoint}"]
        )
    stages = [
        Stage(
            "closed_loop_baselines",
            baseline_command,
            baseline_root / "summary.json",
        )
    ]

    for policy in writeback_policies:
        source = baseline_root / f"{policy}.jsonl"
        converted = args.output_root / "writeback_inputs" / f"{policy}.jsonl"
        stages.append(
            Stage(
                f"convert_{policy}",
                [
                    args.python,
                    "scripts/convert_active_catalog_closed_loop_for_writeback.py",
                    str(source),
                    str(converted),
                    "--split",
                    "val",
                ],
                converted,
            )
        )
        writeback_root = args.output_root / "writebacks" / policy
        command = [
            args.python,
            "scripts/launch_active_catalog_writeback.py",
            str(args.updater_checkpoint),
            str(args.episodes),
            str(converted),
            str(writeback_root),
            "--gpu",
            str(args.gpu),
            "--python",
            args.python,
            "--image-size",
            str(args.image_size),
            "--threshold",
            str(args.threshold),
            "--protocol-name",
            "sn7-matched-baseline-vector-writeback-v2",
            "--split",
            "val",
        ]
        for mapping in args.asset_root_map:
            command.extend(["--asset-root-map", mapping])
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        stages.append(
            Stage(
                f"writeback_{policy}",
                command,
                writeback_root / "evaluation" / "writeback.jsonl",
            )
        )

    closed_comparison = args.output_root / "comparisons" / "closed_loop.json"
    closed_command = [
        args.python,
        "scripts/compare_active_catalog_closed_loop.py",
        str(closed_comparison),
        "--candidate",
        "gate_ranker",
        "--records",
        f"gate_ranker={args.candidate_closed_loop}",
    ]
    for policy in (*ALL_CLOSED_LOOP_POLICIES, *sorted(selectors)):
        closed_command.extend(
            ["--records", f"{policy}={baseline_root / f'{policy}.jsonl'}"]
        )
    closed_command.extend(
        [
            "--repetitions",
            str(args.bootstrap_repetitions),
            "--seed",
            str(args.seed),
            "--split",
            "val",
        ]
    )
    stages.append(
        Stage("compare_closed_loop", closed_command, closed_comparison)
    )

    for policy in writeback_policies:
        comparison = args.output_root / "comparisons" / f"writeback_vs_{policy}.json"
        command = [
            args.python,
            "scripts/compare_agent_writebacks.py",
            str(
                args.output_root
                / "writebacks"
                / policy
                / "evaluation"
                / "writeback.jsonl"
            ),
            str(args.candidate_writeback),
            str(comparison),
            "--bootstrap",
            str(args.bootstrap_repetitions),
            "--seed",
            str(args.seed),
            "--group-key",
            "aoi_id",
            "--split",
            "val",
        ]
        stages.append(Stage(f"compare_writeback_{policy}", command, comparison))
    return stages


def finalize_matrix(
    output_root: Path,
    *,
    deployment_policies: tuple[str, ...] = DEPLOYMENT_POLICIES,
    writeback_policies: tuple[str, ...] = WRITEBACK_POLICIES,
) -> dict[str, Any]:
    closed_loop = json.loads(
        (output_root / "comparisons" / "closed_loop.json").read_text(
            encoding="utf-8"
        )
    )
    comparisons = {
        policy: json.loads(
            (
                output_root / "comparisons" / f"writeback_vs_{policy}.json"
            ).read_text(encoding="utf-8")
        )
        for policy in writeback_policies
    }
    utility_name = "episode_utility_v2_balanced_auc"
    missing = [
        policy
        for policy, comparison in comparisons.items()
        if utility_name not in comparison.get("baseline", {})
    ]
    if missing:
        raise ValueError(f"baseline writebacks lack executable Utility v2: {missing}")
    strongest = max(
        deployment_policies,
        key=lambda policy: (comparisons[policy]["baseline"][utility_name], policy),
    )
    promotion = assess(
        closed_loop,
        comparisons[strongest],
        candidate="gate_ranker",
        baseline=strongest,
    )
    return {
        "schema_version": "gate-ranker-matched-baseline-matrix-v1",
        "split": "val",
        "test_assets_read": False,
        "baseline_selection_rule": (
            "highest executable episode_utility_v2_balanced_auc among the frozen "
            "deployment baseline set"
        ),
        "strongest_deployment_baseline": strongest,
        "baseline_utility_v2_balanced_auc": {
            policy: comparison["baseline"][utility_name]
            for policy, comparison in comparisons.items()
        },
        "candidate_utility_v2_balanced_auc": comparisons[strongest]["candidate"][
            utility_name
        ],
        "promotion": promotion,
        "oracle_is_upper_bound_only": True,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate(args: argparse.Namespace) -> None:
    required = (
        args.states,
        args.episodes,
        args.ranker_checkpoint,
        args.updater_checkpoint,
        args.candidate_closed_loop,
        args.candidate_writeback,
    )
    missing = [str(path) for path in required if not path.is_file()]
    missing.extend(
        str(path)
        for path in learned_selector_paths(args).values()
        if not path.is_file()
    )
    if missing:
        raise FileNotFoundError(f"missing baseline-matrix inputs: {missing}")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")


def main() -> None:
    args = parse_args()
    validate(args)
    stages = build_stages(args)
    deployment_policies, writeback_policies = policy_sets(args)
    manifest = {
        "schema_version": "gate-ranker-matched-baseline-launch-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": "val",
        "test_assets_read": False,
        "physical_gpu": args.gpu,
        "deployment_policies": list(deployment_policies),
        "writeback_policies": list(writeback_policies),
        "learned_selectors": {
            label: str(path.resolve())
            for label, path in learned_selector_paths(args).items()
        },
        "stages": [
            {
                "name": stage.name,
                "command": stage.command,
                "expected_output": str(stage.expected_output),
            }
            for stage in stages
        ],
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    args.output_root.mkdir(parents=True)
    _write_json(args.output_root / "pipeline_manifest.json", manifest)
    completed: list[str] = []
    for stage in stages:
        _write_json(
            args.output_root / "pipeline_state.json",
            {"status": "running", "stage": stage.name, "completed": completed},
        )
        subprocess.run(stage.command, check=True)
        if not stage.expected_output.is_file():
            raise FileNotFoundError(
                f"stage {stage.name} did not produce {stage.expected_output}"
            )
        completed.append(stage.name)

    matrix = finalize_matrix(
        args.output_root,
        deployment_policies=deployment_policies,
        writeback_policies=writeback_policies,
    )
    _write_json(args.output_root / "matrix_summary.json", matrix)
    _write_json(
        args.output_root / "pipeline_state.json",
        {
            "status": "completed",
            "completed": completed,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "matrix_summary": matrix,
        },
    )
    print(json.dumps(matrix, indent=2))


if __name__ == "__main__":
    main()
