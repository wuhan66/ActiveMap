#!/usr/bin/env python3
"""Run a promoted Gate+Ranker policy through closed loop and executable writeback."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    expected_output: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("decision", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("ranker_checkpoint", type=Path)
    parser.add_argument("source_states", type=Path)
    parser.add_argument("source_episodes", type=Path)
    parser.add_argument("val_sft", type=Path)
    parser.add_argument("val_evaluation_index", type=Path)
    parser.add_argument("updater_checkpoint", type=Path)
    parser.add_argument("bundle_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--max-acquisitions", type=int, default=2)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--asset-root-map", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_promoted_adapter(
    decision_path: Path, adapter_override: Path | None = None
) -> tuple[dict[str, Any], Path]:
    decision = _read_json(decision_path)
    if decision.get("test_assets_read") is not False:
        raise ValueError("Gate+Ranker selection must be validation-only")
    promotion = decision.get("promotion")
    if not isinstance(promotion, dict) or promotion.get("passed") is not True:
        raise PermissionError("Gate+Ranker promotion did not pass")

    if adapter_override is not None:
        return decision, adapter_override

    report_path = decision_path.with_name("report.json")
    report = _read_json(report_path)
    selected_label = decision.get("selected_tuning_run")
    selected = report.get("runs", {}).get(selected_label)
    if not isinstance(selected, dict) or not selected.get("trace"):
        raise ValueError("selected tuning run is absent from the round report")
    trace = Path(str(selected["trace"]))
    # <family>/<seed>/active_catalog_gate_ranker_val/traces.jsonl -> <seed>/final
    return decision, trace.parent.parent / "final"


def build_stages(args: argparse.Namespace, adapter: Path) -> list[Stage]:
    bundle_states = args.bundle_root / "states_val_step0.jsonl"
    bundle_episodes = args.bundle_root / "episodes_val.jsonl"
    closed_root = args.output_root / "closed_loop"
    closed_traces = closed_root / "evaluation" / "traces.jsonl"
    rollout_input = args.output_root / "writeback_inputs" / "gate_ranker.jsonl"
    writeback_root = args.output_root / "writeback"

    stages: list[Stage] = []
    if not (args.bundle_root / "summary.json").is_file():
        stages.append(
            Stage(
                "prepare_bundle",
                [
                    args.python,
                    "scripts/prepare_active_catalog_closed_loop_bundle.py",
                    str(args.source_states),
                    str(args.source_episodes),
                    str(args.bundle_root),
                    "--split",
                    "val",
                ],
                args.bundle_root / "summary.json",
            )
        )

    closed_command = [
        args.python,
        "scripts/launch_active_catalog_closed_loop.py",
        str(args.model),
        str(adapter),
        str(bundle_states),
        str(bundle_episodes),
        str(args.val_sft),
        str(args.val_evaluation_index),
        str(closed_root),
        "--gpu",
        str(args.gpu),
        "--python",
        args.python,
        "--seed",
        str(args.seed),
        "--max-candidates",
        str(args.max_candidates),
        "--max-acquisitions",
        str(args.max_acquisitions),
        "--bootstrap-repetitions",
        str(args.bootstrap_repetitions),
        "--policy-mode",
        "gate_ranker",
        "--ranker-checkpoint",
        str(args.ranker_checkpoint),
    ]
    if args.limit is not None:
        closed_command.extend(["--limit", str(args.limit)])
    stages.append(Stage("closed_loop", closed_command, closed_traces))

    stages.append(
        Stage(
            "convert_writeback_input",
            [
                args.python,
                "scripts/convert_active_catalog_closed_loop_for_writeback.py",
                str(closed_traces),
                str(rollout_input),
                "--split",
                "val",
            ],
            rollout_input,
        )
    )

    writeback_command = [
        args.python,
        "scripts/launch_active_catalog_writeback.py",
        str(args.updater_checkpoint),
        str(bundle_episodes),
        str(rollout_input),
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
        "sn7-promoted-gate-ranker-vector-writeback-v2",
        "--split",
        "val",
    ]
    for mapping in args.asset_root_map:
        writeback_command.extend(["--asset-root-map", mapping])
    if args.limit is not None:
        writeback_command.extend(["--limit", str(args.limit)])
    stages.append(
        Stage(
            "executable_writeback",
            writeback_command,
            writeback_root / "evaluation" / "summary.json",
        )
    )
    return stages


def validate_inputs(args: argparse.Namespace, adapter: Path) -> None:
    required = (
        args.decision,
        args.ranker_checkpoint,
        args.source_states,
        args.source_episodes,
        args.val_sft,
        args.val_evaluation_index,
        args.updater_checkpoint,
        adapter / "adapter_config.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing pipeline inputs: {missing}")
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    bundle_summary = args.bundle_root / "summary.json"
    if args.bundle_root.exists() and not bundle_summary.is_file():
        raise FileExistsError(f"incomplete bundle root already exists: {args.bundle_root}")
    if bundle_summary.is_file():
        bundle_files = (
            args.bundle_root / "states_val_step0.jsonl",
            args.bundle_root / "episodes_val.jsonl",
        )
        if not all(path.is_file() for path in bundle_files):
            raise FileNotFoundError("closed-loop bundle summary exists without bundle files")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    decision, adapter = resolve_promoted_adapter(args.decision, args.adapter)
    validate_inputs(args, adapter)
    stages = build_stages(args, adapter)
    manifest = {
        "schema_version": "gate-ranker-closed-loop-writeback-launch-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": "val",
        "test_assets_read": False,
        "selected_tuning_run": decision["selected_tuning_run"],
        "selected_acquire_target": decision["selected_acquire_target"],
        "adapter": str(adapter.resolve()),
        "adapter_config_sha256": _sha256(adapter / "adapter_config.json"),
        "ranker_checkpoint": str(args.ranker_checkpoint.resolve()),
        "ranker_checkpoint_sha256": _sha256(args.ranker_checkpoint),
        "updater_checkpoint": str(args.updater_checkpoint.resolve()),
        "physical_gpu": args.gpu,
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

    writeback_summary = _read_json(
        args.output_root / "writeback" / "evaluation" / "summary.json"
    )
    final_state = {
        "status": "completed",
        "completed": completed,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "writeback_summary": writeback_summary,
    }
    _write_json(args.output_root / "pipeline_state.json", final_state)
    print(json.dumps(final_state, indent=2))


if __name__ == "__main__":
    main()
