#!/usr/bin/env python3
"""Run labeling, assembly, gate fitting, and cached evaluation after cross-fit SFT."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("folds_root", type=Path)
    parser.add_argument("training_root", type=Path)
    parser.add_argument("val_feature_root", type=Path)
    parser.add_argument("val_branch_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=4.0)
    return parser.parse_args()


def labeling_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        "scripts/launch_policy_relative_crossfit_labeling.py",
        str(args.model),
        str(args.folds_root),
        str(args.training_root),
        str(args.output_root / "labeling"),
        "--seed",
        str(args.seed),
        "--pooling",
        "last_mean",
    ]
    for gpu in list(dict.fromkeys(args.gpu)):
        command.extend(["--gpu", str(gpu)])
    if (args.output_root / "labeling").exists():
        command.append("--resume")
    return command


def assembly_command(args: argparse.Namespace) -> list[str]:
    labeling = args.output_root / "labeling"
    return [
        args.python,
        "scripts/assemble_policy_relative_crossfit_features.py",
        str(args.folds_root),
        str(labeling / "branches"),
        str(labeling / "features"),
        str(args.val_branch_root),
        str(args.val_feature_root),
        str(args.output_root / "assembled_features"),
    ]


def gate_command(args: argparse.Namespace) -> list[str]:
    assembled = args.output_root / "assembled_features"
    return [
        args.python,
        "scripts/train_visual_tool_gate.py",
        str(assembled / "train"),
        str(assembled / "val"),
        str(args.output_root / "gate"),
        "--seed",
        str(args.seed),
        "--selection-objective",
        "proxy_utility",
        "--fit-weighting",
        "utility_magnitude",
    ]


def evaluation_command(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        "scripts/evaluate_cached_policy_relative_gate.py",
        str(args.output_root / "assembled_features" / "val"),
        str(args.output_root / "gate"),
        str(args.val_branch_root),
        str(args.output_root / "evaluation"),
    ]


def _write_state(path: Path, **values: Any) -> None:
    payload = {
        "schema_version": "policy-relative-crossfit-pipeline-state-v1",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "test_assets_read": False,
        **values,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run_phase(
    args: argparse.Namespace,
    state_path: Path,
    phase: str,
    command: list[str],
    summary_path: Path,
) -> None:
    if summary_path.is_file():
        _write_state(state_path, status=f"{phase}_reused", summary=str(summary_path.resolve()))
        return
    _write_state(state_path, status=f"{phase}_running", command=command)
    log_path = args.output_root / f"{phase}.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode or not summary_path.is_file():
        _write_state(
            state_path,
            status=f"{phase}_failed",
            returncode=completed.returncode,
            log=str(log_path.resolve()),
        )
        raise RuntimeError(f"pipeline phase failed: {phase}")
    _write_state(
        state_path,
        status=f"{phase}_complete",
        summary=str(summary_path.resolve()),
        log=str(log_path.resolve()),
    )


def main() -> None:
    args = parse_args()
    gpus = list(dict.fromkeys(args.gpu))
    if not gpus or len(gpus) > 2:
        raise ValueError("pipeline requires one or two GPUs")
    if args.poll_seconds < 10 or args.timeout_hours <= 0.0:
        raise ValueError("poll interval or timeout is invalid")
    args.output_root.mkdir(parents=True, exist_ok=True)
    state_path = args.output_root / "pipeline_state.json"
    training_results = args.training_root / "process_results.json"
    deadline = time.monotonic() + args.timeout_hours * 3600.0
    _write_state(state_path, status="waiting_for_crossfit_sft")
    while not training_results.is_file():
        if time.monotonic() >= deadline:
            _write_state(state_path, status="timed_out_waiting_for_crossfit_sft")
            raise TimeoutError("cross-fit SFT did not complete before timeout")
        time.sleep(args.poll_seconds)
    training = json.loads(training_results.read_text(encoding="utf-8"))
    if not training or any(row["returncode"] for row in training):
        _write_state(state_path, status="crossfit_sft_failed")
        raise RuntimeError("cross-fit SFT process results contain a failure")

    phases = (
        (
            "labeling",
            labeling_command(args),
            args.output_root / "labeling" / "process_results.json",
        ),
        (
            "assembly",
            assembly_command(args),
            args.output_root / "assembled_features" / "summary.json",
        ),
        ("gate", gate_command(args), args.output_root / "gate" / "summary.json"),
        (
            "evaluation",
            evaluation_command(args),
            args.output_root / "evaluation" / "summary.json",
        ),
    )
    for phase, command, summary_path in phases:
        _run_phase(args, state_path, phase, command, summary_path)
        if phase == "labeling":
            labeling_results = json.loads(summary_path.read_text(encoding="utf-8"))
            failures = [
                (row["fold"], item["phase"])
                for row in labeling_results
                for item in row["phases"]
                if item.get("returncode", 0)
            ]
            if failures:
                _write_state(state_path, status="labeling_failed", failures=failures)
                raise RuntimeError(f"cross-fit labeling failures: {failures}")
    evaluation = json.loads(
        (args.output_root / "evaluation" / "summary.json").read_text(encoding="utf-8")
    )
    _write_state(
        state_path,
        status="complete",
        promotion_gate=evaluation["promotion_gate"],
        three_seed_ready=bool(evaluation["promotion_gate"]["passed"]),
        rl_ready=False,
        frozen_test_ready=False,
    )


if __name__ == "__main__":
    main()
