#!/usr/bin/env python3
"""Wait for on-policy labels, then fit and evaluate one frozen utility-aware gate."""

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
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("train_feature_root", type=Path)
    parser.add_argument("val_branch_root", type=Path)
    parser.add_argument("val_feature_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout-hours", type=float, default=4.0)
    return parser.parse_args()


def _write_state(path: Path, **values: Any) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "policy-relative-onpolicy-gate-pipeline-v1",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "test_assets_read": False,
                **values,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _run(args: argparse.Namespace, state: Path, phase: str, command: list[str]) -> None:
    _write_state(state, status=f"{phase}_running", command=command)
    with (args.output_root / f"{phase}.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, text=True, check=False
        )
    if completed.returncode:
        _write_state(state, status=f"{phase}_failed", returncode=completed.returncode)
        raise RuntimeError(f"on-policy gate phase failed: {phase}")


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)
    state = args.output_root / "pipeline_state.json"
    merged = args.cache_root / "merged"
    deadline = time.monotonic() + args.timeout_hours * 3600.0
    _write_state(state, status="waiting_for_onpolicy_cache")
    while not (merged / "summary.json").is_file():
        launcher_state = args.cache_root / "launcher_state.json"
        if launcher_state.is_file():
            status = json.loads(launcher_state.read_text(encoding="utf-8"))["status"]
            if status in {"failed", "merge_failed"}:
                _write_state(state, status="cache_failed")
                raise RuntimeError("on-policy cache failed")
        if time.monotonic() >= deadline:
            _write_state(state, status="cache_timeout")
            raise TimeoutError("on-policy cache did not complete before timeout")
        time.sleep(30)

    features = args.output_root / "features"
    gate = args.output_root / "gate"
    evaluation = args.output_root / "evaluation"
    _run(
        args,
        state,
        "assembly",
        [
            args.python,
            "scripts/assemble_policy_relative_onpolicy_features.py",
            str(merged),
            str(args.train_feature_root),
            str(args.val_branch_root),
            str(args.val_feature_root),
            str(features),
        ],
    )
    _run(
        args,
        state,
        "gate",
        [
            args.python,
            "scripts/train_visual_tool_gate.py",
            str(features / "train"),
            str(features / "val"),
            str(gate),
            "--seed",
            str(args.seed),
            "--selection-objective",
            "proxy_utility",
            "--fit-weighting",
            "utility_risk",
            "--thresholds",
            "0.01,0.02,0.03,0.05,0.075,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.975",
        ],
    )
    gate_summary = json.loads((gate / "summary.json").read_text(encoding="utf-8"))
    oof_utility = float(gate_summary["selected"]["metrics"]["proxy_utility_sum"])
    if oof_utility <= 0.0:
        _write_state(
            state,
            status="rejected_negative_oof_utility",
            train_oof_proxy_utility_sum=oof_utility,
            three_seed_ready=False,
        )
        return
    _run(
        args,
        state,
        "evaluation",
        [
            args.python,
            "scripts/evaluate_cached_policy_relative_gate.py",
            str(features / "val"),
            str(gate),
            str(args.val_branch_root),
            str(evaluation),
        ],
    )
    bootstrap = evaluation / "task_bootstrap.json"
    _run(
        args,
        state,
        "bootstrap",
        [
            args.python,
            "scripts/bootstrap_single_policy_evaluation.py",
            str(evaluation / "traces.jsonl"),
            str(bootstrap),
            "--repetitions",
            "2000",
            "--seed",
            str(args.seed),
        ],
    )
    evaluation_summary = json.loads(
        (evaluation / "summary.json").read_text(encoding="utf-8")
    )
    bootstrap_summary = json.loads(bootstrap.read_text(encoding="utf-8"))
    utility_interval = bootstrap_summary["intervals"]["mean_utility_delta"]
    macro_interval = bootstrap_summary["intervals"]["operation_macro_f1_delta"]
    promotion = {
        "positive_train_oof_utility": oof_utility > 0.0,
        "point_promotion_passed": evaluation_summary["promotion_gate"]["passed"],
        "utility_ci_above_zero": utility_interval["ci95_low"] > 0.0,
        "macro_f1_ci_not_below_zero": macro_interval["ci95_low"] >= 0.0,
    }
    _write_state(
        state,
        status="complete",
        train_oof_proxy_utility_sum=oof_utility,
        promotion={**promotion, "passed": all(promotion.values())},
        three_seed_ready=all(promotion.values()),
        rl_ready=False,
        frozen_test_ready=False,
    )


if __name__ == "__main__":
    main()
