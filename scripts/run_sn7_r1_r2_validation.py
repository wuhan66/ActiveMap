#!/usr/bin/env python3
"""Run one full-controller R1/R2 reviewer-validation seed without test access.

The script intentionally has no test split option. It first derives a
train-only call-rate receipt from the hash-bound deployed controller, then
evaluates fixed ActiveMap/forced/STOP traces and rate-matched observable
heuristics on the identical validation state bundle. Finally it filters every
trace to the immutable R2 EDIT-only manifest and produces paired comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.run_sn7_r1_training_trace import load_frozen_controller


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _baseline_command(
    args: argparse.Namespace,
    controller: dict[str, Any],
    *,
    output: Path,
    policy: str,
    tool_mode: str,
) -> list[str]:
    command = [
        args.python,
        "scripts/evaluate_active_catalog_closed_loop_baselines.py",
        str(args.val_states),
        str(output),
        "--episodes",
        str(args.val_episodes),
        "--split",
        "val",
        "--device",
        args.device,
        "--max-candidates",
        str(args.max_candidates),
        "--max-acquisitions",
        "1",
        "--bootstrap-repetitions",
        "0",
        "--seed",
        str(args.seed),
        "--policy",
        policy,
    ]
    if policy != "always_stop":
        command.extend(
            [
                "--learned-selector",
                f"activemap={controller['selector']['path']}",
                "--learned-selector-stop-margin",
                f"activemap={controller['selector']['stop_margin']}",
            ]
        )
    if tool_mode != "none":
        command.extend(
            [
                "--tool-mode",
                tool_mode,
                "--belief-mode",
                "recurrent",
                "--tool-belief-checkpoint",
                str(controller["tool_belief"]["path"]),
                "--post-tool-action-adapter",
                str(controller["post_tool_adapter"]["path"]),
                "--tool-artifact-root",
                str(output / "tool_artifacts"),
                "--tool-out-size",
                str(args.tool_out_size),
            ]
        )
        if tool_mode == "selective":
            command.extend(
                [
                    "--tool-gate",
                    str(controller["tool_gate"]["path"]),
                    "--tool-gate-summary",
                    str(controller["tool_gate_summary"]["path"]),
                ]
            )
    for mapping in args.asset_root_map:
        command.extend(["--asset-root-map", mapping])
    return command


def _rate_matched_command(
    args: argparse.Namespace,
    *,
    output: Path,
    receipt: Path,
    trace: Path,
    bootstrap_repetitions: int,
) -> list[str]:
    return [
        args.python,
        "scripts/evaluate_rate_matched_baselines.py",
        str(args.train_states),
        str(args.val_states),
        str(output),
        "--target-rate-receipt",
        str(receipt),
        "--target-train-trace",
        str(trace),
        "--bootstrap-repetitions",
        str(bootstrap_repetitions),
        "--bootstrap-seed",
        str(args.bootstrap_seed),
        "--max-candidates",
        str(args.max_candidates),
    ]


def _filter_command(trace: Path, manifest: Path, output: Path, python: str) -> list[str]:
    return [
        python,
        "scripts/filter_closed_loop_trace_to_manifest.py",
        str(trace),
        str(manifest),
        str(output),
    ]


def _comparison_command(
    args: argparse.Namespace,
    *,
    output: Path,
    records: dict[str, Path],
    manifest: Path | None,
    bootstrap_repetitions: int,
) -> list[str]:
    command = [
        args.python,
        "scripts/compare_active_catalog_closed_loop.py",
        str(output),
        "--candidate",
        "activemap",
        "--repetitions",
        str(bootstrap_repetitions),
        "--seed",
        str(args.bootstrap_seed),
    ]
    if manifest is not None:
        command.extend(["--manifest", str(manifest)])
    for label, path in records.items():
        command.extend(["--records", f"{label}={path}"])
    return command


def _required_trace(root: Path, policy: str) -> Path:
    path = root / f"{policy}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_states", type=Path)
    parser.add_argument("train_episodes", type=Path)
    parser.add_argument("val_states", type=Path)
    parser.add_argument("val_episodes", type=Path)
    parser.add_argument("registry", type=Path)
    parser.add_argument("seed", type=int)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--tool-out-size", type=int, default=256)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument(
        "--per-seed-bootstrap-repetitions",
        type=int,
        default=1,
        help="Intermediate receipt only; final three-seed aggregation owns inference.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    parser.add_argument("--asset-root-map", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    for path in (
        args.train_states,
        args.train_episodes,
        args.val_states,
        args.val_episodes,
        args.registry,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (
        args.bootstrap_repetitions <= 0
        or args.per_seed_bootstrap_repetitions <= 0
        or args.max_candidates <= 0
        or args.tool_out_size <= 0
    ):
        raise ValueError("invalid R1/R2 evaluation limit")
    controller = load_frozen_controller(
        args.registry,
        args.seed,
        storage_root=args.storage_root,
        project_root=args.project_root,
    )
    args.output_dir.mkdir(parents=True)
    edit_manifest = args.output_dir / "r2_edit_only_manifest.json"
    _run(
        [
            args.python,
            "scripts/build_edit_only_validation_manifest.py",
            str(args.val_states),
            str(edit_manifest),
        ],
        cwd=args.project_root,
    )
    train_root = args.output_dir / "r1_train"
    train_command = [
        args.python,
        "scripts/run_sn7_r1_training_trace.py",
        str(args.train_states),
        str(args.train_episodes),
        str(args.registry),
        str(args.seed),
        str(train_root),
        "--storage-root",
        str(args.storage_root),
        "--project-root",
        str(args.project_root),
        "--device",
        args.device,
        "--max-candidates",
        str(args.max_candidates),
        "--tool-out-size",
        str(args.tool_out_size),
    ]
    for mapping in args.asset_root_map:
        train_command.extend(["--asset-root-map", mapping])
    _run(train_command, cwd=args.project_root)

    raw_root = args.output_dir / "natural_validation"
    variants = {
        "activemap": ("activemap", "selective"),
        "forced": ("activemap", "forced"),
        "always_stop": ("always_stop", "none"),
    }
    raw_traces: dict[str, Path] = {}
    for label, (policy, tool_mode) in variants.items():
        output = raw_root / label
        _run(
            _baseline_command(
                args, controller, output=output, policy=policy, tool_mode=tool_mode
            ),
            cwd=args.project_root,
        )
        raw_traces[label] = _required_trace(output, policy)
    rate_root = raw_root / "rate_matched"
    _run(
        _rate_matched_command(
            args,
            output=rate_root,
            receipt=train_root / "active_train_rate_receipt.json",
            trace=train_root / "evaluation" / "activemap.jsonl",
            bootstrap_repetitions=args.per_seed_bootstrap_repetitions,
        ),
        cwd=args.project_root,
    )
    for mode in json.loads((rate_root / "summary.json").read_text(encoding="utf-8"))["policies"]:
        raw_traces[f"rate_matched_{mode}"] = rate_root / f"rate_matched_{mode}.jsonl"

    _run(
        _comparison_command(
            args,
            output=raw_root / "paired_comparison.json",
            records=raw_traces,
            manifest=None,
            bootstrap_repetitions=args.per_seed_bootstrap_repetitions,
        ),
        cwd=args.project_root,
    )

    r2_root = args.output_dir / "r2_edit_only"
    filtered: dict[str, Path] = {}
    for label, trace in raw_traces.items():
        target = r2_root / f"{label}.jsonl"
        _run(_filter_command(trace, edit_manifest, target, args.python), cwd=args.project_root)
        filtered[label] = target
    _run(
        _comparison_command(
            args,
            output=r2_root / "paired_comparison.json",
            records=filtered,
            manifest=edit_manifest,
            bootstrap_repetitions=args.per_seed_bootstrap_repetitions,
        ),
        cwd=args.project_root,
    )
    summary = {
        "schema_version": "activemap-r1-r2-validation-seed-v1",
        "split": "val",
        "test_assets_read": False,
        "controller_seed": args.seed,
        "final_aggregate_bootstrap_repetitions": args.bootstrap_repetitions,
        "per_seed_bootstrap_repetitions": args.per_seed_bootstrap_repetitions,
        "registry": {"path": str(args.registry.resolve()), "sha256": _sha256(args.registry)},
        "inputs": {
            "train_states": {
                "path": str(args.train_states.resolve()),
                "sha256": _sha256(args.train_states),
            },
            "train_episodes": {
                "path": str(args.train_episodes.resolve()),
                "sha256": _sha256(args.train_episodes),
            },
            "val_states": {
                "path": str(args.val_states.resolve()),
                "sha256": _sha256(args.val_states),
            },
            "val_episodes": {
                "path": str(args.val_episodes.resolve()),
                "sha256": _sha256(args.val_episodes),
            },
            "edit_manifest": {
                "path": str(edit_manifest.resolve()),
                "sha256": _sha256(edit_manifest),
            },
        },
        "controller_components": {
            key: {"path": str(value["path"].resolve()), "sha256": value["sha256"]}
            for key, value in controller.items()
        },
        "natural_validation_traces": {
            label: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for label, path in raw_traces.items()
        },
        "r1_natural_comparison": {
            "path": str((raw_root / "paired_comparison.json").resolve()),
            "sha256": _sha256(raw_root / "paired_comparison.json"),
        },
        "r2_filtered_traces": {
            label: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for label, path in filtered.items()
        },
        "r2_comparison": {
            "path": str((r2_root / "paired_comparison.json").resolve()),
            "sha256": _sha256(r2_root / "paired_comparison.json"),
        },
    }
    (args.output_dir / "COMPLETE.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
