#!/usr/bin/env python3
"""Evaluate an architecture-matched learned-defer baseline on frozen SN7 val.

This extension reuses the immutable ActiveMap R1/R2 traces and evaluates only
the generic selector. It cannot select the test split and refuses any drift in
the validation inputs, frozen controller components, or reference traces.
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

POLICY = "learned_defer"
SHARED_COMPONENTS = (
    "selector",
    "tool_gate",
    "tool_gate_summary",
    "tool_belief",
    "post_tool_adapter",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _verified_path(receipt: dict[str, Any], *, description: str) -> Path:
    path = Path(str(receipt["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    if _sha256(path) != receipt["sha256"]:
        raise ValueError(f"{description} hash mismatch: {path}")
    return path


def validate_base_receipt(
    receipt: dict[str, Any],
    *,
    seed: int,
    val_states: Path,
    val_episodes: Path,
    registry: Path,
    controller: dict[str, Any],
) -> dict[str, Path]:
    if receipt.get("split") != "val" or receipt.get("test_assets_read") is not False:
        raise ValueError("base R1/R2 receipt is not validation-only")
    if int(receipt.get("controller_seed", -1)) != seed:
        raise ValueError("base R1/R2 receipt seed mismatch")
    expected_inputs = {
        "val_states": val_states,
        "val_episodes": val_episodes,
    }
    for key, path in expected_inputs.items():
        if not path.is_file() or _sha256(path) != receipt["inputs"][key]["sha256"]:
            raise ValueError(f"base R1/R2 {key} hash mismatch")
    if not registry.is_file() or _sha256(registry) != receipt["registry"]["sha256"]:
        raise ValueError("base R1/R2 registry hash mismatch")
    for key in SHARED_COMPONENTS:
        current = controller[key]
        expected = receipt["controller_components"][key]
        if current["sha256"] != expected["sha256"]:
            raise ValueError(f"frozen controller component drift: {key}")
        _verified_path(expected, description=f"base controller component {key}")
    return {
        "active_natural": _verified_path(
            receipt["natural_validation_traces"]["activemap"],
            description="base ActiveMap natural trace",
        ),
        "active_edit_only": _verified_path(
            receipt["r2_filtered_traces"]["activemap"],
            description="base ActiveMap EDIT-only trace",
        ),
        "edit_manifest": _verified_path(
            receipt["inputs"]["edit_manifest"],
            description="base EDIT-only manifest",
        ),
    }


def validate_evaluation_summary(summary: dict[str, Any], checkpoint: Path) -> None:
    protocol = summary.get("protocol", {})
    learned = protocol.get("learned_selectors", {}).get(POLICY)
    if summary.get("split") != "val" or protocol.get("test_assets_read") is not False:
        raise ValueError("learned-defer evaluation is not validation-only")
    if protocol.get("evaluated_policies") != [POLICY]:
        raise ValueError("learned-defer evaluation contains unexpected policies")
    if protocol.get("same_max_acquisitions") != 1:
        raise ValueError("learned-defer evaluation is not one-acquisition matched")
    if protocol.get("tool_mode") != "selective" or protocol.get("belief_mode") != "recurrent":
        raise ValueError("learned-defer controller protocol drift")
    if not isinstance(learned, dict):
        raise ValueError("learned-defer selector receipt is missing")
    if learned.get("condition_on_hypothesis") is not False:
        raise ValueError("learned-defer checkpoint is not hypothesis-agnostic")
    if learned.get("stop_margin_source") != "checkpoint":
        raise ValueError("learned-defer STOP margin was not frozen in the checkpoint")
    if float(learned["checkpoint_stop_margin"]) != float(learned["effective_stop_margin"]):
        raise ValueError("learned-defer effective STOP margin differs from checkpoint")
    if learned.get("sha256") != _sha256(checkpoint):
        raise ValueError("learned-defer checkpoint hash mismatch in evaluator receipt")


def _evaluation_command(
    args: argparse.Namespace,
    controller: dict[str, Any],
    *,
    output: Path,
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
        POLICY,
        "--learned-selector",
        f"{POLICY}={args.generic_selector}",
        "--tool-mode",
        "selective",
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
        "--tool-gate",
        str(controller["tool_gate"]["path"]),
        "--tool-gate-summary",
        str(controller["tool_gate_summary"]["path"]),
    ]
    for mapping in args.asset_root_map:
        command.extend(["--asset-root-map", mapping])
    return command


def _comparison_command(
    args: argparse.Namespace,
    *,
    active: Path,
    learned_defer: Path,
    output: Path,
    manifest: Path | None,
) -> list[str]:
    command = [
        args.python,
        "scripts/compare_active_catalog_closed_loop.py",
        str(output),
        "--candidate",
        "activemap",
        "--records",
        f"activemap={active}",
        "--records",
        f"{POLICY}={learned_defer}",
        "--repetitions",
        str(args.per_seed_bootstrap_repetitions),
        "--seed",
        str(args.bootstrap_seed),
    ]
    if manifest is not None:
        command.extend(["--manifest", str(manifest)])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("val_states", type=Path)
    parser.add_argument("val_episodes", type=Path)
    parser.add_argument("registry", type=Path)
    parser.add_argument("generic_selector", type=Path)
    parser.add_argument("seed", type=int)
    parser.add_argument("base_r1_r2_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--tool-out-size", type=int, default=256)
    parser.add_argument("--per-seed-bootstrap-repetitions", type=int, default=1)
    parser.add_argument("--bootstrap-seed", type=int, default=20260830)
    parser.add_argument("--asset-root-map", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.per_seed_bootstrap_repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    if not args.generic_selector.is_file():
        raise FileNotFoundError(args.generic_selector)
    base_receipt_path = args.base_r1_r2_root / f"seed{args.seed}" / "COMPLETE.json"
    if not base_receipt_path.is_file():
        raise FileNotFoundError(base_receipt_path)
    controller = load_frozen_controller(
        args.registry,
        args.seed,
        storage_root=args.storage_root,
        project_root=args.project_root,
    )
    base_receipt = json.loads(base_receipt_path.read_text(encoding="utf-8"))
    references = validate_base_receipt(
        base_receipt,
        seed=args.seed,
        val_states=args.val_states,
        val_episodes=args.val_episodes,
        registry=args.registry,
        controller=controller,
    )

    args.output_dir.mkdir(parents=True)
    natural_output = args.output_dir / "natural_validation" / POLICY
    _run(_evaluation_command(args, controller, output=natural_output), cwd=args.project_root)
    evaluation_summary_path = natural_output / "summary.json"
    evaluation_summary = json.loads(evaluation_summary_path.read_text(encoding="utf-8"))
    validate_evaluation_summary(evaluation_summary, args.generic_selector)
    natural_trace = natural_output / f"{POLICY}.jsonl"
    if not natural_trace.is_file():
        raise FileNotFoundError(natural_trace)

    r2_trace = args.output_dir / "r2_edit_only" / f"{POLICY}.jsonl"
    _run(
        [
            args.python,
            "scripts/filter_closed_loop_trace_to_manifest.py",
            str(natural_trace),
            str(references["edit_manifest"]),
            str(r2_trace),
        ],
        cwd=args.project_root,
    )
    natural_comparison = args.output_dir / "natural_validation" / "paired_comparison.json"
    _run(
        _comparison_command(
            args,
            active=references["active_natural"],
            learned_defer=natural_trace,
            output=natural_comparison,
            manifest=None,
        ),
        cwd=args.project_root,
    )
    r2_comparison = args.output_dir / "r2_edit_only" / "paired_comparison.json"
    _run(
        _comparison_command(
            args,
            active=references["active_edit_only"],
            learned_defer=r2_trace,
            output=r2_comparison,
            manifest=references["edit_manifest"],
        ),
        cwd=args.project_root,
    )

    receipt = {
        "schema_version": "activemap-sn7-learned-defer-extension-seed-v1",
        "split": "val",
        "test_assets_read": False,
        "controller_seed": args.seed,
        "comparison_contract": {
            "candidate": "activemap",
            "reference": POLICY,
            "only_selector_conditioning_differs": True,
            "same_architecture_training_budget_and_seed": True,
            "same_recurrent_belief_tool_gate_safe_commit": True,
            "max_acquisitions": 1,
        },
        "base_receipt": {
            "path": str(base_receipt_path.resolve()),
            "sha256": _sha256(base_receipt_path),
        },
        "inputs": {
            "val_states": {
                "path": str(args.val_states.resolve()),
                "sha256": _sha256(args.val_states),
            },
            "val_episodes": {
                "path": str(args.val_episodes.resolve()),
                "sha256": _sha256(args.val_episodes),
            },
            "edit_manifest": {
                "path": str(references["edit_manifest"].resolve()),
                "sha256": _sha256(references["edit_manifest"]),
            },
        },
        "selectors": {
            "activemap": base_receipt["controller_components"]["selector"],
            POLICY: evaluation_summary["protocol"]["learned_selectors"][POLICY],
        },
        "shared_controller_components": {
            key: base_receipt["controller_components"][key]
            for key in SHARED_COMPONENTS
            if key != "selector"
        },
        "natural_validation_traces": {
            "activemap": base_receipt["natural_validation_traces"]["activemap"],
            POLICY: {"path": str(natural_trace.resolve()), "sha256": _sha256(natural_trace)},
        },
        "r2_filtered_traces": {
            "activemap": base_receipt["r2_filtered_traces"]["activemap"],
            POLICY: {"path": str(r2_trace.resolve()), "sha256": _sha256(r2_trace)},
        },
        "comparisons": {
            "r1_natural": {
                "path": str(natural_comparison.resolve()),
                "sha256": _sha256(natural_comparison),
            },
            "r2_edit_only": {
                "path": str(r2_comparison.resolve()),
                "sha256": _sha256(r2_comparison),
            },
        },
    }
    complete = args.output_dir / "COMPLETE.json"
    complete.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
