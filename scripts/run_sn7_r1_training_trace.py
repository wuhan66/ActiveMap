#!/usr/bin/env python3
"""Create one immutable, train-only full-controller trace for reviewer R1.

R1 matches simple heuristic acquisition policies to the *deployed* ActiveMap
controller.  It must therefore include the frozen selector, selective tool
gate, recurrent tool-belief updater, and post-tool terminal adapter for one
registered seed.  A selector-only trace is deliberately not accepted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from scripts.build_train_call_rate_receipt import build_receipt


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_registered_path(
    raw_path: str, *, storage_root: Path, project_root: Path
) -> Path:
    return Path(
        raw_path.replace("${STORAGE_ROOT}", str(storage_root)).replace(
            "${PROJECT_ROOT}", str(project_root)
        )
    )


def load_frozen_controller(
    registry_path: Path,
    seed: int,
    *,
    storage_root: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Load and hash-check one full frozen controller from the registry."""
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or registry.get("test_assets_read") is not False:
        raise ValueError("R1 registry must be a non-test frozen controller registry")
    assets = registry.get("seed_artifacts", {}).get(str(seed))
    if not isinstance(assets, dict):
        raise ValueError(f"frozen registry has no controller seed {seed}")
    required = ("selector", "tool_gate", "tool_belief", "post_tool_adapter")
    result: dict[str, Any] = {}
    for name in required:
        entry = assets.get(name)
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(f"frozen registry seed {seed} lacks {name}")
        path = _resolve_registered_path(
            entry["path"], storage_root=storage_root, project_root=project_root
        )
        if not path.is_file():
            raise FileNotFoundError(f"registered {name} is missing: {path}")
        actual_sha256 = _sha256(path)
        if actual_sha256 != entry.get("sha256"):
            raise ValueError(f"registered {name} hash mismatch: {path}")
        result[name] = {"path": path, "sha256": actual_sha256}
        if name == "selector":
            result[name]["stop_margin"] = float(entry["stop_margin"])
    gate_summary = result["tool_gate"]["path"].with_name("summary.json")
    if not gate_summary.is_file():
        raise FileNotFoundError(f"frozen tool-gate summary is missing: {gate_summary}")
    result["tool_gate_summary"] = {
        "path": gate_summary,
        "sha256": _sha256(gate_summary),
    }
    return result


def evaluator_command(
    args: argparse.Namespace, controller: dict[str, Any], output_dir: Path
) -> list[str]:
    return [
        args.python,
        "scripts/evaluate_active_catalog_closed_loop_baselines.py",
        str(args.states),
        str(output_dir),
        "--episodes",
        str(args.episodes),
        "--split",
        "train",
        "--device",
        args.device,
        "--max-candidates",
        str(args.max_candidates),
        "--max-acquisitions",
        "1",
        "--bootstrap-repetitions",
        str(args.bootstrap_repetitions),
        "--seed",
        str(args.seed),
        "--learned-selector",
        f"activemap={controller['selector']['path']}",
        "--learned-selector-stop-margin",
        f"activemap={controller['selector']['stop_margin']}",
        "--policy",
        "activemap",
        "--tool-mode",
        "selective",
        "--belief-mode",
        "recurrent",
        "--tool-belief-checkpoint",
        str(controller["tool_belief"]["path"]),
        "--post-tool-action-adapter",
        str(controller["post_tool_adapter"]["path"]),
        "--tool-gate",
        str(controller["tool_gate"]["path"]),
        "--tool-gate-summary",
        str(controller["tool_gate_summary"]["path"]),
        "--tool-artifact-root",
        str(output_dir / "tool_artifacts"),
        "--tool-out-size",
        str(args.tool_out_size),
        *sum((["--asset-root-map", mapping] for mapping in args.asset_root_map), []),
    ]


def validate_trace(path: Path) -> None:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("R1 training trace is empty")
    for index, row in enumerate(rows, 1):
        if row.get("split") != "train" or row.get("policy") != "activemap":
            raise ValueError(f"R1 trace row {index} violates the train/policy contract")
        if bool(row.get("test_assets_read")):
            raise ValueError(f"R1 trace row {index} records test access")
        if int(row.get("acquisitions", -1)) not in {0, 1}:
            raise ValueError(f"R1 trace row {index} has an invalid acquisition count")


def build_summary(
    args: argparse.Namespace,
    controller: dict[str, Any],
    trace_path: Path,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    components = {
        name: {"path": str(value["path"].resolve()), "sha256": value["sha256"]}
        for name, value in controller.items()
        if name != "selector"
    }
    components["selector"] = {
        "path": str(controller["selector"]["path"].resolve()),
        "sha256": controller["selector"]["sha256"],
        "stop_margin": controller["selector"]["stop_margin"],
    }
    return {
        "schema_version": "activemap-r1-full-controller-train-trace-v2",
        "split": "train",
        "test_assets_read": False,
        "policy": "activemap",
        "registry": {"path": str(args.registry.resolve()), "sha256": _sha256(args.registry)},
        "controller_seed": args.seed,
        "controller_components": components,
        "states": {"path": str(args.states.resolve()), "sha256": _sha256(args.states)},
        "episodes": {"path": str(args.episodes.resolve()), "sha256": _sha256(args.episodes)},
        "trace": {"path": str(trace_path.resolve()), "sha256": _sha256(trace_path)},
        "call_rate_receipt": receipt,
        "protocol": {
            "max_acquisitions": 1,
            "tool_mode": "selective",
            "belief_mode": "recurrent",
            "selection_split": "train",
            "purpose": "fixed R1 matched-rate target only",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("states", type=Path)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("registry", type=Path)
    parser.add_argument("seed", type=int)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--tool-out-size", type=int, default=256)
    parser.add_argument("--bootstrap-repetitions", type=int, default=0)
    parser.add_argument("--asset-root-map", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    for path in (args.states, args.episodes, args.registry):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.max_candidates <= 0 or args.tool_out_size <= 0 or args.bootstrap_repetitions < 0:
        raise ValueError("invalid R1 launcher limits")
    controller = load_frozen_controller(
        args.registry,
        args.seed,
        storage_root=args.storage_root,
        project_root=args.project_root,
    )
    evaluation_dir = args.output_dir / "evaluation"
    subprocess.run(evaluator_command(args, controller, evaluation_dir), check=True)
    trace_path = evaluation_dir / "activemap.jsonl"
    validate_trace(trace_path)
    receipt = build_receipt(trace_path)
    (args.output_dir / "active_train_rate_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    summary = build_summary(args, controller, trace_path, receipt)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
