#!/usr/bin/env python3
"""Export the completed one-time SN7 Step-0 frozen-test result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.export_sn7_step0_paper_tables import (
    CONTROLLER_METRICS,
    WRITEBACK_METRICS,
    _controller_rows,
    _latex,
    _load,
    _markdown,
    _write_csv,
    _writeback_rows,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_test(payload: dict[str, Any], name: str) -> None:
    split = payload.get("split", payload.get("protocol", {}).get("split"))
    test_read = payload.get(
        "test_assets_read", payload.get("protocol", {}).get("test_assets_read")
    )
    if name == "promotion":
        split = "test" if str(payload.get("claim_boundary", "")).startswith(
            "Three-seed frozen-test"
        ) else split
    if split != "test" or test_read is not True:
        raise ValueError(f"{name} is not consistent frozen-test evidence")


def _validate_ledger(
    ledger: dict[str, Any], ledger_path: Path, registry_path: Path
) -> None:
    if ledger.get("schema_version") != "activemap-frozen-test-access-v1":
        raise ValueError("unexpected frozen-test ledger schema")
    if ledger.get("status") != "complete" or int(ledger.get("returncode", -1)) != 0:
        raise ValueError("frozen-test ledger is not complete and successful")
    if ledger.get("registry_sha256") != _sha256(registry_path):
        raise ValueError("ledger registry hash differs from the supplied registry")
    command = " ".join(map(str, ledger.get("command", [])))
    if "run_sn7_step0_frozen_test_v2.sh" not in command:
        raise ValueError("ledger command is not the frozen Step-0 launcher")
    if not ledger_path.is_file():
        raise ValueError("frozen-test ledger does not exist")


def export(
    registry_path: Path,
    ledger_path: Path,
    run_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    ledger = _load(ledger_path)
    _validate_ledger(ledger, ledger_path, registry_path)
    complete_path = run_root / "COMPLETE.json"
    complete = _load(complete_path)
    if (
        complete.get("schema_version") != "sn7-step0-frozen-test-complete-v2"
        or complete.get("test_assets_read") is not True
    ):
        raise ValueError("run root is not a completed frozen-test v2 run")

    paths = {
        "controller_summary": run_root / "three_policy_summary.json",
        "benefit_vs_notool": run_root / "benefit_vs_notool.json",
        "benefit_vs_forced": run_root / "benefit_vs_forced.json",
        "promotion": run_root / "writeback_promotion.json",
    }
    payloads = {name: _load(path) for name, path in paths.items()}
    for name, payload in payloads.items():
        _assert_test(payload, name)

    seeds = sorted(int(seed) for seed in payloads["controller_summary"]["seeds"])
    if len(seeds) != 3:
        raise ValueError("frozen test requires exactly three policy seeds")
    for name in ("benefit_vs_notool", "benefit_vs_forced"):
        if sorted(int(seed) for seed in payloads[name]["model_seeds"]) != seeds:
            raise ValueError(f"{name} seed support differs from the frozen controller")

    controller = _controller_rows(payloads["controller_summary"])
    writeback = _writeback_rows(
        payloads["benefit_vs_notool"], payloads["benefit_vs_forced"]
    )
    paired = {
        "controller": payloads["controller_summary"]["comparisons"],
        "writeback": {
            "benefit_vs_notool": payloads["benefit_vs_notool"]["paired_delta"],
            "benefit_vs_forced": payloads["benefit_vs_forced"]["paired_delta"],
        },
    }

    output_dir.mkdir(parents=True)
    _write_csv(
        output_dir / "controller_table.csv",
        controller,
        CONTROLLER_METRICS,
        seed_stats=True,
    )
    _write_csv(
        output_dir / "writeback_table.csv",
        writeback,
        WRITEBACK_METRICS,
        seed_stats=False,
    )
    footer = (
        "Validation only; test assets were not read.",
        "One-time frozen test; results are reported regardless of promotion outcome.",
    )
    (output_dir / "controller_table.md").write_text(
        _markdown(
            "SN7 Step-0 Frozen-Test Controller Results",
            controller,
            CONTROLLER_METRICS,
            seed_stats=True,
        ).replace(*footer),
        encoding="utf-8",
    )
    (output_dir / "writeback_table.md").write_text(
        _markdown(
            "SN7 Frozen-Test Executable Writeback Results",
            writeback,
            WRITEBACK_METRICS,
            seed_stats=False,
        ).replace(*footer),
        encoding="utf-8",
    )
    (output_dir / "controller_table.tex").write_text(
        _latex(controller, CONTROLLER_METRICS, seed_stats=True), encoding="utf-8"
    )
    (output_dir / "writeback_table.tex").write_text(
        _latex(writeback, WRITEBACK_METRICS, seed_stats=False), encoding="utf-8"
    )
    (output_dir / "paired_intervals.json").write_text(
        json.dumps(paired, indent=2) + "\n", encoding="utf-8"
    )

    promotion_passed = bool(payloads["promotion"].get("promote"))
    if bool(complete.get("promotion_passed")) != promotion_passed:
        raise ValueError("COMPLETE and promotion artifacts disagree")
    source_paths = {
        "registry": registry_path,
        "ledger": ledger_path,
        "complete": complete_path,
        **paths,
    }
    manifest = {
        "schema_version": "sn7-step0-frozen-test-tables-v1",
        "split": "test",
        "test_assets_read": True,
        "seeds": seeds,
        "promotion_passed": promotion_passed,
        "reporting_policy": "report_regardless_of_promotion_outcome",
        "sources": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in source_paths.items()
        },
        "outputs": sorted(path.name for path in output_dir.iterdir()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            export(args.registry, args.ledger, args.run_root, args.output_dir),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
