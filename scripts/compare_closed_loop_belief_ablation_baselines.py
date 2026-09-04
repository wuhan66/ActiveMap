#!/usr/bin/env python3
"""Compare belief ablations under the matched closed-loop baseline schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.compare_active_catalog_closed_loop import load_rows, paired_aoi_bootstrap


FIXED_PROTOCOL_KEYS = (
    "same_max_acquisitions",
    "tool_mode",
    "max_tool_calls",
    "explicit_geospatial_tool_calls",
    "selective_tool_calling",
    "map_relative_semantic_tool",
    "semantic_selective_calling",
    "inputs",
    "learned_selectors",
    "post_tool_action_adapter",
    "evaluated_policies",
    "deterministic_sample_seed",
    "tool_gate",
    "quality_gain_semantics",
    "test_assets_read",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint(protocol: dict[str, Any]) -> dict[str, Any]:
    checkpoint = protocol.get("tool_belief_checkpoint")
    if not isinstance(checkpoint, dict) or not checkpoint.get("sha256"):
        raise ValueError("hashed Tool-Belief checkpoint is required")
    return checkpoint


def verify_controlled_difference(
    reference: dict[str, Any],
    gated: dict[str, Any],
    *,
    ablation: str,
) -> dict[str, Any]:
    schema = "active-catalog-closed-loop-baselines-v1"
    if reference.get("schema_version") != schema or gated.get("schema_version") != schema:
        raise ValueError("unexpected closed-loop baseline schema")
    if reference.get("sample_count") != gated.get("sample_count"):
        raise ValueError("sample counts differ")
    if reference.get("split") != "val" or gated.get("split") != "val":
        raise ValueError("belief ablation must remain validation-only")
    left = reference["protocol"]
    right = gated["protocol"]
    mismatches = [key for key in FIXED_PROTOCOL_KEYS if left.get(key) != right.get(key)]
    if mismatches:
        raise ValueError(f"fixed protocol mismatch: {mismatches}")
    if right.get("belief_mode") != "recurrent":
        raise ValueError("gated candidate must use recurrent belief")
    if right.get("tool_mode") != "selective":
        raise ValueError("belief ablation requires selective tool mode")
    if right.get("test_assets_read") is not False:
        raise ValueError("test assets must remain unopened")

    reference_checkpoint = _checkpoint(left)
    gated_checkpoint = _checkpoint(right)
    if gated_checkpoint.get("reliability_gate") is not True:
        raise ValueError("candidate is not reliability gated")
    if ablation == "ungated":
        if left.get("belief_mode") != "recurrent":
            raise ValueError("ungated reference must use recurrent belief")
        if reference_checkpoint.get("reliability_gate") is not False:
            raise ValueError("reference is not ungated")
        if reference_checkpoint["sha256"] == gated_checkpoint["sha256"]:
            raise ValueError("gated and ungated checkpoints must differ")
        controlled_difference = "tool_belief_reliability_gate"
    elif ablation == "frozen_prior":
        if left.get("belief_mode") != "frozen_prior":
            raise ValueError("reference is not the frozen-prior ablation")
        if reference_checkpoint != gated_checkpoint:
            raise ValueError("frozen-prior ablation must reuse the gated checkpoint")
        controlled_difference = "recurrent_belief_revision"
    else:
        raise ValueError(f"unsupported ablation: {ablation}")
    return {
        "controlled_difference": controlled_difference,
        "sample_count": int(gated["sample_count"]),
        "gated_checkpoint_sha256": gated_checkpoint["sha256"],
        "reference_checkpoint_sha256": reference_checkpoint["sha256"],
        "fixed_state_sha256": right["inputs"]["states"]["sha256"],
        "fixed_episode_sha256": right["inputs"]["episodes"]["sha256"],
        "fixed_selector_sha256": right["learned_selectors"]["edit_utility"]["sha256"],
        "fixed_tool_gate_sha256": right["tool_gate"]["sha256"],
        "fixed_post_tool_adapter_sha256": right["post_tool_action_adapter"]["sha256"],
    }


def compare(
    reference_summary: dict[str, Any],
    gated_summary: dict[str, Any],
    reference_rows: dict[tuple[str, float], dict[str, Any]],
    gated_rows: dict[tuple[str, float], dict[str, Any]],
    *,
    ablation: str,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    control = verify_controlled_difference(
        reference_summary,
        gated_summary,
        ablation=ablation,
    )
    paired = paired_aoi_bootstrap(
        gated_rows,
        reference_rows,
        repetitions=repetitions,
        seed=seed,
    )
    return {
        "schema_version": "sn7-closed-loop-belief-ablation-v1",
        "ablation": ablation,
        "direction": "gated_recurrent_minus_reference",
        **control,
        "reference_metrics": reference_summary["policies"]["edit_utility"]["metrics"],
        "gated_metrics": gated_summary["policies"]["edit_utility"]["metrics"],
        "paired_aoi_bootstrap": paired,
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--ablation", choices=("ungated", "frozen_prior"), required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--gated-summary", type=Path, required=True)
    parser.add_argument("--reference-traces", type=Path, required=True)
    parser.add_argument("--gated-traces", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = compare(
        _read(args.reference_summary),
        _read(args.gated_summary),
        load_rows(args.reference_traces),
        load_rows(args.gated_traces),
        ablation=args.ablation,
        repetitions=args.repetitions,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
