from scripts.figures.build_reviewer_closure_evidence import (
    build_bundle,
    render_markdown,
    validate_receipts,
)


def _interval(value: float) -> dict:
    return {
        "observed_delta": value,
        "ci95_low": value - 0.01,
        "ci95_high": value + 0.01,
    }


def _r1_r2() -> dict:
    variants = [
        "activemap",
        "always_stop",
        "forced",
        "rate_matched_random",
        "rate_matched_uncertainty",
        "rate_matched_low_confidence",
        "rate_matched_clear_per_cost",
        "rate_matched_cheap_positive",
        "rate_matched_minimum_entropy",
    ]
    results = {}
    for analysis in ("r1_natural", "r2_edit_only"):
        results[analysis] = {}
        for index, reference in enumerate(variants[1:]):
            value = -0.02 if analysis == "r2_edit_only" else 0.02
            if reference == "forced":
                value = -0.1
            results[analysis][f"activemap_minus_{reference}"] = {
                "controller_seed_count": 3,
                "intervals": {
                    "safety_utility": _interval(value),
                    "balanced_utility": _interval(value),
                    "mean_cost": _interval(value),
                    "marker": _interval(float(index)),
                },
            }
    return {
        "schema_version": "activemap-r1-r2-validation-three-seed-v1",
        "split": "val",
        "test_assets_read": False,
        "variants": variants,
        "results": results,
    }


def _causal() -> dict:
    metrics = {
        metric: _interval(value)
        for metric, value in {
            "map_quality_after": 0.03,
            "balanced_utility": 0.02,
            "false_edit_rate": -0.02,
            "missed_edit_rate": 0.0,
        }.items()
    }
    return {
        "schema_version": "activemap-selection-safe-commit-2x2-v1",
        "split": "val",
        "test_assets_read": False,
        "seed_count": 3,
        "aoi_count": 4,
        "cells": {
            f"{selection}__{commit}": {
                metric: {"mean": value, "seed_std": 0.0}
                for metric, value in {
                    "map_quality_after": 0.5,
                    "balanced_utility": 0.1,
                    "false_edit_rate": 0.02,
                    "missed_edit_rate": 0.2,
                    "commit_rate": 0.3,
                    "tool_call_rate": 0.01,
                    "mean_cost": 0.4,
                }.items()
            }
            for selection in ("notool", "benefit")
            for commit in ("always_commit", "safe_commit")
        },
        "paired_comparisons": {
            "selection_gain_always_commit": metrics,
            "selection_gain_safe_commit": metrics,
            "safe_commit_gain_no_extra": metrics,
            "safe_commit_gain_learned": metrics,
        },
    }


def _cost() -> dict:
    def metrics(tool_calls: float, total: float) -> dict:
        values = {
            "shared_preacquisition_perception_ms": 100.0,
            "incremental_tool_calls": tool_calls,
            "incremental_total_budget": total,
        }
        return {name: {"mean": value, "seed_std": 0.0} for name, value in values.items()}

    return {
        "schema_version": "activemap-active-catalog-cost-accounting-v1",
        "split": "val",
        "test_assets_read": False,
        "policies": {
            "notool": metrics(0.0, 0.2),
            "benefit": metrics(0.1, 0.4),
            "forced": metrics(0.5, 0.8),
        },
    }


def _operations() -> dict:
    operations = {}
    for operation in ("KEEP", "ADD", "DELETE", "RESHAPE"):
        value = 0.01 if operation == "KEEP" else 0.0
        operations[operation] = {
            "record_count": 30,
            "seed_aoi_block_count": 6,
            "candidate_minus_baseline": {
                "raster_iou": {
                    "observed_delta": value,
                    "ci95_low": value,
                    "ci95_high": value,
                }
            },
        }
    return {
        "schema_version": "agent-writeback-operation-stratified-v1",
        "split": "val",
        "test_assets_read": False,
        "seed_count": 3,
        "operations": operations,
    }


def _learned_defer() -> dict:
    return {
        "schema_version": "activemap-sn7-learned-defer-extension-three-seed-v1",
        "split": "val",
        "test_assets_read": False,
        "seeds": [20260730, 20260731, 20260801],
        "results": {
            analysis: {
                "activemap_minus_learned_defer": {
                    "controller_seed_count": 3,
                    "intervals": {
                        "safety_utility": _interval(0.001),
                        "mean_cost": _interval(0.0),
                    },
                }
            }
            for analysis in ("r1_natural", "r2_edit_only")
        },
    }


def test_bundle_derives_cost_and_scope_claims(tmp_path):
    payloads = {
        "r1_r2": _r1_r2(),
        "causal_2x2": _causal(),
        "cost": _cost(),
        "operations": _operations(),
    }
    paths = {}
    for name, payload in payloads.items():
        path = tmp_path / f"{name}.json"
        import json

        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path

    bundle = build_bundle(paths)

    assert bundle["derived"]["incremental_tool_call_reduction_vs_forced"] == 0.8
    assert bundle["derived"]["incremental_budget_reduction_vs_forced"] == 0.5
    assert bundle["derived"]["shared_perception_identical_across_policies"] is True
    assert bundle["derived"]["nonkeep_map_quality_effect_is_zero"] is True
    assert len(bundle["selection_safe_commit_cells"]) == 4
    assert "does not establish non-KEEP recovery" in render_markdown(bundle)


def test_validation_rejects_operation_slices_with_test_access():
    operations = _operations()
    operations["test_assets_read"] = True

    import pytest

    with pytest.raises(ValueError, match="validation-only"):
        validate_receipts(_r1_r2(), _causal(), _cost(), operations)


def test_bundle_includes_architecture_matched_learned_defer(tmp_path):
    payloads = {
        "r1_r2": _r1_r2(),
        "causal_2x2": _causal(),
        "cost": _cost(),
        "operations": _operations(),
        "learned_defer": _learned_defer(),
    }
    paths = {}
    for name, payload in payloads.items():
        path = tmp_path / f"{name}.json"
        import json

        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path

    bundle = build_bundle(paths)
    rows = [
        row
        for row in bundle["matched_baselines"]
        if row["comparison"] == "learned_defer"
    ]
    assert len(rows) == 4
    assert "confidence intervals" in render_markdown(bundle)
