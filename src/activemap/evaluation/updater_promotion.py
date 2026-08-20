"""Validation-gated promotion for hierarchical updater experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def select_promoted_checkpoint(
    calibrations: list[dict[str, Any]],
    *,
    baseline_macro_f1: float,
    baseline_delete_f1: float,
    max_false_edit: float,
) -> dict[str, Any] | None:
    """Return the strongest candidate satisfying every preregistered validation gate."""

    eligible = []
    for calibration in calibrations:
        selected = calibration["selected"]
        if (
            calibration.get("constraint_satisfied", False)
            and float(selected["false_edit_rate"]) <= max_false_edit
            and float(selected["macro_f1"]) > baseline_macro_f1
            and float(selected["delete_f1"]) > baseline_delete_f1
        ):
            eligible.append(calibration)
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda item: (
            float(item["selected"]["macro_f1"]),
            float(item["selected"]["delete_f1"]),
        ),
    )


def finalize_hierarchical_updater(
    run_dir: Path,
    samples_path: Path,
    *,
    device: str = "auto",
    batch_size: int = 64,
    max_false_edit: float = 0.05,
    grid_steps: int = 33,
    baseline_macro_f1: float = 0.789307,
    baseline_delete_f1: float = 0.365979,
    bootstrap_iterations: int = 1000,
    evaluate_test: bool = True,
) -> dict[str, Any]:
    """Calibrate validation checkpoints and test exactly once after promotion."""

    from activemap.evaluation.updater_calibration import calibrate_updater_hierarchy

    calibration_dir = run_dir / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    calibrations = []
    for checkpoint_name in ("best_quality", "best_safety", "best_val_loss"):
        checkpoint_path = run_dir / f"{checkpoint_name}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        calibration = calibrate_updater_hierarchy(
            checkpoint_path,
            samples_path,
            calibration_dir / f"{checkpoint_name}_val.json",
            split="val",
            device=device,
            batch_size=batch_size,
            max_false_edit=max_false_edit,
            grid_steps=grid_steps,
        )
        calibration["checkpoint_name"] = checkpoint_name
        calibrations.append(calibration)

    promoted = select_promoted_checkpoint(
        calibrations,
        baseline_macro_f1=baseline_macro_f1,
        baseline_delete_f1=baseline_delete_f1,
        max_false_edit=max_false_edit,
    )
    status = "rejected"
    if promoted is not None:
        status = "promoted" if evaluate_test else "promoted_validation_only"
    decision: dict[str, Any] = {
        "status": status,
        "validation_only_selection": True,
        "test_enabled": evaluate_test,
        "max_false_edit": max_false_edit,
        "baseline_macro_f1": baseline_macro_f1,
        "baseline_delete_f1": baseline_delete_f1,
        "candidates": [
            {
                "checkpoint_name": item["checkpoint_name"],
                "selected": item["selected"],
                "constraint_satisfied": item["constraint_satisfied"],
            }
            for item in calibrations
        ],
        "promoted_checkpoint": None,
        "validation_evaluation": None,
        "test_evaluation": None,
    }
    if promoted is not None:
        checkpoint_name = str(promoted["checkpoint_name"])
        decision["promoted_checkpoint"] = checkpoint_name
        from activemap.evaluation.updater import evaluate_updater_checkpoint

        selected = promoted["selected"]
        decision["validation_evaluation"] = evaluate_updater_checkpoint(
            run_dir / f"{checkpoint_name}.pt",
            samples_path,
            run_dir / "evaluation" / "val_promoted",
            split="val",
            device=device,
            batch_size=batch_size,
            bootstrap_iterations=bootstrap_iterations,
            presence_threshold=float(selected["presence_threshold"]),
            change_threshold=float(selected["change_threshold"]),
        )
    if promoted is not None and evaluate_test:
        checkpoint_name = str(promoted["checkpoint_name"])
        selected = promoted["selected"]
        test_output_dir = run_dir / "evaluation" / "test_promoted"
        test_summary = evaluate_updater_checkpoint(
            run_dir / f"{checkpoint_name}.pt",
            samples_path,
            test_output_dir,
            split="test",
            device=device,
            batch_size=batch_size,
            bootstrap_iterations=bootstrap_iterations,
            presence_threshold=float(selected["presence_threshold"]),
            change_threshold=float(selected["change_threshold"]),
        )
        decision["test_evaluation"] = test_summary

    decision_path = run_dir / "promotion_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    return decision
