"""Train, evaluate, and aggregate every registered ablation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from activemap.config import deep_merge, load_yaml


def _resolve_from(path: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (path.parent / candidate).resolve()


def _evaluate_selector_run(
    run_dir: Path,
    samples_path: Path,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    from activemap.evaluation.selector import evaluate_score_policy
    from activemap.inference import SelectorPredictor
    from activemap.training.data import load_selector_samples

    predictor = SelectorPredictor(run_dir / "best.pt")
    samples = load_selector_samples(samples_path, split=str(evaluation.get("split", "test")))
    budgets = [float(value) for value in evaluation.get("budgets", [1, 2, 4, 8])]
    rows = [
        evaluate_score_policy(
            samples,
            method="learned",
            budget=budget,
            score_fn=predictor.action_scores,
            allow_stop=predictor.ablation.allow_stop,
        ).as_dict()
        for budget in budgets
    ]
    (run_dir / "test_metrics.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(run_dir / "test_metrics.csv", index=False)
    primary_budget = float(evaluation.get("primary_budget", budgets[-1]))
    primary = min(rows, key=lambda row: abs(float(row["budget"]) - primary_budget))
    return {
        "test_budget": float(primary["budget"]),
        "test_mean_utility": float(primary["mean_utility"]),
        "test_mean_regret": float(primary["mean_regret"]),
        "test_top1_accuracy": float(primary["top1_accuracy"]),
        "test_stop_accuracy": float(primary["stop_accuracy"]),
        "test_mean_cost": float(primary["mean_cost"]),
    }


def _evaluate_updater_run(
    run_dir: Path,
    samples_path: Path,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    from activemap.evaluation.updater import evaluate_updater_checkpoint

    summary = evaluate_updater_checkpoint(
        run_dir / "best.pt",
        samples_path,
        run_dir / "evaluation",
        split=str(evaluation.get("split", "test")),
        device=str(evaluation.get("device", "auto")),
        batch_size=int(evaluation.get("batch_size", 32)),
        num_workers=int(evaluation.get("num_workers", 0)),
        commit_threshold=float(evaluation.get("commit_threshold", 0.0)),
        bootstrap_iterations=0,
    )
    scalar_names = (
        "edit_accuracy",
        "macro_f1",
        "stable_f1",
        "update_f1",
        "false_edit_rate",
        "missed_update_rate",
        "mean_raster_iou",
        "ece",
        "aurc",
    )
    return {
        f"test_{name}": float(summary[name])
        for name in scalar_names
        if summary.get(name) is not None
    }


def aggregate_ablation_runs(
    runs: list[dict[str, Any]],
    *,
    reference: str = "full",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Return mean/std tables plus seed-paired deltas against the full model."""

    frame = pd.DataFrame(runs)
    numeric_columns = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column != "seed"
    ]
    aggregate = frame.groupby("name", sort=False)[numeric_columns].agg(["mean", "std"])
    aggregate.columns = [f"{name}_{statistic}" for name, statistic in aggregate.columns]
    aggregate = aggregate.reset_index().fillna(0.0)

    reference_rows = frame[frame["name"] == reference]
    paired: list[dict[str, Any]] = []
    if not reference_rows.empty:
        for name in frame["name"].drop_duplicates():
            if name == reference:
                continue
            comparison = frame[frame["name"] == name].merge(
                reference_rows,
                on="seed",
                suffixes=("_experiment", "_reference"),
            )
            for metric in numeric_columns:
                left = f"{metric}_experiment"
                right = f"{metric}_reference"
                if left not in comparison or right not in comparison:
                    continue
                difference = comparison[left] - comparison[right]
                paired.append(
                    {
                        "name": name,
                        "reference": reference,
                        "metric": metric,
                        "seed_count": len(difference),
                        "mean_delta": float(difference.mean()),
                        "std_delta": float(difference.std(ddof=1))
                        if len(difference) > 1
                        else 0.0,
                    }
                )
    return aggregate, paired


def _write_markdown_summary(
    path: Path,
    task: str,
    aggregate: pd.DataFrame,
    seeds: list[int],
) -> None:
    preferred = (
        ["test_mean_utility_mean", "test_mean_regret_mean", "test_top1_accuracy_mean"]
        if task == "selector"
        else ["test_update_f1_mean", "test_stable_f1_mean", "test_mean_raster_iou_mean"]
    )
    columns = ["name", *[column for column in preferred if column in aggregate.columns]]
    display = aggregate[columns].copy()
    for column in columns[1:]:
        display[column] = display[column].map(lambda value: f"{float(value):.4f}")
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    table = [
        "| "
        + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
        + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
        *[
            "| "
            + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
            + " |"
            for row in rows
        ],
    ]
    lines = [
        f"# {task.title()} ablation summary",
        "",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
        *table,
        "",
        "Generated by `activemap run-ablations`; do not edit by hand.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_ablation_matrix(
    matrix_path: Path,
    *,
    output_root: Path | None = None,
) -> list[dict[str, Any]]:
    matrix = load_yaml(matrix_path)
    task = str(matrix.get("task", "selector"))
    if task not in {"selector", "updater"}:
        raise ValueError("matrix task must be selector or updater")
    base_path = _resolve_from(matrix_path, str(matrix["base_config"]))
    base = load_yaml(base_path)
    data_path = _resolve_from(base_path, str(base["data"]["samples"]))
    base["data"]["samples"] = str(data_path)
    root = output_root or _resolve_from(matrix_path, str(matrix.get("output_root", "outputs")))
    root.mkdir(parents=True, exist_ok=True)

    seeds = [int(seed) for seed in matrix.get("seeds", [20260710])]
    experiments = matrix.get("experiments", [])
    if not experiments:
        raise ValueError("ablation matrix contains no experiments")
    evaluation = dict(matrix.get("evaluation", {}))
    evaluate = bool(evaluation.get("enabled", True))
    summaries: list[dict[str, Any]] = []

    for experiment in experiments:
        name = str(experiment["name"])
        overrides = dict(experiment.get("overrides", {}))
        for seed in seeds:
            run_name = f"{name}__seed{seed}"
            run_dir = root / run_name
            resolved = deep_merge(base, overrides)
            resolved["seed"] = seed
            resolved["output_dir"] = str(run_dir)
            if task == "selector":
                resolved.setdefault("ablation", {})["name"] = name
            run_dir.mkdir(parents=True, exist_ok=True)
            resolved_path = run_dir / "resolved_config.yaml"
            resolved_path.write_text(
                yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
            )

            if task == "selector":
                from activemap.training.selector import train_selector

                training_summary = train_selector(resolved_path, output_override=run_dir)
                summary: dict[str, Any] = {
                    "name": name,
                    "seed": seed,
                    "best_val_regret": training_summary["best_val_regret"],
                    "parameter_count": training_summary["parameter_count"],
                    "epochs_completed": training_summary["epochs_completed"],
                    "output_dir": str(run_dir),
                }
                if evaluate:
                    summary.update(_evaluate_selector_run(run_dir, data_path, evaluation))
            else:
                from activemap.training.updater import train_updater

                training_summary = train_updater(resolved_path, output_override=run_dir)
                summary = {
                    "name": name,
                    "seed": seed,
                    "best_val_score": training_summary["best_val_score"],
                    "parameter_count": training_summary["parameter_count"],
                    "epochs_completed": training_summary["epochs_completed"],
                    "output_dir": str(run_dir),
                }
                if evaluate:
                    summary.update(_evaluate_updater_run(run_dir, data_path, evaluation))
            summaries.append(summary)

    aggregate, paired = aggregate_ablation_runs(
        summaries, reference=str(matrix.get("reference", "full"))
    )
    (root / "runs.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(summaries).to_csv(root / "runs.csv", index=False)
    aggregate.to_csv(root / "summary.csv", index=False)
    (root / "paired_deltas.json").write_text(
        json.dumps(paired, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown_summary(root / "README.md", task, aggregate, seeds)
    return summaries
