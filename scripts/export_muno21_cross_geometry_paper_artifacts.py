#!/usr/bin/env python3
"""Export validation-only MUNO21 cross-geometry paper evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


SEMANTIC_METRICS = (
    "terminal_accuracy",
    "false_edit_rate",
    "missed_edit_rate",
    "mean_episode_utility_v2_proxy_balanced",
    "mean_episode_utility_v2_proxy_safety",
    "mean_episode_utility_v2_proxy_cost_aware",
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_validation(payload: dict[str, Any], path: Path) -> None:
    test_read = payload.get(
        "test_assets_read",
        payload.get("protocol", {}).get(
            "test_assets_read", payload.get("interpretation", {}).get("test_assets_read")
        ),
    )
    split = payload.get("split")
    if test_read is not False or (split is not None and split != "val"):
        raise ValueError(f"not a validation-only artifact: {path}")


def _load_official(path: Path, comparison: str) -> list[dict[str, Any]]:
    payload = _load(path)
    _assert_validation(payload, path)
    if payload.get("schema_version") != "muno21-official-three-seed-aggregate-v1":
        raise ValueError(f"unexpected official schema: {path}")
    if int(payload.get("seed_count", 0)) != 3:
        raise ValueError(f"official result must contain three seeds: {path}")
    aggregate = payload["aggregate"]
    rows = []
    for metric in ("apls_improvement", "pixel_f1_improvement"):
        item = aggregate["paired_metrics"][metric]
        rows.append(
            {
                "comparison": comparison,
                "metric": metric,
                "mean_delta": float(item["mean_delta"]),
                "seed_std": float(item["seed_standard_deviation"]),
                "ci95_low": float(item["hierarchical_ci95_low"]),
                "ci95_high": float(item["hierarchical_ci95_high"]),
                "per_seed_delta": {
                    key: float(value) for key, value in item["per_seed_delta"].items()
                },
            }
        )
    rows.append(
        {
            "comparison": comparison,
            "metric": "no_change_error_rate",
            "mean_delta": float(aggregate["mean_no_change_error_rate_delta"]),
            "seed_std": float(aggregate["no_change_error_seed_standard_deviation"]),
            "ci95_low": None,
            "ci95_high": None,
            "per_seed_delta": {
                key: float(value)
                for key, value in aggregate[
                    "per_seed_no_change_error_rate_delta"
                ].items()
            },
        }
    )
    return rows


def _load_semantic(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(paths) != 3:
        raise ValueError("semantic evidence requires exactly three policy seeds")
    per_seed = []
    for seed, path in enumerate(paths, start=1):
        payload = _load(path)
        _assert_validation(payload, path)
        if (
            payload.get("schema_version")
            != "active-catalog-closed-loop-paired-comparison-v1"
            or int(payload.get("record_count", 0)) != 414
        ):
            raise ValueError(f"unexpected semantic-tool result: {path}")
        paired = payload["paired_aoi_comparisons"]["selective_minus_notool"]
        if int(paired.get("group_count", 0)) != 2:
            raise ValueError(f"expected two validation AOIs: {path}")
        row: dict[str, Any] = {
            "policy_seed": seed,
            "episodes": int(payload["record_count"]),
            "aoi_count": int(paired["group_count"]),
            "tool_call_rate": float(payload["metrics"]["selective"]["tool_call_episode_rate"]),
        }
        for metric in SEMANTIC_METRICS:
            interval = paired["intervals"][metric]
            row[f"{metric}_delta"] = float(interval["observed_delta"])
            row[f"{metric}_ci95_low"] = float(interval["ci95_low"])
            row[f"{metric}_ci95_high"] = float(interval["ci95_high"])
        per_seed.append(row)

    aggregate: dict[str, Any] = {
        "policy_seed_count": 3,
        "episodes_per_seed": 414,
        "validation_aois_per_seed": 2,
        "tool_call_rate": {
            "mean": mean(row["tool_call_rate"] for row in per_seed),
            "seed_std": stdev(row["tool_call_rate"] for row in per_seed),
        },
    }
    for metric in SEMANTIC_METRICS:
        values = [row[f"{metric}_delta"] for row in per_seed]
        aggregate[f"{metric}_delta"] = {
            "mean": mean(values),
            "seed_std": stdev(values),
            "all_seeds_same_direction": all(value >= 0 for value in values)
            or all(value <= 0 for value in values),
        }
    return per_seed, aggregate


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [key for key in rows[0] if key != "per_seed_delta"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def _markdown(
    official: list[dict[str, Any]],
    semantic: dict[str, Any],
) -> str:
    apls = {row["comparison"]: row for row in official if row["metric"] == "apls_improvement"}
    lines = [
        "# MUNO21 Cross-Geometry Validation Evidence",
        "",
        "## Official Road-Graph Metric",
        "",
        "| Comparison | APLS delta | Hierarchical 95% CI |",
        "| --- | ---: | ---: |",
    ]
    for comparison in ("vs_always_stop", "vs_old_selector"):
        row = apls[comparison]
        lines.append(
            f"| {comparison} | {row['mean_delta']:+.6f} | "
            f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] |"
        )
    lines.extend(
        [
            "",
            "## Selective Semantic Tool Use",
            "",
            "| Metric | Mean delta across policy seeds | Seed std |",
            "| --- | ---: | ---: |",
        ]
    )
    labels = (
        ("terminal_accuracy", "Terminal accuracy"),
        ("false_edit_rate", "False-edit rate"),
        ("missed_edit_rate", "Missed-edit rate"),
        ("mean_episode_utility_v2_proxy_balanced", "Balanced utility"),
        ("mean_episode_utility_v2_proxy_safety", "Safety utility"),
        ("mean_episode_utility_v2_proxy_cost_aware", "Cost-aware utility"),
    )
    for metric, label in labels:
        item = semantic[f"{metric}_delta"]
        lines.append(f"| {label} | {item['mean']:+.6f} | {item['seed_std']:.6f} |")
    lines.extend(
        [
            f"| Tool-call rate | {semantic['tool_call_rate']['mean']:.6f} | "
            f"{semantic['tool_call_rate']['seed_std']:.6f} |",
            "",
            "Validation only; test assets were not read.",
            "The official APLS result supports cross-geometry transfer. Semantic-tool "
            "results support a consistent three-seed trend, but each seed contains "
            "only two validation AOIs and is not claimed as a strong per-seed "
            "significance result.",
            "",
        ]
    )
    return "\n".join(lines)


def _latex(
    official: list[dict[str, Any]],
    semantic: dict[str, Any],
) -> str:
    apls = {
        row["comparison"]: row
        for row in official
        if row["metric"] == "apls_improvement"
    }
    lines = [
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Comparison & APLS $\\Delta$ & Hierarchical 95\\% CI \\\\",
        "\\midrule",
    ]
    for comparison, label in (
        ("vs_always_stop", "vs. always-stop"),
        ("vs_old_selector", "vs. old selector"),
    ):
        row = apls[comparison]
        lines.append(
            f"{label} & {row['mean_delta']:+.6f} & "
            f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "",
            "\\begin{tabular}{lrr}",
            "\\toprule",
            "Selective semantic tool & Mean $\\Delta$ & Seed std. \\\\",
            "\\midrule",
        ]
    )
    labels = (
        ("terminal_accuracy", "Terminal accuracy"),
        ("false_edit_rate", "False-edit rate"),
        ("missed_edit_rate", "Missed-edit rate"),
        ("mean_episode_utility_v2_proxy_balanced", "Balanced utility"),
        ("mean_episode_utility_v2_proxy_safety", "Safety utility"),
        ("mean_episode_utility_v2_proxy_cost_aware", "Cost-aware utility"),
    )
    for metric, label in labels:
        item = semantic[f"{metric}_delta"]
        lines.append(f"{label} & {item['mean']:+.6f} & {item['seed_std']:.6f} \\\\")
    call = semantic["tool_call_rate"]
    lines.extend(
        [
            f"Tool-call rate & {call['mean']:.6f} & {call['seed_std']:.6f} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "",
        ]
    )
    return "\n".join(lines)


def _plot(
    official: list[dict[str, Any]],
    semantic: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "savefig.dpi": 400,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    apls = [row for row in official if row["metric"] == "apls_improvement"]
    labels = ["Always-stop", "Old selector"]
    values = [row["mean_delta"] for row in apls]
    low = [row["mean_delta"] - row["ci95_low"] for row in apls]
    high = [row["ci95_high"] - row["mean_delta"] for row in apls]
    axes[0].bar(labels, values, color=["#008F83", "#4472A8"], width=0.58)
    axes[0].errorbar(
        range(len(values)),
        values,
        yerr=[low, high],
        fmt="none",
        color="#222222",
        capsize=3,
        linewidth=1.0,
    )
    axes[0].axhline(0.0, color="#222222", linewidth=0.8)
    axes[0].set_ylabel("Official APLS delta")
    axes[0].set_title("(a) Road-graph transfer", loc="left", fontweight="bold")

    metrics = (
        ("terminal_accuracy", "Accuracy"),
        ("false_edit_rate", "False edit"),
        ("mean_episode_utility_v2_proxy_balanced", "Utility"),
    )
    values = [semantic[f"{metric}_delta"]["mean"] for metric, _ in metrics]
    errors = [semantic[f"{metric}_delta"]["seed_std"] for metric, _ in metrics]
    colors = ["#008F83", "#D95F4A", "#4472A8"]
    axes[1].bar([label for _, label in metrics], values, color=colors, width=0.58)
    axes[1].errorbar(
        range(len(values)),
        values,
        yerr=errors,
        fmt="none",
        color="#222222",
        capsize=3,
        linewidth=1.0,
    )
    axes[1].axhline(0.0, color="#222222", linewidth=0.8)
    axes[1].set_ylabel("Selective - no-tool")
    axes[1].set_title("(b) Multi-operation control", loc="left", fontweight="bold")

    for ax in axes:
        ax.grid(axis="y", color="#D9DDE0", linewidth=0.6, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.18, top=0.9, wspace=0.32)
    outputs = []
    for suffix in ("png", "pdf", "svg"):
        path = output_dir / f"muno21_cross_geometry_validation.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    plt.close(fig)
    return outputs


def export(
    official_stop: Path,
    official_old: Path,
    semantic_paths: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    official = [
        *_load_official(official_stop, "vs_always_stop"),
        *_load_official(official_old, "vs_old_selector"),
    ]
    semantic_rows, semantic_aggregate = _load_semantic(semantic_paths)
    output_dir.mkdir(parents=True)
    _write_csv(output_dir / "official_graph_metrics.csv", official)
    _write_csv(output_dir / "semantic_tool_per_seed.csv", semantic_rows)
    (output_dir / "semantic_tool_aggregate.json").write_text(
        json.dumps(semantic_aggregate, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "cross_geometry_summary.md").write_text(
        _markdown(official, semantic_aggregate), encoding="utf-8"
    )
    (output_dir / "cross_geometry_tables.tex").write_text(
        _latex(official, semantic_aggregate), encoding="utf-8"
    )
    figures = _plot(official, semantic_aggregate, output_dir)
    inputs = [official_stop, official_old, *semantic_paths]
    manifest = {
        "schema_version": "muno21-cross-geometry-paper-artifacts-v1",
        "split": "val",
        "test_assets_read": False,
        "support": {
            "official_policy_seeds": 3,
            "official_tasks": 29,
            "official_budgets": [1.5, 3.0, 4.5],
            "semantic_policy_seeds": 3,
            "semantic_episodes_per_seed": 414,
            "semantic_validation_aois_per_seed": 2,
        },
        "claim_boundary": (
            "Official APLS supports road-graph cross-geometry transfer. "
            "Selective semantic-tool results are a consistent validation trend; "
            "they are not universal-domain or strong per-seed significance claims."
        ),
        "figure_error_bars": {
            "official_apls": "hierarchical bootstrap 95% confidence interval",
            "semantic_tool": "sample standard deviation across three policy seeds",
        },
        "sources": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in inputs
        ],
        "figures": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in figures
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("official_stop", type=Path)
    parser.add_argument("official_old", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--semantic", type=Path, action="append", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                args.official_stop,
                args.official_old,
                args.semantic,
                args.output_dir,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
