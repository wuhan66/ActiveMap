#!/usr/bin/env python3
"""Import a sealed SN7 frozen-test export into the ICLR paper package.

The importer is sign-agnostic: a failed promotion is rendered and recorded in
exactly the same way as a passed promotion. It never reads model checkpoints or
test assets; it consumes only the sealed reporter outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


EXPECTED_SEEDS = [20260730, 20260731, 20260801]
EXPECTED_VARIANTS = ("notool", "forced", "benefit")
SNAPSHOT_SOURCE = "evidence/generated/sn7_frozen_result_snapshot.json"
LEGACY_SNAPSHOT_SOURCE = "evidence/evidence_snapshot.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != "sn7-step0-frozen-test-tables-v1":
        raise ValueError("unexpected frozen table schema")
    if manifest.get("split") != "test" or manifest.get("test_assets_read") is not True:
        raise ValueError("result bundle is not sealed test evidence")
    if manifest.get("reporting_policy") != "report_regardless_of_promotion_outcome":
        raise ValueError("result bundle does not enforce sign-agnostic reporting")
    if sorted(map(int, manifest.get("seeds", []))) != EXPECTED_SEEDS:
        raise ValueError("frozen seed support differs from the paper contract")
    if manifest.get("protocol_recovery") != "v3_cap20_launcher_only":
        raise ValueError("result bundle is not the audited v3-cap20 recovery")
    for key in ("metric_reporter_sha256", "compatibility_adapter_sha256"):
        value = str(manifest.get(key, ""))
        if len(value) != 64:
            raise ValueError(f"missing or invalid {key}")


def _validate_rows(rows: list[dict[str, str]], name: str) -> None:
    variants = [row.get("variant") for row in rows]
    if variants != list(EXPECTED_VARIANTS):
        raise ValueError(f"{name} variants differ from the frozen contract: {variants}")


def _tex_number(value: Any, *, signed: bool = False) -> str:
    number = float(value)
    return f"{number:+.6f}" if signed else f"{number:.6f}"


def _interval(record: dict[str, Any]) -> str:
    return (
        f"{_tex_number(record['delta'], signed=True)} "
        f"[{_tex_number(record['ci95_low'], signed=True)},"
        f"{_tex_number(record['ci95_high'], signed=True)}]"
    )


def _escape(value: str) -> str:
    return value.replace("&", r"\&").replace("%", r"\%")


def _render_macros(paired: dict[str, Any], promotion_passed: bool) -> str:
    no_tool = paired["writeback"]["benefit_vs_notool"]
    forced = paired["writeback"]["benefit_vs_forced"]
    raster = no_tool["raster_iou_auc"]
    false_edit = no_tool["false_edit_auc"]
    missed = no_tool["missed_edit_auc"]
    safety = no_tool["episode_utility_v2_safety_auc"]
    cost = forced["spent_cost_auc"]
    gate = "passed" if promotion_passed else "failed"
    consequence = (
        "extends the validation-supported acquisition claim to the registered test"
        if promotion_passed
        else "does not promote the validation-supported acquisition claim to the registered test"
    )
    return f"""% Generated from sealed SN7 frozen-test tables. Do not edit by hand.
\\newcommand{{\\SNSevenFrozenAbstractSentence}}{{%
On the registered one-time SN7 test, ActiveMap changes Raster-IoU AUC by
${_interval(raster)}$, false-edit AUC by ${_interval(false_edit)}$, and safety
utility by ${_interval(safety)}$ relative to no additional evidence; the
predeclared joint promotion gate {gate}.}}
\\newcommand{{\\SNSevenFrozenResultParagraph}}{{%
The immutable cap-matched test contains 6,573 episodes from nine held-out AOIs
and uses the three predeclared controller seeds. Relative to no additional
evidence, ActiveMap changes Raster-IoU AUC by ${_interval(raster)}$,
false-edit AUC by ${_interval(false_edit)}$, missed-edit AUC by
${_interval(missed)}$, and safety utility by ${_interval(safety)}$. Relative
to forced acquisition, spent-cost AUC changes by ${_interval(cost)}$. The
registered promotion gate {gate}; all pooled and operation-stratified results
are reported in Appendix~\\ref{{app:sn7_frozen_result}} regardless of sign.}}
\\newcommand{{\\SNSevenFrozenConclusionSentence}}{{%
The registered frozen-test gate {gate} and therefore {consequence}.}}
"""


def _render_tables(
    controller_rows: list[dict[str, str]],
    writeback_rows: list[dict[str, str]],
    paired: dict[str, Any],
    operation_slices: dict[str, Any] | None,
    promotion_passed: bool,
) -> str:
    gate = "passed" if promotion_passed else "failed"
    lines = [
        "% Generated from sealed SN7 frozen-test tables. Do not edit by hand.",
        r"\section{One-Time SN7 Frozen-Test Result}",
        r"\label{app:sn7_frozen_result}",
        "",
        "The v3-cap20 recovery preserves the registered checkpoints, seeds,",
        "budgets, thresholds, AOIs, and metric reporter. The predeclared joint",
        f"promotion gate \\textbf{{{gate}}}; the complete result is retained",
        "regardless of that decision.",
        "",
        r"\begin{table*}[t]",
        r"\caption{Registered SN7 frozen-test controller behavior. Values are three-seed mean $\pm$ sample standard deviation.}",
        r"\label{tab:sn7_frozen_controller}",
        r"\centering\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Policy & Terminal acc. & False edit & Missed edit & Q-C utility & Tool calls \\",
        r"\midrule",
    ]
    controller_fields = (
        "terminal_accuracy",
        "false_edit_rate",
        "missed_edit_rate",
        "mean_quality_cost_utility",
        "mean_tool_calls",
    )
    for row in controller_rows:
        values = [
            f"{float(row[f'{field}_mean']):.6f} $\\pm$ {float(row[f'{field}_std']):.6f}"
            for field in controller_fields
        ]
        lines.append(_escape(row["method"]) + " & " + " & ".join(values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
            r"\begin{table*}[t]",
            r"\caption{Registered SN7 executable-writeback aggregates over budgets. All methods share the same initial maps, updater, and terminal evaluator.}",
            r"\label{tab:sn7_frozen_writeback}",
            r"\centering\scriptsize",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{@{}lrrrrrr@{}}",
            r"\toprule",
            r"Policy & Raster IoU & False edit & Missed edit & Cost & Balanced utility & Safety utility \\",
            r"\midrule",
        ]
    )
    writeback_fields = (
        "raster_iou_auc",
        "false_edit_auc",
        "missed_edit_auc",
        "spent_cost_auc",
        "episode_utility_v2_balanced_auc",
        "episode_utility_v2_safety_auc",
    )
    for row in writeback_rows:
        values = [_tex_number(row[field]) for field in writeback_fields]
        lines.append(_escape(row["method"]) + " & " + " & ".join(values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
            r"\begin{table*}[t]",
            r"\caption{ActiveMap paired frozen-test deltas with hierarchical AOI bootstrap intervals. Negative false edit, missed edit, and cost are favorable.}",
            r"\label{tab:sn7_frozen_paired}",
            r"\centering\scriptsize",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{@{}llrrr@{}}",
            r"\toprule",
            r"Reference & Raster IoU & False edit & Missed edit & Safety utility \\",
            r"\midrule",
        ]
    )
    for key, label in (("benefit_vs_notool", "No additional evidence"), ("benefit_vs_forced", "Forced acquisition")):
        record = paired["writeback"][key]
        values = [
            _interval(record[field])
            for field in (
                "raster_iou_auc",
                "false_edit_auc",
                "missed_edit_auc",
                "episode_utility_v2_safety_auc",
            )
        ]
        lines.append(label + " & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""])

    if operation_slices is not None:
        lines.extend(
            [
                r"\begin{table*}[t]",
                r"\caption{Post-hoc descriptive frozen-test slices by target edit. Deltas are ActiveMap minus no additional evidence and cannot alter promotion.}",
                r"\label{tab:sn7_frozen_operations}",
                r"\centering\scriptsize",
                r"\begin{tabular}{@{}lrrrrrr@{}}",
                r"\toprule",
                r"Target & Support/seed & AOIs & Map quality & False edit & Missed edit & Cost \\",
                r"\midrule",
            ]
        )
        for operation in ("KEEP", "ADD", "DELETE", "RESHAPE"):
            row = operation_slices["operations"][operation]
            delta = row["candidate_minus_reference_mean"]
            values = [
                str(int(row["support_per_seed"])),
                str(int(row["aoi_count"])),
                _tex_number(delta["map_quality_after"], signed=True),
                _tex_number(delta["false_edit"], signed=True),
                _tex_number(delta["missed_edit"], signed=True),
                _tex_number(delta["spent_cost"], signed=True),
            ]
            lines.append(operation + " & " + " & ".join(values) + r" \\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines) + "\n"


def _numeric_entries(value: Any, pointer: str = "") -> Iterator[dict[str, Any]]:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return
    if isinstance(value, (int, float)):
        yield {
            "value": value,
            "source": SNAPSHOT_SOURCE,
            "selector": {"kind": "json-pointer", "pointer": pointer or "/"},
            "aggregate": "identity",
            "representations": ["raw"],
            "tolerance": 1e-6,
        }
        return
    if isinstance(value, dict):
        for key, item in value.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            yield from _numeric_entries(item, f"{pointer}/{escaped}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _numeric_entries(item, f"{pointer}/{index}")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def import_result(
    export_dir: Path,
    paper_dir: Path,
    *,
    operation_slices_path: Path | None = None,
) -> dict[str, Any]:
    inputs = {
        "manifest": export_dir / "manifest.json",
        "controller_table": export_dir / "controller_table.csv",
        "writeback_table": export_dir / "writeback_table.csv",
        "paired_intervals": export_dir / "paired_intervals.json",
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = _load(inputs["manifest"])
    _validate_manifest(manifest)
    controller_rows = _rows(inputs["controller_table"])
    writeback_rows = _rows(inputs["writeback_table"])
    _validate_rows(controller_rows, "controller table")
    _validate_rows(writeback_rows, "writeback table")
    paired = _load(inputs["paired_intervals"])

    operation_slices = None
    if operation_slices_path is not None:
        operation_slices = _load(operation_slices_path)
        if operation_slices.get("schema_version") != "sn7-frozen-test-operation-slices-v1":
            raise ValueError("unexpected operation-slice schema")
        if operation_slices.get("split") != "test" or operation_slices.get("test_assets_read") is not True:
            raise ValueError("operation slices are not frozen-test evidence")
        if sorted(map(int, operation_slices.get("model_seeds", []))) != EXPECTED_SEEDS:
            raise ValueError("operation-slice seeds differ from the frozen contract")
        inputs["operation_slices"] = operation_slices_path

    promotion_passed = bool(manifest["promotion_passed"])
    snapshot = {
        "schema_version": "sn7-frozen-result-paper-snapshot-v1",
        "split": "test",
        "test_assets_read": True,
        "promotion_passed": promotion_passed,
        "manifest": manifest,
        "controller_rows": controller_rows,
        "writeback_rows": writeback_rows,
        "paired_intervals": paired,
        "operation_slices": operation_slices,
    }
    generated = paper_dir / "evidence" / "generated"
    snapshot_path = generated / "sn7_frozen_result_snapshot.json"
    macros_path = generated / "sn7_frozen_result_macros.tex"
    tables_path = generated / "sn7_frozen_result_tables.tex"
    _atomic_text(snapshot_path, json.dumps(snapshot, indent=2) + "\n")
    _atomic_text(macros_path, _render_macros(paired, promotion_passed))
    _atomic_text(
        tables_path,
        _render_tables(
            controller_rows,
            writeback_rows,
            paired,
            operation_slices,
            promotion_passed,
        ),
    )

    numeric_path = paper_dir / "numeric_evidence.json"
    numeric = _load(numeric_path)
    # Replace only the historical SN7 entries. The legacy snapshot still owns
    # MUNO21 and SpaceNet8 provenance, so removing its complete entry set would
    # silently detach those sections from their existing evidence sources.
    entries = []
    for entry in numeric.get("entries", []):
        source = entry.get("source")
        selector = entry.get("selector", {})
        pointer = selector.get("pointer", "") if isinstance(selector, dict) else ""
        is_legacy_sn7 = source == LEGACY_SNAPSHOT_SOURCE and str(pointer).startswith("/sn7/")
        if source != SNAPSHOT_SOURCE and not is_legacy_sn7:
            entries.append(entry)
    entries.extend(_numeric_entries(snapshot))
    numeric["entries"] = entries
    _atomic_text(numeric_path, json.dumps(numeric, indent=2) + "\n")

    receipt = {
        "schema_version": "sn7-frozen-result-paper-import-v1",
        "split": "test",
        "test_assets_read": True,
        "reporting_policy": "report_regardless_of_promotion_outcome",
        "promotion_passed": promotion_passed,
        "seeds": EXPECTED_SEEDS,
        "inputs": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in inputs.items()},
        "outputs": {},
    }
    receipt_path = generated / "sn7_frozen_result_import_receipt.json"
    for name, path in {
        "snapshot": snapshot_path,
        "macros": macros_path,
        "tables": tables_path,
        "numeric_evidence": numeric_path,
    }.items():
        receipt["outputs"][name] = {"path": str(path), "sha256": _sha256(path)}
    _atomic_text(receipt_path, json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("paper_dir", type=Path)
    parser.add_argument("--operation-slices", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            import_result(
                args.export_dir,
                args.paper_dir,
                operation_slices_path=args.operation_slices,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
