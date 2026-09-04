#!/usr/bin/env python3
"""Select an SFT checkpoint using generated validation actions, never token loss."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def action_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("candidate trace is empty")
    if {str(row.get("split")) for row in rows} != {"val"}:
        raise ValueError("checkpoint selection only permits validation traces")
    targets = [str(row["target_selection"]) for row in rows]
    predictions = [str(row["predicted_selection"]) for row in rows]
    acquire_targets = sum(value == "ACQUIRE" for value in targets)
    stop_targets = len(rows) - acquire_targets
    true_acquire = sum(t == "ACQUIRE" and p == "ACQUIRE" for t, p in zip(targets, predictions))
    false_acquire = sum(t == "STOP" and p == "ACQUIRE" for t, p in zip(targets, predictions))
    missed_acquire = acquire_targets - true_acquire
    true_stop = stop_targets - false_acquire

    acquire_precision = true_acquire / max(true_acquire + false_acquire, 1)
    acquire_recall = true_acquire / max(acquire_targets, 1)
    acquire_f1 = 2 * acquire_precision * acquire_recall / max(acquire_precision + acquire_recall, 1e-12)
    stop_precision = true_stop / max(true_stop + missed_acquire, 1)
    stop_recall = true_stop / max(stop_targets, 1)
    stop_f1 = 2 * stop_precision * stop_recall / max(stop_precision + stop_recall, 1e-12)
    exact = sum(
        row["target_selection"] == "ACQUIRE"
        and row["predicted_selection"] == "ACQUIRE"
        and row.get("target_evidence_id") == row.get("predicted_evidence_id")
        for row in rows
    )
    harmful = sum(
        row["predicted_selection"] == "ACQUIRE"
        and float(row["policy_utility"]) < float(row["stop_utility"])
        for row in rows
    )
    return {
        "states": float(len(rows)),
        "valid_action_rate": sum(bool(row.get("valid_action")) for row in rows) / len(rows),
        "predicted_call_rate": sum(value == "ACQUIRE" for value in predictions) / len(rows),
        "acquire_precision": acquire_precision,
        "acquire_recall": acquire_recall,
        "acquire_f1": acquire_f1,
        "selection_macro_f1": (acquire_f1 + stop_f1) / 2,
        "false_call_rate": false_acquire / max(stop_targets, 1),
        "exact_evidence_recall": exact / max(acquire_targets, 1),
        "harmful_call_rate_all_states": harmful / len(rows),
        "realized_utility_mean": sum(float(row["policy_utility"]) for row in rows) / len(rows),
        "mean_regret": sum(float(row["regret"]) for row in rows) / len(rows),
    }


def choose_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda candidate: (
            candidate["metrics"]["realized_utility_mean"],
            candidate["metrics"]["selection_macro_f1"],
            candidate["metrics"]["exact_evidence_recall"],
            -candidate["metrics"]["false_call_rate"],
            candidate["label"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--candidate", action="append", nargs=3, metavar=("LABEL", "ADAPTER", "TRACE"), required=True
    )
    parser.add_argument("--min-valid-action-rate", type=float, default=0.99)
    parser.add_argument("--min-acquire-recall", type=float, default=0.05)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    candidates = []
    for label, adapter_text, trace_text in args.candidate:
        adapter, trace = Path(adapter_text), Path(trace_text)
        config = adapter / "adapter_config.json"
        weights = adapter / "adapter_model.safetensors"
        if not config.is_file() or not weights.is_file() or not trace.is_file():
            raise FileNotFoundError(f"incomplete candidate {label}")
        rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line]
        metrics = action_metrics(rows)
        eligible = (
            metrics["valid_action_rate"] >= args.min_valid_action_rate
            and metrics["predicted_call_rate"] > 0
            and metrics["acquire_recall"] >= args.min_acquire_recall
        )
        candidates.append(
            {
                "label": label,
                "adapter": str(adapter.resolve()),
                "adapter_config_sha256": _sha256(config),
                "adapter_weights_sha256": _sha256(weights),
                "trace": str(trace.resolve()),
                "trace_sha256": _sha256(trace),
                "metrics": metrics,
                "eligible": eligible,
            }
        )
    selected = choose_candidate(candidates)
    payload = {
        "schema_version": "active-catalog-sentinel-checkpoint-selection-v1",
        "selection_basis": "generated_validation_actions_not_teacher_forced_loss",
        "candidates": candidates,
        "selected": selected,
        "passed": selected is not None,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "test_assets_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if selected is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
