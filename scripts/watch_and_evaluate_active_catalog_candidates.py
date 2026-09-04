from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def candidate_adapters(run_dir: Path) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    retained = run_dir / "retained_candidates"
    if retained.exists():
        for path in sorted(retained.glob("checkpoint-*")):
            if (path / "adapter_config.json").exists():
                candidates.append((path.name, path))
    final = run_dir / "final"
    if (final / "adapter_config.json").exists():
        candidates.append(("final", final))
    return candidates


def completed_process_result(run_dir: Path) -> dict[str, object] | None:
    path = run_dir / "process_result.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("returncode", -1)) != 0:
        raise RuntimeError(f"training failed according to {path}: {payload}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for a finished SFT run and sentinel-evaluate its retained adapters."
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("sentinel_dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--expected-records", type=int, default=90)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    deadline = time.monotonic() + args.timeout_hours * 3600
    final_marker = args.run_dir / "final" / "adapter_config.json"
    process_result = None
    while not final_marker.exists() or process_result is None:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for completed run {args.run_dir}")
        process_result = completed_process_result(args.run_dir)
        time.sleep(args.poll_seconds)

    val_jsonl = args.sentinel_dir / "val.jsonl"
    evaluation_index = args.sentinel_dir / "val_evaluation_index.jsonl"
    if not val_jsonl.exists() or not evaluation_index.exists():
        raise FileNotFoundError(f"incomplete sentinel directory: {args.sentinel_dir}")

    records: list[dict[str, object]] = []
    evaluator = Path(__file__).with_name("evaluate_active_catalog_selector.py")
    for label, adapter in candidate_adapters(args.run_dir):
        output_dir = args.run_dir / f"sentinel_eval_{label}"
        summary = output_dir / "summary.json"
        if summary.exists():
            status = "skipped_complete"
        else:
            command = [
                sys.executable,
                str(evaluator),
                str(args.model),
                str(adapter),
                str(val_jsonl),
                str(evaluation_index),
                str(output_dir),
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--expected-records",
                str(args.expected_records),
                "--bootstrap-repetitions",
                str(args.bootstrap_repetitions),
            ]
            subprocess.run(command, check=True)
            status = "completed"
        records.append(
            {
                "label": label,
                "adapter": str(adapter),
                "summary": str(summary),
                "status": status,
            }
        )

    manifest = {
        "schema_version": "active-catalog-candidate-sentinel-relay-v1",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(args.run_dir),
        "sentinel_dir": str(args.sentinel_dir),
        "expected_records": args.expected_records,
        "training_process_result": process_result,
        "candidates": records,
        "test_assets_read": False,
    }
    output = args.run_dir / "sentinel_candidate_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
