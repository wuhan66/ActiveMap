#!/usr/bin/env python3
"""Average normalized evidence scores from independently trained selectors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load(path: Path) -> dict[str, dict[str, float]]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        result[str(row["episode_id"])] = {
            str(evidence_id): float(score)
            for evidence_id, score in zip(row["ranked_evidence_ids"], row["scores"], strict=True)
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rankings", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    models = [_load(path) for path in args.rankings]
    episodes = set(models[0])
    if any(set(model) != episodes for model in models[1:]):
        raise ValueError("ranking files do not contain identical episodes")
    output = []
    for episode_id in sorted(episodes):
        evidence_ids = set(models[0][episode_id])
        if any(set(model[episode_id]) != evidence_ids for model in models[1:]):
            raise ValueError(f"evidence mismatch for {episode_id}")
        normalized = []
        for model in models:
            scores = model[episode_id]
            mean = sum(scores.values()) / len(scores)
            variance = sum((value - mean) ** 2 for value in scores.values()) / len(scores)
            std = max(math.sqrt(variance), 1e-8)
            normalized.append({key: (value - mean) / std for key, value in scores.items()})
        ensemble = {
            evidence_id: sum(model[evidence_id] for model in normalized) / len(normalized)
            for evidence_id in evidence_ids
        }
        ranked = sorted(evidence_ids, key=lambda key: (-ensemble[key], key))
        output.append(
            {
                "episode_id": episode_id,
                "ranked_evidence_ids": ranked,
                "scores": [ensemble[key] for key in ranked],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in output),
        encoding="utf-8",
    )
    print(json.dumps({"episodes": len(output), "members": len(models), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
