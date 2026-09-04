#!/usr/bin/env python3
"""Evaluate an ArgoTweak visual selector on a frozen, unseen split."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    import numpy as np
    import torch
    from torch import nn

    records = [
        json.loads(line)
        for line in (args.features / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw = np.load(args.features / "features.npy").astype(np.float32)
    costs = np.asarray([float(row["cost"]) for row in records], dtype=np.float32)
    costs = (costs - costs.mean()) / max(float(costs.std()), 1e-6)
    features = np.concatenate([raw, costs[:, None]], axis=1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    features = (features - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    model = nn.Sequential(
        nn.Linear(checkpoint["input_dim"], 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1),
    ).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.inference_mode():
        scores = model(torch.from_numpy(features).to(args.device)).squeeze(1).cpu().tolist()
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        grouped[str(row["episode_id"])].append(index)
    selected = oracle = direct = exact = selected_cost = 0.0
    rankings: list[dict[str, Any]] = []
    for episode_id, indices in sorted(grouped.items()):
        ranked = sorted(indices, key=lambda index: (-scores[index], records[index]["evidence_id"]))
        oracle_index = max(
            indices,
            key=lambda index: (records[index]["utility"], records[index]["evidence_id"]),
        )
        direct_index = indices[len(indices) // 2]
        selected += float(records[ranked[0]]["utility"])
        oracle += float(records[oracle_index]["utility"])
        direct += float(records[direct_index]["utility"])
        selected_cost += float(records[ranked[0]]["cost"])
        exact += float(ranked[0] == oracle_index)
        rankings.append({
            "episode_id": episode_id,
            "ranked_evidence_ids": [records[index]["evidence_id"] for index in ranked],
            "scores": [scores[index] for index in ranked],
        })
    count = len(grouped)
    summary = {
        "schema_version": "activemap-argotweak-visual-selector-evaluation-v1",
        "episodes": count,
        "mean_selected_utility": selected / count,
        "mean_direct_center_utility": direct / count,
        "mean_oracle_utility": oracle / count,
        "mean_regret": (oracle - selected) / count,
        "utility_gain_over_direct": (selected - direct) / count,
        "exact_top1": exact / count,
        "mean_selected_cost": selected_cost / count,
        "test_assets_read": False,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "rankings.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rankings), encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
