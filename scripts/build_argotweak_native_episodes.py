from __future__ import annotations

import argparse
import json
from pathlib import Path

from activemap.data.argotweak_native import build_argotweak_native_episodes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build native ActiveMap episodes from frozen ArgoTweak proposals."
    )
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit-confidence", type=float, default=0.5)
    parser.add_argument("--false-edit-weight", type=float, default=0.5)
    parser.add_argument("--missed-edit-weight", type=float, default=0.5)
    parser.add_argument("--cost-weight", type=float, default=0.05)
    args = parser.parse_args()
    summary = build_argotweak_native_episodes(
        args.proposals,
        args.scenes,
        args.output,
        commit_confidence=args.commit_confidence,
        false_edit_weight=args.false_edit_weight,
        missed_edit_weight=args.missed_edit_weight,
        cost_weight=args.cost_weight,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
