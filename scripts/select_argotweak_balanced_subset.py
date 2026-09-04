from __future__ import annotations

import argparse
from pathlib import Path

from activemap.data.argotweak_subset import build_argotweak_balanced_subset


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a balanced ArgoTweak train/val pilot.")
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=24)
    parser.add_argument("--val-count", type=int, default=8)
    args = parser.parse_args()
    summary = build_argotweak_balanced_subset(
        args.splits,
        args.annotation_root,
        args.output_dir,
        train_count=args.train_count,
        val_count=args.val_count,
    )
    print(summary)


if __name__ == "__main__":
    main()
