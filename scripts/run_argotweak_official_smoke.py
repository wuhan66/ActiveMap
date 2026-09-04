from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the official ArgoTweak test entry point with release compatibility fixes."
    )
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    official_root = args.official_root.resolve()
    sys.path.insert(0, str(official_root))
    os.chdir(official_root)

    from tools import test as official_test

    original_parse_args = official_test.parse_args

    def parse_args_with_release_fix():
        parsed = original_parse_args()
        # The released test.py reads show_dir although its parser omits it.
        if not hasattr(parsed, "show_dir"):
            parsed.show_dir = None
        return parsed

    official_test.parse_args = parse_args_with_release_fix
    sys.argv = [
        "test.py",
        str(args.config),
        str(args.checkpoint),
        "--out",
        "--out-dir",
        str(args.out_dir),
        "--cfg-options",
        f"data.test.ann_file={args.ann_file}",
        f"data.workers_per_gpu={args.workers}",
    ]
    official_test.main()


if __name__ == "__main__":
    main()
