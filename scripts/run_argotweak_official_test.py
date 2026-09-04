from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official ArgoTweak validation compatibly.")
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    if (args.output_dir / "results.pkl").exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir / 'results.pkl'}")

    official_root = args.official_root.resolve()
    sys.path.insert(0, str(official_root))
    os.chdir(official_root)

    import mmcv.utils.config as mmcv_config
    from yapf.yapflib.yapf_api import FormatCode as yapf_format_code

    def format_code_compat(text, style_config=None, verify=False):
        del verify
        return yapf_format_code(text, style_config=style_config)

    mmcv_config.FormatCode = format_code_compat
    from tools import test as official_test

    original_parse_args = official_test.parse_args

    def parse_args_compat():
        parsed = original_parse_args()
        if not hasattr(parsed, "show_dir"):
            parsed.show_dir = None
        return parsed

    official_test.parse_args = parse_args_compat
    sys.argv = [
        "test.py",
        str(args.config.resolve()),
        str(args.checkpoint.resolve()),
        "--out",
        "--eval",
        "all",
        "--out-dir",
        str(args.output_dir.resolve()),
        "--seed",
        str(args.seed),
        "--deterministic",
        "--launcher",
        "none",
        "--cfg-options",
        f"data.test.ann_file={args.annotations.resolve()}",
        f"data.workers_per_gpu={args.workers}",
    ]
    official_test.main()


if __name__ == "__main__":
    main()
