#!/usr/bin/env python3
"""Verify an SFT adapter promotion record and every referenced file hash."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.promote_sparse_tool_sft_adapter import verify_adapter_promotion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("promotion", type=Path)
    args = parser.parse_args()
    promotion = verify_adapter_promotion(args.promotion)
    print(
        json.dumps(
            {
                "verified": True,
                "seed": promotion["seed"],
                "selected_checkpoint": promotion["selected_checkpoint"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
