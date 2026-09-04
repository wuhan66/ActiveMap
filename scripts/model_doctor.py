"""Check deployment assets and optionally require selected models to be ready."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from activemap.serving.runtime import DeploymentRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=Path, nargs="?", default=Path("configs/deployment/server.yaml")
    )
    parser.add_argument("--require", action="append", default=[])
    args = parser.parse_args()
    runtime = DeploymentRuntime.from_path(args.config)
    status = runtime.status()
    print(json.dumps(status, indent=2))
    failed = [
        name
        for name in args.require
        if name not in status["models"]
        or status["models"][name]["state"] not in {"ready", "loaded"}
    ]
    if failed:
        raise SystemExit(f"required models are not ready: {', '.join(failed)}")


if __name__ == "__main__":
    main()
