"""Start the ActiveMap model API from a deployment manifest."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=Path, nargs="?", default=Path("configs/deployment/server.yaml")
    )
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    os.environ["ACTIVEMAP_DEPLOY_CONFIG"] = str(args.config.resolve())
    from activemap.serving.config import load_deployment_config

    config = load_deployment_config(args.config)
    import uvicorn

    uvicorn.run(
        "activemap.serving.api:app",
        host=args.host or config.host,
        port=args.port or config.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
