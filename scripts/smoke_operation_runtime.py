#!/usr/bin/env python3
"""Run one real deployment-manifest operation-selector inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from activemap.agent.tools import belief_from_operation_prediction
from activemap.serving.runtime import DeploymentRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--model", default="operation_selector")
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    runtime = DeploymentRuntime.from_path(args.config)
    prediction = runtime.predict_operation(
        np.zeros((args.size, args.size, 3), dtype=np.uint8),
        np.zeros((args.size, args.size), dtype=np.float32),
        model=args.model,
    )
    belief = belief_from_operation_prediction(prediction)
    summary = {
        "model_status": runtime.model_status(args.model),
        "mask_shape": list(np.asarray(prediction["mask_probability"]).shape),
        "edit_probabilities": np.asarray(prediction["edit_probabilities"]).tolist(),
        "predicted_edit": prediction["predicted_edit"],
        "gated_edit": prediction["gated_edit"],
        "update_probability": prediction["update_probability"],
        "update_threshold": prediction["update_threshold"],
        "belief": belief.model_dump(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
