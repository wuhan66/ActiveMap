"""Exercise health, unavailable-model handling, and one structured agent action."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from typing import Any


def _request(url: str, payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--skip-agent", action="store_true")
    args = parser.parse_args()

    health_code, health = _request(f"{args.base_url}/health")
    if health_code != 200:
        raise SystemExit(f"health failed: {health_code} {health}")

    updater_code, updater = _request(
        f"{args.base_url}/v1/updater/predict",
        {"image": [[[0.0] * 16] * 16] * 3, "prior_mask": [[0.0] * 16] * 16},
    )
    if health["models"]["updater"]["state"] == "unavailable" and updater_code != 503:
        raise SystemExit(f"unavailable updater did not return 503: {updater_code}")

    result: dict[str, Any] = {
        "health": health,
        "updater_probe": {"status": updater_code, "response": updater},
    }
    if not args.skip_agent:
        observation = {
            "task_id": "deployment-smoke",
            "split": "test",
            "step": 0,
            "initial_budget": 2.0,
            "remaining_budget": 2.0,
            "spent_cost": 0.0,
            "selected_evidence_ids": [],
            "belief": {
                "edit_probabilities": [0.70, 0.10, 0.10, 0.10],
                "confidence": 0.70,
                "geometry_delta": [0.0] * 8,
                "uncertainty": 0.50,
            },
            "candidates": [
                {
                    "evidence_id": "candidate-1",
                    "cost": 1.0,
                    "selector_score": 0.90,
                    "features": [],
                }
            ],
            "terminal_score": 0.20,
            "available_tools": [],
            "tool_history": [],
        }
        agent_code, agent = _request(
            f"{args.base_url}/v1/agent/action",
            {"observation": observation, "include_raw_output": True},
        )
        if agent_code != 200:
            raise SystemExit(f"agent failed: {agent_code} {agent}")
        result["agent_probe"] = agent
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
