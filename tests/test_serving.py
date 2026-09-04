from pathlib import Path

import pytest
import yaml

from activemap.serving.agent import _first_json_object
from activemap.serving.config import load_deployment_config
from activemap.serving.runtime import DeploymentRuntime


def test_deployment_config_expands_environment_and_reports_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACTIVE_MODEL_ROOT", str(tmp_path))
    config_path = tmp_path / "deployment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "updater": {
                        "kind": "updater",
                        "checkpoint": "$ACTIVE_MODEL_ROOT/missing.pt",
                        "enabled": True,
                    },
                    "agent": {
                        "kind": "hf_agent",
                        "model_path": "$ACTIVE_MODEL_ROOT/agent",
                        "enabled": False,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    runtime = DeploymentRuntime(load_deployment_config(config_path))
    status = runtime.status()
    assert status["status"] == "degraded"
    assert status["models"]["updater"]["state"] == "unavailable"
    assert status["models"]["agent"]["state"] == "disabled"


def test_agent_json_parser_ignores_surrounding_text() -> None:
    payload = _first_json_object('thinking... {"action":"REJECT"} trailing')
    assert payload == {"action": "REJECT"}


def test_invalid_deployment_manifest_rejects_missing_kind_path(tmp_path: Path) -> None:
    config_path = tmp_path / "deployment.yaml"
    config_path.write_text(
        yaml.safe_dump({"models": {"agent": {"kind": "hf_agent"}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="model_path"):
        load_deployment_config(config_path)
