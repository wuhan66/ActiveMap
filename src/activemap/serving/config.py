"""Deployment manifest parsing with path and model-kind validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from activemap.config import load_yaml
from activemap.models import StrictModel


class DeployedModel(StrictModel):
    kind: Literal["updater", "selector", "operation_selector", "hf_agent", "rsprompter"]
    enabled: bool = True
    required: bool = False
    device: str = "cpu"
    checkpoint: Path | None = None
    model_path: Path | None = None
    python: Path | None = None
    repository: Path | None = None
    config_path: Path | None = None
    adapter_script: Path | None = None
    work_dir: Path | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def required_paths_are_declared(self) -> DeployedModel:
        if self.kind in {"updater", "selector", "operation_selector"} and self.checkpoint is None:
            raise ValueError(f"{self.kind} requires checkpoint")
        if self.kind == "hf_agent" and self.model_path is None:
            raise ValueError("hf_agent requires model_path")
        if self.kind == "rsprompter":
            required = (
                self.python,
                self.repository,
                self.config_path,
                self.checkpoint,
                self.adapter_script,
                self.work_dir,
            )
            if any(path is None for path in required):
                raise ValueError("rsprompter requires all isolated backend paths")
        return self

    def asset_paths(self) -> list[Path]:
        values = [
            self.checkpoint,
            self.model_path,
            self.python,
            self.repository,
            self.config_path,
            self.adapter_script,
        ]
        return [path for path in values if path is not None]


class DeploymentConfig(StrictModel):
    service_name: str = "activemap-models"
    host: str = "127.0.0.1"
    port: int = Field(default=8008, ge=1, le=65535)
    models: dict[str, DeployedModel]


def _expanded(payload: Any) -> Any:
    if isinstance(payload, str):
        return os.path.expanduser(os.path.expandvars(payload))
    if isinstance(payload, dict):
        return {key: _expanded(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_expanded(value) for value in payload]
    return payload


def load_deployment_config(path: Path) -> DeploymentConfig:
    return DeploymentConfig.model_validate(_expanded(load_yaml(path)))
