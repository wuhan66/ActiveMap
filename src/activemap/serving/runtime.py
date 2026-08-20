"""Lazy, truthful runtime for all ActiveMap model backends."""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from typing import Any

import numpy as np

from activemap.agent.records import AgentAction, AgentObservation
from activemap.integrations.rsprompter import (
    RSPrompterRefinementConfig,
    RSPrompterRefiner,
)
from activemap.models import EditOperation
from activemap.selector_records import SelectorSample
from activemap.serving.agent import HuggingFaceAgentPolicy
from activemap.serving.config import DeployedModel, DeploymentConfig, load_deployment_config


def _required_path(path: Path | None, label: str) -> Path:
    if path is None:
        raise RuntimeError(f"deployment manifest is missing {label}")
    return path


class DeploymentRuntime:
    def __init__(self, config: DeploymentConfig) -> None:
        self.config = config
        self._loaded: dict[str, Any] = {}
        self._locks = {name: threading.Lock() for name in config.models}

    @classmethod
    def from_path(cls, path: Path) -> DeploymentRuntime:
        return cls(load_deployment_config(path))

    @staticmethod
    def _dependency(entry: DeployedModel) -> str | None:
        package = {
            "updater": "torch",
            "selector": "torch",
            "operation_selector": "torch",
            "hf_agent": "transformers",
            "rsprompter": None,
        }[entry.kind]
        if package is not None and importlib.util.find_spec(package) is None:
            return package
        return None

    def model_status(self, name: str) -> dict[str, Any]:
        entry = self.config.models[name]
        missing_assets = [str(path) for path in entry.asset_paths() if not path.exists()]
        missing_dependency = self._dependency(entry)
        if not entry.enabled:
            state = "disabled"
        elif missing_assets or missing_dependency:
            state = "unavailable"
        elif name in self._loaded:
            state = "loaded"
        else:
            state = "ready"
        return {
            "name": name,
            "kind": entry.kind,
            "state": state,
            "required": entry.required,
            "device": entry.device,
            "missing_assets": missing_assets,
            "missing_dependency": missing_dependency,
        }

    def status(self) -> dict[str, Any]:
        models = {name: self.model_status(name) for name in self.config.models}
        required_ready = all(
            model["state"] in {"ready", "loaded", "disabled"}
            for model in models.values()
            if model["required"]
        )
        all_ready = all(model["state"] != "unavailable" for model in models.values())
        return {
            "service": self.config.service_name,
            "status": "ready" if all_ready else ("degraded" if required_ready else "unavailable"),
            "models": models,
        }

    def _build(self, entry: DeployedModel) -> Any:
        if entry.kind == "updater":
            from activemap.inference import UpdaterPredictor

            return UpdaterPredictor(
                _required_path(entry.checkpoint, "checkpoint"), device=entry.device
            )
        if entry.kind == "selector":
            from activemap.inference import SelectorPredictor

            return SelectorPredictor(
                _required_path(entry.checkpoint, "checkpoint"), device=entry.device
            )
        if entry.kind == "operation_selector":
            from activemap.inference import OperationSelectorPredictor

            updater_path = entry.options.get("updater_checkpoint")
            return OperationSelectorPredictor(
                _required_path(entry.checkpoint, "checkpoint"),
                updater_checkpoint_path=Path(updater_path) if updater_path else None,
                update_threshold=float(entry.options.get("update_threshold", 0.5)),
                device=entry.device,
            )
        if entry.kind == "hf_agent":
            return HuggingFaceAgentPolicy(
                str(_required_path(entry.model_path, "model_path")),
                device=entry.device,
                max_new_tokens=int(entry.options.get("max_new_tokens", 160)),
                fallback_to_greedy=bool(entry.options.get("fallback_to_greedy", True)),
                enable_thinking=bool(entry.options.get("enable_thinking", False)),
            )
        if entry.kind == "rsprompter":
            return RSPrompterRefiner(
                RSPrompterRefinementConfig(
                    python=_required_path(entry.python, "python"),
                    repository=_required_path(entry.repository, "repository"),
                    model_config=_required_path(entry.config_path, "config_path"),
                    checkpoint=_required_path(entry.checkpoint, "checkpoint"),
                    adapter_script=_required_path(entry.adapter_script, "adapter_script"),
                    work_dir=_required_path(entry.work_dir, "work_dir"),
                    device=entry.device,
                    score_threshold=float(entry.options.get("score_threshold", 0.35)),
                    overlap_threshold=float(entry.options.get("overlap_threshold", 0.10)),
                )
            )
        raise ValueError(f"unsupported model kind: {entry.kind}")

    def get(self, name: str) -> Any:
        if name not in self.config.models:
            raise KeyError(f"unknown deployed model: {name}")
        status = self.model_status(name)
        if status["state"] in {"disabled", "unavailable"}:
            raise RuntimeError(f"model {name} is {status['state']}: {status}")
        if name not in self._loaded:
            with self._locks[name]:
                if name not in self._loaded:
                    self._loaded[name] = self._build(self.config.models[name])
        return self._loaded[name]

    def predict_updater(
        self, image: np.ndarray, prior_mask: np.ndarray, *, model: str = "updater"
    ) -> dict[str, np.ndarray | float]:
        return self.get(model).predict(image, prior_mask)

    def score_selector(self, sample: SelectorSample, *, model: str = "selector") -> np.ndarray:
        return self.get(model).action_scores(sample)

    def predict_operation(
        self,
        image: np.ndarray,
        prior_mask: np.ndarray,
        *,
        model: str = "operation_selector",
    ) -> dict[str, Any]:
        return self.get(model).predict(image, prior_mask)

    def operation_belief(
        self,
        image: np.ndarray,
        prior_mask: np.ndarray,
        *,
        model: str = "operation_selector",
    ) -> Any:
        from activemap.agent.tools import belief_from_operation_prediction

        prediction = self.predict_operation(image, prior_mask, model=model)
        return belief_from_operation_prediction(prediction)

    def act_agent(
        self, observation: AgentObservation, *, model: str = "agent_qwen3_4b"
    ) -> tuple[AgentAction, str, str]:
        policy = self.get(model)
        action = policy.act(observation)
        return action, str(policy.last_source), str(policy.last_raw_output)

    def refine_mask(
        self,
        image: np.ndarray,
        coarse_mask: np.ndarray,
        prior_mask: np.ndarray,
        *,
        edit_type: EditOperation,
        model: str = "rsprompter",
    ) -> np.ndarray:
        return self.get(model).refine(image, coarse_mask, prior_mask, edit_type=edit_type)
