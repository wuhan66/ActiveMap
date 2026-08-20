"""FastAPI application exposing all deployed ActiveMap model backends."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import Field

from activemap.agent.records import AgentObservation
from activemap.models import EditOperation, StrictModel
from activemap.selector_records import SelectorSample
from activemap.serving.runtime import DeploymentRuntime


class UpdaterRequest(StrictModel):
    image: list[Any]
    prior_mask: list[Any]
    model: str = "updater"


class RefinementRequest(StrictModel):
    image: list[Any]
    coarse_mask: list[Any]
    prior_mask: list[Any]
    edit_type: EditOperation
    model: str = "rsprompter"


class SelectorRequest(StrictModel):
    sample: SelectorSample
    model: str = "selector"


class AgentRequest(StrictModel):
    observation: AgentObservation
    model: str = "agent_qwen3_4b"
    include_raw_output: bool = False


class ArrayResponse(StrictModel):
    values: list[Any]
    shape: list[int] = Field(min_length=1)


def _service_error(exc: Exception) -> HTTPException:
    if isinstance(exc, (KeyError, RuntimeError, FileNotFoundError)):
        return HTTPException(status_code=503, detail=str(exc))
    return HTTPException(status_code=422, detail=str(exc))


def create_app(config_path: Path | None = None) -> FastAPI:
    resolved = config_path or Path(
        os.environ.get("ACTIVEMAP_DEPLOY_CONFIG", "configs/deployment/server.yaml")
    )
    runtime = DeploymentRuntime.from_path(resolved)
    app = FastAPI(title="ActiveMap Models", version="0.1.0")
    app.state.runtime = runtime
    app.state.config_path = str(resolved.resolve())

    @app.get("/health")
    def health() -> dict[str, Any]:
        return runtime.status()

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return runtime.status()["models"]

    @app.post("/v1/updater/predict")
    def updater_predict(request: UpdaterRequest) -> dict[str, Any]:
        try:
            result = runtime.predict_updater(
                np.asarray(request.image, dtype=np.float32),
                np.asarray(request.prior_mask, dtype=np.float32),
                model=request.model,
            )
        except Exception as exc:
            raise _service_error(exc) from exc
        return {
            "mask_probability": np.asarray(result["mask_probability"]).tolist(),
            "edit_probabilities": np.asarray(result["edit_probabilities"]).tolist(),
            "geometry_delta": np.asarray(result["geometry_delta"]).tolist(),
            "confidence": float(result["confidence"]),
        }

    @app.post("/v1/selector/score")
    def selector_score(request: SelectorRequest) -> dict[str, Any]:
        try:
            scores = runtime.score_selector(request.sample, model=request.model)
        except Exception as exc:
            raise _service_error(exc) from exc
        return {"scores": scores.tolist(), "action_count": len(scores)}

    @app.post("/v1/agent/action")
    def agent_action(request: AgentRequest) -> dict[str, Any]:
        try:
            action, source, raw_output = runtime.act_agent(
                request.observation, model=request.model
            )
        except Exception as exc:
            raise _service_error(exc) from exc
        response: dict[str, Any] = {
            "action": action.model_dump(mode="json"),
            "source": source,
        }
        if request.include_raw_output:
            response["raw_output"] = raw_output
        return response

    @app.post("/v1/rsprompter/refine")
    def rsprompter_refine(request: RefinementRequest) -> dict[str, Any]:
        try:
            mask = runtime.refine_mask(
                np.asarray(request.image, dtype=np.float32),
                np.asarray(request.coarse_mask, dtype=np.float32),
                np.asarray(request.prior_mask, dtype=np.float32),
                edit_type=request.edit_type,
                model=request.model,
            )
        except Exception as exc:
            raise _service_error(exc) from exc
        return {"mask": mask.tolist(), "shape": list(mask.shape)}

    return app


app = create_app()
