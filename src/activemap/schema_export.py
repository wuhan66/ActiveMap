"""Export version-controlled JSON Schemas from the canonical Pydantic records."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from activemap.agent.records import AgentTrajectory
from activemap.evaluation.update import UpdatePrediction
from activemap.geo_tools.records import GeoToolCall, GeoToolResult
from activemap.models import DecisionRecord, EditRecord, EpisodeRecord
from activemap.selector_records import SelectorSample
from activemap.updater_records import UpdaterSample

SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "edit.schema.json": EditRecord,
    "episode.schema.json": EpisodeRecord,
    "decision.schema.json": DecisionRecord,
    "selector-sample.schema.json": SelectorSample,
    "updater-sample.schema.json": UpdaterSample,
    "update-prediction.schema.json": UpdatePrediction,
    "agent-trajectory.schema.json": AgentTrajectory,
    "geo-tool-call.schema.json": GeoToolCall,
    "geo-tool-result.schema.json": GeoToolResult,
}


def export_schemas(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, model in SCHEMA_MODELS.items():
        path = output_dir / name
        path.write_text(
            json.dumps(model.model_json_schema(), indent=2) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    return written
