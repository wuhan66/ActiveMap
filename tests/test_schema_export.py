import json
from pathlib import Path

from activemap.schema_export import SCHEMA_MODELS, export_schemas


def test_export_all_registered_schemas(tmp_path: Path) -> None:
    paths = export_schemas(tmp_path)
    assert {path.name for path in paths} == set(SCHEMA_MODELS)
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "$defs" in payload or "properties" in payload
