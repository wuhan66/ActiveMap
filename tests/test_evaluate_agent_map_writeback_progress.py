import json
from pathlib import Path

from scripts.evaluate_agent_map_writeback import _write_progress


def test_write_progress_replaces_atomically(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    _write_progress(path, {"status": "running", "rows_processed": 25})
    _write_progress(path, {"status": "complete", "rows_processed": 50})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "status": "complete",
        "rows_processed": 50,
    }
    assert list(tmp_path.glob("progress.json.tmp.*")) == []
