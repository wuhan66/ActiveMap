import json
from pathlib import Path

import pytest

from scripts.finalize_sn7_executable_selector_sweep import finalize


def test_finalize_refuses_incomplete_sweep(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="incomplete"):
        finalize(tmp_path)


def test_finalize_requires_four_successful_jobs(tmp_path: Path) -> None:
    (tmp_path / "process_results.json").write_text(
        json.dumps([{"returncode": 0}]), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="four sweep jobs"):
        finalize(tmp_path)
