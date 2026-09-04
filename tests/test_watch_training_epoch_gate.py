import json

from scripts.watch_training_epoch_gate import _read_json_if_ready, _write


def test_write_replaces_manifest_atomically(tmp_path):
    path = tmp_path / "gate.json"
    _write(path, {"status": "running"})
    _write(path, {"status": "complete"})
    assert json.loads(path.read_text()) == {"status": "complete"}
    assert not path.with_suffix(".json.tmp").exists()


def test_read_json_if_ready_tolerates_partial_writer_state(tmp_path):
    path = tmp_path / "run_state.json"
    path.write_text("", encoding="utf-8")
    assert _read_json_if_ready(path) is None
    path.write_text('{"status":', encoding="utf-8")
    assert _read_json_if_ready(path) is None
    path.write_text('{"status":"running"}', encoding="utf-8")
    assert _read_json_if_ready(path) == {"status": "running"}
