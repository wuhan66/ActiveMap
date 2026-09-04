from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.finalize_sn7_v6_evidence_value_writeback import load_protocol


def _write_protocol(root, checkpoint, checkpoint_hash):
    (root / "protocol.json").write_text(
        json.dumps(
            {
                "schema_version": "sn7-v6-evidence-value-writeback-v1",
                "test_assets_read": False,
                "seeds": [
                    {
                        "seed": seed,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": checkpoint_hash,
                    }
                    for seed in (20260817, 20260818, 20260819)
                ],
            }
        ),
        encoding="utf-8",
    )


def test_load_protocol_rejects_checkpoint_drift(tmp_path):
    checkpoint = tmp_path / "value.pt"
    checkpoint.write_bytes(b"checkpoint")
    _write_protocol(
        tmp_path,
        checkpoint,
        hashlib.sha256(b"different-checkpoint").hexdigest(),
    )

    with pytest.raises(ValueError, match="registered checkpoint changed"):
        load_protocol(tmp_path)


def test_load_protocol_accepts_matching_registered_checkpoints(tmp_path):
    checkpoint = tmp_path / "value.pt"
    checkpoint.write_bytes(b"checkpoint")
    _write_protocol(
        tmp_path,
        checkpoint,
        hashlib.sha256(b"checkpoint").hexdigest(),
    )

    payload = load_protocol(tmp_path)

    assert payload["test_assets_read"] is False


def test_finalizer_entrypoint_imports_project_modules_outside_repo(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "finalize_sn7_v6_evidence_value_writeback.py"

    completed = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "v6"), str(tmp_path / "output")],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "V6 queue is not complete" in completed.stderr
    assert "No module named 'scripts'" not in completed.stderr
