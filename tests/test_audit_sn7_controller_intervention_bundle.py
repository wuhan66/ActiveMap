import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_sn7_controller_intervention_bundle import audit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")


def _bundle(root: Path) -> None:
    inputs = {}
    for name in ("benefit_vs_notool", "benefit_vs_forced", "controller"):
        relative = f"inputs/4px/{name}.json"
        path = root / relative
        _write(path, {"name": name})
        inputs[name] = {
            "path": f"/server/{name}.json",
            "bundle_path": relative,
            "sha256": _sha256(path),
        }
    _write(
        root / "summary.json",
        {
            "rows": [{"condition": "4px", "inputs": inputs}],
            "protocol": {"test_assets_read": False},
        },
    )
    _write(
        root / "claim_gate.json",
        {
            "checks": [{"condition": "4px", "strict_gate_passed": True}],
            "all_conditions_strict": True,
            "test_assets_read": False,
        },
    )
    for name in (
        "table.csv",
        "table.md",
        "table.tex",
        "intervention_summary.png",
        "intervention_summary.pdf",
        "intervention_summary.svg",
    ):
        _write(root / name, name.encode())
    files = sorted(path for path in root.rglob("*") if path.is_file())
    _write(
        root / "manifest.json",
        {
            "schema_version": "sn7-controller-intervention-manifest-v1",
            "files": {
                path.relative_to(root).as_posix(): {
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in files
            },
            "test_assets_read": False,
        },
    )


def test_bundle_audit_detects_hash_tampering(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _bundle(bundle)

    report = audit(bundle)
    assert report["passed"] is True
    assert report["input_count"] == 3

    (bundle / "table.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        audit(bundle)
