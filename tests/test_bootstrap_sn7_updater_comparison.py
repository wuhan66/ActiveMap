import json
from pathlib import Path

import pytest

from scripts.bootstrap_sn7_updater_comparison import paired_bootstrap


def _audit(tmp_path: Path, name: str, gain: float) -> Path:
    directory = tmp_path / name
    directory.mkdir()
    (directory / "summary.json").write_text(
        json.dumps({"test_assets_read": False})
    )
    rows = []
    for index, target_edit in enumerate(("KEEP", "ADD", "KEEP", "DELETE")):
        rows.append(
            {
                "sample_id": f"s{index}",
                "aoi_id": f"a{index // 2}",
                "target_edit": target_edit,
                "committed_map_iou": 0.5 + gain,
                "map_iou_delta": gain,
                "change_iou": 0.4 + gain,
                "operation_correct": gain > 0,
                "false_edit": target_edit == "KEEP" and gain < 0,
                "missed_edit": target_edit != "KEEP" and gain < 0,
                "wrong_edit": False,
            }
        )
    (directory / "per_sample.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    return directory


def test_paired_bootstrap_reports_candidate_minus_baseline(tmp_path: Path) -> None:
    baseline = [_audit(tmp_path, f"base{i}", 0.1) for i in range(3)]
    candidate = [_audit(tmp_path, f"candidate{i}", 0.2) for i in range(3)]
    result = paired_bootstrap(
        baseline,
        candidate,
        baseline_name="base",
        candidate_name="candidate",
        draws=100,
        seed=7,
    )
    delta = result["results"]["candidate_minus_base"]["map_iou_delta"]
    assert delta["observed"] == pytest.approx(0.1)


def test_paired_bootstrap_rejects_identity_mismatch(tmp_path: Path) -> None:
    baseline = [_audit(tmp_path, f"base{i}", 0.1) for i in range(3)]
    candidate = [_audit(tmp_path, f"candidate{i}", 0.2) for i in range(3)]
    path = candidate[0] / "per_sample.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["sample_id"] = "different"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="identity mismatch"):
        paired_bootstrap(
            baseline,
            candidate,
            baseline_name="base",
            candidate_name="candidate",
            draws=100,
            seed=7,
        )
