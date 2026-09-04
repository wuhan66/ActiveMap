import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_policy_relative_staleness.py"


def _state(sample_id: str, utilities: list[float]) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "evidence_ids": ["candidate-a", "candidate-b"],
        "oracle_utilities": utilities,
        "evidence_costs": [0.2, 0.3],
        "false_edit_risks": [0.1, 0.2],
        "stop_utility": 0.0,
    }


def test_audit_expands_selector_oracle_candidates(tmp_path: Path) -> None:
    old_path = tmp_path / "old.jsonl"
    new_path = tmp_path / "new.jsonl"
    old_path.write_text(json.dumps(_state("episode-1__b1p5__s0", [0.2, -0.1])) + "\n")
    new_path.write_text(json.dumps(_state("episode-1__b1p5__s0", [-0.2, 0.1])) + "\n")
    output = tmp_path / "audit"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--old-traces",
            str(old_path),
            "--new-traces",
            str(new_path),
            "--oracle-step",
            "0",
            "--output-dir",
            str(output),
        ],
        check=True,
    )

    summary = json.loads((output / "summary.json").read_text())
    assert summary["record_schema"] == "selector-oracle-state"
    assert summary["oracle_step_filter"] == 0
    assert summary["matched_examples"] == 2
    assert summary["label_transitions"]["positive_to_nonpositive"] == 1
    assert summary["label_transitions"]["nonpositive_to_positive"] == 1
