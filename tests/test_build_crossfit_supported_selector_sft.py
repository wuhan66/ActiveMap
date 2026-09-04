import json
import subprocess
import sys
from pathlib import Path


def _selector_row(trajectory_id: str, selection: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {"role": "user", "content": [{"type": "text", "text": "observation"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"stage": "SELECT", "selection": selection}),
                    }
                ],
            },
        ],
        "trajectory_id": trajectory_id,
        "task_id": f"task-{trajectory_id}",
        "split": "train",
        "stage": "SELECT",
        "selected_tool": selection == "ACQUIRE",
        "policy_relative_advantage": 1.0 if selection == "ACQUIRE" else -0.75,
    }


def test_restores_primary_labels_and_attaches_crossfit_reliability(tmp_path):
    manifest = tmp_path / "selector_sft.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(row)
            for row in (_selector_row("seq-a", "STOP"), _selector_row("seq-b", "STOP"))
        )
        + "\n",
        encoding="utf-8",
    )
    branches = tmp_path / "branches"
    branches.mkdir()
    (branches / "summary.json").write_text(json.dumps({"test_assets_read": False}))
    (branches / "traces.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "example_id": "a",
                    "split": "train",
                    "onpolicy_advantage": 1.0,
                    "crossfit_advantage": -0.75,
                },
                {
                    "example_id": "b",
                    "split": "train",
                    "onpolicy_advantage": -0.75,
                    "crossfit_advantage": -0.75,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "supported.jsonl"
    script = Path(__file__).parents[1] / "scripts" / "build_crossfit_supported_selector_sft.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            str(manifest),
            str(branches),
            str(output),
            "--split",
            "train",
            "--disagreement-weight",
            "0.5",
        ],
        check=True,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    first_target = json.loads(rows[0]["messages"][-1]["content"][0]["text"])
    assert first_target["selection"] == "ACQUIRE"
    assert rows[0]["policy_relative_advantage"] == 1.0
    assert rows[0]["crossfit_reliability_weight"] == 0.5
    assert rows[1]["crossfit_reliability_weight"] == 1.0
    summary = json.loads(output.with_suffix(".jsonl.summary.json").read_text(encoding="utf-8"))
    assert summary["changed_label_count"] == 1
    assert summary["crossfit_agreement_counts"] == {"agree": 1, "disagree": 1}
