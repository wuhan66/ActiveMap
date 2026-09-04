import json

from scripts.build_vlm_sft_smoke import _action, _select


def _row(example_id: str, action: str) -> dict:
    return {
        "example_id": example_id,
        "messages": [
            {"role": "system", "content": []},
            {
                "role": "user",
                "content": [{"type": "image", "image": "images/example.jpg"}],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"stage": "SELECT", "selection": action}),
                    }
                ],
            },
        ],
    }


def test_selects_one_deterministic_record_per_action(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [_row("z", "STOP"), _row("b", "ACQUIRE"), _row("a", "STOP")]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    selected = _select(path, ("STOP", "ACQUIRE"))
    assert [row["example_id"] for row in selected] == ["a", "b"]
    assert [_action(row) for row in selected] == ["STOP", "ACQUIRE"]
    assert all(
        row["messages"][1]["content"][0]["image"]
        == str((tmp_path / "images" / "example.jpg").resolve())
        for row in selected
    )
