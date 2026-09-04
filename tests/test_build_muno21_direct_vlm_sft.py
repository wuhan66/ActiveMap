from scripts.build_muno21_direct_vlm_sft import _balance, build_rows


def _input(example_id: str) -> dict:
    return {
        "example_id": example_id,
        "split": "train",
        "images": {"composite": f"/tmp/{example_id}.png"},
    }


def test_balance_cycles_minority_classes_deterministically() -> None:
    records = [
        (_input("k1"), "KEEP"),
        (_input("k2"), "KEEP"),
        (_input("a1"), "ADD"),
        (_input("d1"), "DELETE"),
        (_input("r1"), "RESHAPE"),
    ]
    balanced = _balance(records)
    assert len(balanced) == 8
    assert [operation for _, operation, _ in balanced].count("ADD") == 2
    assert [repeat for record, operation, repeat in balanced if operation == "ADD"] == [0, 1]


def test_build_rows_uses_one_composite_and_executable_actions() -> None:
    inputs = [_input("k1"), _input("a1")]
    labels = [
        {"example_id": "k1", "split": "train", "edit": "KEEP"},
        {"example_id": "a1", "split": "train", "edit": "ADD"},
    ]
    rows = build_rows(inputs, labels, split="train", balance=False)
    assert rows[0]["messages"][1]["content"][0]["image"] == "/tmp/k1.png"
    assert rows[0]["messages"][2]["content"][0]["text"] == '{"action":"REJECT"}'
    assert rows[1]["messages"][2]["content"][0]["text"] == (
        '{"action":"COMMIT","edit":"ADD"}'
    )
    assert all(row["test_assets_read"] is False for row in rows)
