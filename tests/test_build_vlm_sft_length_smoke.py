import json

from scripts.build_vlm_sft_length_smoke import _first_example_id, _load_example


def test_load_example_resolves_relative_image(tmp_path):
    source = tmp_path / "train.jsonl"
    row = {
        "example_id": "longest",
        "messages": [
            {"role": "system", "content": []},
            {
                "role": "user",
                "content": [{"type": "image", "image": "images/a.jpg"}],
            },
            {"role": "assistant", "content": []},
        ],
    }
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    loaded = _load_example(source, "longest")
    assert loaded["messages"][1]["content"][0]["image"] == str(
        (tmp_path / "images" / "a.jpg").resolve()
    )


def test_first_example_id_skips_blank_lines(tmp_path):
    source = tmp_path / "val.jsonl"
    source.write_text('\n{"example_id":"first"}\n', encoding="utf-8")
    assert _first_example_id(source) == "first"
