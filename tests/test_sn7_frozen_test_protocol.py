import json
from pathlib import Path

from scripts.evaluate_active_catalog_closed_loop import image_index


def test_label_free_visual_index_resolves_relative_images(tmp_path: Path) -> None:
    image = tmp_path / "images" / "a.jpg"
    image.parent.mkdir()
    image.write_bytes(b"fixture")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        json.dumps(
            {
                "example_id": "x",
                "split": "val",
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": "images/a.jpg"}],
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    index = tmp_path / "index.jsonl"
    index.write_text(
        json.dumps({"example_id": "x", "split": "val", "source_episode": "episode"})
        + "\n",
        encoding="utf-8",
    )
    assert image_index(prompts, index)["episode"] == image.resolve()


def test_frozen_suite_is_one_shot_and_records_negative_results() -> None:
    text = Path("scripts/run_sn7_frozen_test_suite.sh").read_text(encoding="utf-8")
    assert "assert_frozen_test_access" in text
    assert "--split test --frozen-test" in text
    assert "--record-failed-upstream" in text
    assert "refusing to reuse frozen test root" in text
    assert "refusing" in text and "occupied" in text
    assert "rm -" not in text


def test_sn7_launchers_preserve_optional_agent_overlay() -> None:
    for path in (
        Path("scripts/run_sn7_active_catalog_qwen.sh"),
        Path("scripts/run_sn7_frozen_test_suite.sh"),
    ):
        text = path.read_text(encoding="utf-8")
        assert 'export PYTHONPATH="src:.:${PYTHONPATH:-}"' in text


def test_sn7_preflight_handles_a_new_run_root() -> None:
    text = Path("scripts/run_sn7_active_catalog_qwen.sh").read_text(encoding="utf-8")
    assert 'df -Pk "${RUN_ROOT}" 2>/dev/null' in text
    assert "awk 'NR==2 {print $4}' || true" in text
