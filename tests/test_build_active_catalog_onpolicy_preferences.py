import json

from PIL import Image

from scripts.build_active_catalog_onpolicy_preferences import build_preferences


def _write(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_builds_preference_only_from_matched_executed_states(tmp_path):
    image = tmp_path / "x.png"
    Image.new("RGB", (4, 4)).save(image)
    sft = tmp_path / "train.jsonl"
    index = tmp_path / "index.jsonl"
    _write(sft, [{
        "example_id": "x", "task_id": "task", "split": "train",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {"role": "user", "content": [{"type": "image", "image": str(image)}]},
            {"role": "assistant", "content": [{"type": "text", "text": "{}"}]},
        ],
    }])
    _write(index, [{
        "example_id": "x", "source_episode": "episode", "split": "train",
        "test_assets_read": False,
    }])
    state = {
        "controller_stage": "SELECT", "policy_snapshot": "/adapter",
        "candidate_evidence": [{"evidence_id": "e1", "cost": 0.5}],
    }
    traces = []
    for index_value, selector, utility, false_edit in (
        (0, {"stage": "SELECT", "selection": "STOP"}, 0.0, False),
        (1, {"stage": "SELECT", "selection": "ACQUIRE", "evidence_id": "e1"}, 1.5, True),
    ):
        root = tmp_path / f"rollout-{index_value}"
        root.mkdir()
        trace = root / "traces.jsonl"
        traces.append(trace)
        (root / "summary.json").write_text(json.dumps({
            "split": "train", "test_assets_read": False,
            "protocol": {"stochastic_policy_sampling": True},
            "sources": {"adapter": "/adapter"},
        }), encoding="utf-8")
        _write(trace, [{
            "split": "train", "test_assets_read": False,
            "source_episode": "episode", "budget": 2.0,
            "quality_cost_utility": utility, "terminal_correct": True,
            "false_edit": false_edit, "missed_edit": False,
            "events": [
                {"valid_action": True, "observable_state": state, "selector_action": selector},
                {"valid_action": True, "tool_name": "road_segmentation"},
            ],
        }])
    output = tmp_path / "preferences.jsonl"
    summary = build_preferences(
        traces, sft, index, output, expected_split="train", minimum_margin=0.01
    )
    row = json.loads(output.read_text(encoding="utf-8"))
    assert summary["actions_executed_in_recurrent_environment"] is True
    assert summary["unsupported_nonselector_events_excluded"] == 2
    assert row["chosen_action_key"] == "ACQUIRE:e1"
    assert row["rejected_action_key"] == "STOP"
    assert row["on_policy_executed_recurrent"] is True

    safety_output = tmp_path / "preferences-safety.jsonl"
    safety_summary = build_preferences(
        traces,
        sft,
        index,
        safety_output,
        expected_split="train",
        minimum_margin=0.01,
        preference_mode="safety_first",
    )
    safety_row = json.loads(safety_output.read_text(encoding="utf-8"))
    assert safety_summary["preference_mode"] == "safety_first"
    assert safety_row["chosen_action_key"] == "STOP"
    assert safety_row["rejected_action_key"] == "ACQUIRE:e1"
    assert safety_row["false_edit_rate_gap"] == 1.0
