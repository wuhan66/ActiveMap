from argparse import Namespace
from pathlib import Path

from scripts.launch_active_catalog_closed_loop import evaluator_command


def test_evaluator_command_preserves_closed_loop_smoke_limit():
    args = Namespace(
        python="python",
        model=Path("model"),
        adapter=Path("adapter"),
        states=Path("states.jsonl"),
        episodes=Path("episodes.jsonl"),
        val_sft=Path("val.jsonl"),
        val_evaluation_index=Path("index.jsonl"),
        max_candidates=16,
        max_acquisitions=2,
        max_new_tokens=64,
        bootstrap_repetitions=0,
        seed=7,
        limit=2,
    )
    command = evaluator_command(args, Path("out"))
    assert command[1] == "scripts/evaluate_active_catalog_closed_loop.py"
    assert command[command.index("--limit") + 1] == "2"
    assert command[command.index("--tool-mode") + 1] == "none"
    assert command[command.index("--belief-mode") + 1] == "recurrent"
    assert command[command.index("--bootstrap-repetitions") + 1] == "0"
    assert command[command.index("--policy-mode") + 1] == "full_action"
    assert command[command.index("--num-shards") + 1] == "1"
    assert command[command.index("--shard-index") + 1] == "0"


def test_evaluator_command_forwards_frozen_test_authorization():
    args = Namespace(
        python="python",
        model=Path("model"),
        adapter=Path("adapter"),
        states=Path("states.jsonl"),
        episodes=Path("episodes.jsonl"),
        val_sft=Path("prompts.jsonl"),
        val_evaluation_index=Path("index.jsonl"),
        max_candidates=16,
        max_acquisitions=2,
        max_new_tokens=64,
        bootstrap_repetitions=2000,
        seed=7,
        limit=None,
        split="test",
        frozen_test=True,
    )
    command = evaluator_command(args, Path("out"))
    assert command[command.index("--split") + 1] == "test"
    assert "--frozen-test" in command


def test_evaluator_command_forwards_gate_ranker_composition():
    args = Namespace(
        python="python",
        model=Path("model"),
        adapter=Path("adapter"),
        states=Path("states.jsonl"),
        episodes=Path("episodes.jsonl"),
        val_sft=Path("prompts.jsonl"),
        val_evaluation_index=Path("index.jsonl"),
        max_candidates=16,
        max_acquisitions=2,
        max_new_tokens=32,
        bootstrap_repetitions=10,
        seed=7,
        limit=None,
        split="val",
        frozen_test=False,
        policy_mode="gate_ranker",
        ranker_checkpoint=Path("ranker.pt"),
    )
    command = evaluator_command(args, Path("out"))
    assert command[command.index("--policy-mode") + 1] == "gate_ranker"
    assert command[command.index("--ranker-checkpoint") + 1] == "ranker.pt"


def test_evaluator_command_forwards_utility_head_composition():
    args = Namespace(
        python="python",
        model=Path("model"),
        adapter=Path("adapter"),
        states=Path("states.jsonl"),
        episodes=Path("episodes.jsonl"),
        val_sft=Path("prompts.jsonl"),
        val_evaluation_index=Path("index.jsonl"),
        max_candidates=16,
        max_acquisitions=2,
        max_new_tokens=32,
        bootstrap_repetitions=10,
        seed=7,
        limit=4,
        split="val",
        frozen_test=False,
        policy_mode="utility_head_ranker",
        ranker_checkpoint=Path("ranker.pt"),
        utility_head=Path("gate.joblib"),
        utility_head_summary=Path("summary.json"),
    )
    command = evaluator_command(args, Path("out"))
    assert command[command.index("--policy-mode") + 1] == "utility_head_ranker"
    assert command[command.index("--utility-head") + 1] == "gate.joblib"
    assert command[command.index("--utility-head-summary") + 1] == "summary.json"


def test_evaluator_command_forwards_react_model_tool_control():
    args = Namespace(
        python="python",
        model=Path("model"),
        adapter=Path("adapter"),
        states=Path("states.jsonl"),
        episodes=Path("episodes.jsonl"),
        val_sft=Path("prompts.jsonl"),
        val_evaluation_index=Path("index.jsonl"),
        max_candidates=16,
        max_acquisitions=2,
        max_new_tokens=64,
        bootstrap_repetitions=0,
        seed=7,
        limit=2,
        split="val",
        frozen_test=False,
        policy_mode="react",
        tool_mode="model",
        tool_belief_checkpoint=Path("belief.pt"),
        tool_artifact_root=Path("tool-artifacts"),
        tool_out_size=256,
    )
    command = evaluator_command(args, Path("out"))
    assert command[command.index("--policy-mode") + 1] == "react"
    assert command[command.index("--tool-mode") + 1] == "model"
    assert command[command.index("--tool-belief-checkpoint") + 1] == "belief.pt"


def test_evaluator_command_forwards_modern_agent_protocol_modes():
    for mode in ("geommagent_style", "sensesearch_style"):
        args = Namespace(
            python="python",
            model=Path("model"),
            adapter=Path("adapter"),
            states=Path("states.jsonl"),
            episodes=Path("episodes.jsonl"),
            val_sft=Path("prompts.jsonl"),
            val_evaluation_index=Path("index.jsonl"),
            max_candidates=16,
            max_acquisitions=2,
            max_new_tokens=96,
            bootstrap_repetitions=0,
            seed=7,
            limit=2,
            split="val",
            frozen_test=False,
            policy_mode=mode,
            tool_mode="model",
            tool_belief_checkpoint=Path("belief.pt"),
            tool_artifact_root=Path("tool-artifacts"),
            tool_out_size=256,
            asset_root_map=["/old=/new"],
        )

        command = evaluator_command(args, Path("out"))

        assert command[command.index("--policy-mode") + 1] == mode
        assert command[command.index("--tool-mode") + 1] == "model"
        assert command[command.index("--asset-root-map") + 1] == "/old=/new"


def test_evaluator_command_forwards_frozen_prior_belief_ablation():
    args = Namespace(
        python="python",
        model=Path("model"),
        adapter=Path("adapter"),
        states=Path("states.jsonl"),
        episodes=Path("episodes.jsonl"),
        val_sft=Path("prompts.jsonl"),
        val_evaluation_index=Path("index.jsonl"),
        max_candidates=16,
        max_acquisitions=2,
        max_new_tokens=64,
        bootstrap_repetitions=0,
        seed=7,
        limit=2,
        tool_mode="selective",
        belief_mode="frozen_prior",
        tool_belief_checkpoint=Path("belief.pt"),
        tool_artifact_root=Path("tool-artifacts"),
        tool_out_size=256,
        tool_gate=Path("gate.joblib"),
        tool_gate_summary=Path("gate.json"),
    )
    command = evaluator_command(args, Path("out"))
    assert command[command.index("--belief-mode") + 1] == "frozen_prior"


def test_evaluator_command_forwards_always_stop_control():
    args = Namespace(
        python="python",
        model=Path("model"),
        adapter=Path("adapter"),
        states=Path("states.jsonl"),
        episodes=Path("episodes.jsonl"),
        val_sft=Path("prompts.jsonl"),
        val_evaluation_index=Path("index.jsonl"),
        max_candidates=16,
        max_acquisitions=2,
        max_new_tokens=16,
        bootstrap_repetitions=0,
        seed=7,
        limit=2,
        split="val",
        frozen_test=False,
        policy_mode="always_stop",
        tool_mode="none",
    )
    command = evaluator_command(args, Path("out"))
    assert command[command.index("--policy-mode") + 1] == "always_stop"
    assert command[command.index("--tool-mode") + 1] == "none"
