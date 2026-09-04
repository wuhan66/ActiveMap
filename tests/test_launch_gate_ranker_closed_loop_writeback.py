from argparse import Namespace
from pathlib import Path

import pytest

from scripts.launch_gate_ranker_closed_loop_writeback import (
    build_stages,
    resolve_promoted_adapter,
)


def _write_json(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def test_resolve_promoted_adapter_from_selected_trace(tmp_path: Path):
    run_root = tmp_path / "family" / "seed7"
    trace = run_root / "active_catalog_gate_ranker_val" / "traces.jsonl"
    _write_json(trace, "{}\n")
    decision = tmp_path / "round" / "promotion_decision.json"
    _write_json(
        decision,
        '{"test_assets_read":false,"selected_tuning_run":"t10-s1",'
        '"selected_acquire_target":0.1,"promotion":{"passed":true}}',
    )
    _write_json(
        decision.with_name("report.json"),
        '{"runs":{"t10-s1":{"trace":"' + str(trace).replace("\\", "\\\\") + '"}}}',
    )

    _, adapter = resolve_promoted_adapter(decision)

    assert adapter == run_root / "final"


def test_rejects_unpromoted_round(tmp_path: Path):
    decision = tmp_path / "promotion_decision.json"
    _write_json(
        decision,
        '{"test_assets_read":false,"promotion":{"passed":false}}',
    )

    with pytest.raises(PermissionError, match="did not pass"):
        resolve_promoted_adapter(decision)


def test_pipeline_composes_gate_ranker_and_executable_writeback(tmp_path: Path):
    args = Namespace(
        python="python",
        source_states=Path("states.jsonl"),
        source_episodes=Path("episodes.jsonl"),
        bundle_root=tmp_path / "bundle",
        output_root=tmp_path / "output",
        model=Path("model"),
        val_sft=Path("val.jsonl"),
        val_evaluation_index=Path("index.jsonl"),
        ranker_checkpoint=Path("ranker.pt"),
        updater_checkpoint=Path("updater.pt"),
        gpu=4,
        seed=7,
        max_candidates=16,
        max_acquisitions=2,
        bootstrap_repetitions=20,
        image_size=512,
        threshold=0.5,
        asset_root_map=["/old=/new"],
        limit=3,
    )

    stages = build_stages(args, Path("adapter"))
    closed = next(stage.command for stage in stages if stage.name == "closed_loop")
    writeback = next(
        stage.command for stage in stages if stage.name == "executable_writeback"
    )

    assert closed[closed.index("--policy-mode") + 1] == "gate_ranker"
    assert closed[closed.index("--ranker-checkpoint") + 1] == "ranker.pt"
    assert closed[closed.index("--limit") + 1] == "3"
    assert writeback[writeback.index("--asset-root-map") + 1] == "/old=/new"
    assert writeback[writeback.index("--protocol-name") + 1].endswith("writeback-v2")
