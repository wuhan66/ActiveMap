import json
from types import SimpleNamespace

from activemap.agent.records import AgentBelief
from activemap.geo_tools.records import GeoToolName, GeoToolResult
from activemap.models import EditOperation
from scripts import build_semantic_vlm_rollout_eval as module


def _row(example_id, operation):
    return SimpleNamespace(
        example_id=example_id,
        task_id=f"task-{example_id}",
        evidence_id=f"evidence-{example_id}",
        gt_edit=EditOperation.ADD,
        split="val",
        post_acquisition_belief=AgentBelief(
            edit_probabilities=[0.8, 0.1, 0.05, 0.05],
            confidence=0.8,
            uncertainty=0.2,
        ),
        semantic_result=GeoToolResult(
            call_id="semantic",
            tool=GeoToolName.RASTER_SEGMENT,
            success=True,
            outputs={"gated_edit": operation.value},
            cost=0.75,
        ),
    )


def test_rollout_builder_emits_post_state_for_negative_tool_target(tmp_path, monkeypatch):
    row = _row("negative", EditOperation.KEEP)
    monkeypatch.setattr(
        module,
        "_index_rows",
        lambda paths: ([row.example_id], [{row.example_id: row}] * 3),
    )
    monkeypatch.setattr(module, "_sha256", lambda path: "sha")
    visual_root = tmp_path / "images"
    visual_root.mkdir()
    (visual_root / f"{row.task_id}.jpg").write_bytes(b"image")
    output = tmp_path / "rollout"
    summary = module.build_rollout_dataset(
        [tmp_path / "a", tmp_path / "b", tmp_path / "c"],
        visual_root,
        output,
        minimum_votes=2,
    )
    records = [json.loads(line) for line in (output / "rollout.jsonl").read_text().splitlines()]
    assert summary["complete_pre_post_pairs"] is True
    assert [record["stage"] for record in records] == ["PRE_TOOL", "POST_TOOL"]
    assert records[0]["oracle_use_tool"] is False
