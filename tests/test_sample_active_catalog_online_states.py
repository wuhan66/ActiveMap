from activemap.models import EditOperation
from activemap.selector_records import SelectorSample
from scripts.sample_active_catalog_online_states import sample_states


def sample():
    return SelectorSample(
        sample_id="sample", split="val", edit_type=EditOperation.RESHAPE,
        hypothesis_features=[0.0] * 16,
        state_features=[1.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.0, 0.0],
        evidence_ids=["e-0", "e-1"],
        evidence_features=[[0.0] * 13, [0.0] * 13],
        evidence_costs=[1.0, 2.0], false_edit_risks=[0.0, 0.0],
        oracle_utilities=[-0.1, 0.4], stop_utility=0.0,
        metadata={
            "source_episode": "episode", "selected_evidence_ids": [],
            "budget": 3.0, "aoi_id": "a", "gt_edit": "RESHAPE",
        },
    )


def test_online_sampler_balances_aois_and_keeps_step_zero(tmp_path):
    rows = []
    for index, aoi in enumerate(("a", "a", "b", "b")):
        base = sample()
        rows.append(base.model_copy(update={
            "sample_id": f"sample-{index}",
            "metadata": {
                **base.metadata, "source_episode": f"episode-{index}",
                "aoi_id": aoi, "oracle_step": 0,
            },
        }))
    source = tmp_path / "states.jsonl"
    source.write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "sampled.jsonl"
    summary = sample_states(source, output, split="val", count=2, seed=7)
    assert summary["selected"] == 2
    assert summary["aoi_count"] == 2
