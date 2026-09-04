from types import SimpleNamespace

from activemap.agent.identifiers import public_evidence_id
from scripts.build_reachable_tool_sft import _match_pair, _public_selected_ids


def test_reachable_builder_resolves_selected_evidence_outside_candidate_list():
    sample = SimpleNamespace(
        evidence_ids=["scene-candidate"],
        metadata={
            "initial_evidence_id": "scene-initial",
            "selected_evidence_ids": ["scene-initial", "scene-selected"],
        },
    )
    pair = SimpleNamespace(
        task_id="task-1",
        example_id="pair-1",
        evidence_id=public_evidence_id("scene-selected"),
        metadata={"initial_evidence_id": public_evidence_id("scene-initial")},
    )

    assert _public_selected_ids(sample) == {
        public_evidence_id("scene-initial"),
        public_evidence_id("scene-selected"),
    }
    matched = _match_pair(pair, {"task-1": [sample]})
    assert matched == (sample, public_evidence_id("scene-selected"))
