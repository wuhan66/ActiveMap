import pytest

from activemap.agent.identifiers import public_evidence_id, resolve_evidence_id


def test_resolve_evidence_id_accepts_raw_and_public_handles():
    raw_ids = ["scene-2017", "scene-2019"]
    assert resolve_evidence_id("scene-2017", raw_ids) == "scene-2017"
    assert resolve_evidence_id(public_evidence_id("scene-2019"), raw_ids) == "scene-2019"


def test_resolve_evidence_id_refuses_unknown_handle():
    with pytest.raises(ValueError, match="unknown or ambiguous"):
        resolve_evidence_id("evidence-deadbeefdeadbeef", ["scene-2019"])
