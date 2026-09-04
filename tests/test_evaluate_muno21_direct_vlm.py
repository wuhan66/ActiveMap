from scripts.evaluate_muno21_direct_vlm import (
    SYSTEM_PROMPTS,
    _operation,
    _prompt_text,
    _terminal_prediction,
)


def test_direct_vlm_terminal_parser_accepts_executable_actions() -> None:
    assert _terminal_prediction('{"action":"REJECT"}')[:2] == ("REJECT", True)
    assert _terminal_prediction('{"action":"COMMIT","edit":"ADD"}')[:2] == (
        "COMMIT:ADD",
        True,
    )
    assert _operation("REJECT") == "KEEP"
    assert _operation("COMMIT:RESHAPE") == "RESHAPE"


def test_direct_vlm_terminal_parser_falls_back_on_tools_or_invalid_json() -> None:
    assert _terminal_prediction('{"action":"USE_TOOL"}')[:2] == ("REJECT", False)
    assert _terminal_prediction("not-json")[:2] == ("REJECT", False)


def test_operational_prompt_defines_conservative_edit_semantics() -> None:
    system = SYSTEM_PROMPTS["operational_v2"]
    normalized = " ".join(system.split())
    assert "unclear, occluded, or ambiguous" in normalized
    assert "absent from the old map" in normalized
    assert "clearly absent from the current RGB image" in normalized
    assert "Do not choose RESHAPE merely" in normalized
    assert "exact old map" in _prompt_text("operational_v2")
    composite = _prompt_text("operational_v2", "composite")
    assert "left is the current aerial RGB image" in composite
    assert "center is the old editable road-map mask" in composite
