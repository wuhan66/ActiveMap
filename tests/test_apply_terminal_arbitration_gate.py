from scripts.apply_terminal_arbitration_gate import passes_gate


def test_gate_directions_include_threshold() -> None:
    assert passes_gate(0.5, "ge", 0.5)
    assert passes_gate(0.5, "le", 0.5)
    assert not passes_gate(0.4, "ge", 0.5)
    assert not passes_gate(0.6, "le", 0.5)
