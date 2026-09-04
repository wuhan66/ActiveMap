from scripts.build_agent_safety_preferences import (
    _preference_family,
    _rejected_candidates,
)


def test_reject_state_uses_commit_as_safety_negative() -> None:
    utilities = {
        "REJECT": 0.0,
        "ACQUIRE:evidence-a": -0.01,
        "COMMIT:ADD": -0.75,
        "COMMIT:DELETE": -0.5,
    }

    candidates = _rejected_candidates("REJECT", utilities)

    assert candidates == ["COMMIT:ADD", "COMMIT:DELETE"]
    assert max(candidates, key=utilities.__getitem__) == "COMMIT:DELETE"
    assert _preference_family("REJECT") == "safety"


def test_commit_state_uses_reject_as_update_negative() -> None:
    utilities = {
        "COMMIT:ADD": 0.0,
        "REJECT": -0.75,
        "ACQUIRE:evidence-a": -0.02,
    }

    assert _rejected_candidates("COMMIT:ADD", utilities) == ["REJECT"]
    assert _preference_family("COMMIT:ADD") == "update"


def test_acquire_state_uses_best_terminal_negative() -> None:
    utilities = {
        "ACQUIRE:evidence-a": 0.1,
        "ACQUIRE:evidence-b": 0.05,
        "REJECT": -0.75,
        "COMMIT:ADD": 0.0,
        "COMMIT:DELETE": -0.5,
    }

    candidates = _rejected_candidates("ACQUIRE:evidence-a", utilities)

    assert candidates == ["REJECT", "COMMIT:ADD", "COMMIT:DELETE"]
    assert max(candidates, key=utilities.__getitem__) == "COMMIT:ADD"
    assert _preference_family("ACQUIRE:evidence-a") == "acquire"
