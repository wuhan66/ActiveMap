import hashlib
import json

import pytest

from scripts.audit_true_sequential_rollout import audit


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _row(policy: str, chain: str, step: int, prior: str, executed: str) -> dict:
    return {
        "policy": policy,
        "chain_id": chain,
        "step": step,
        "split": "val",
        "test_assets_read": False,
        "prior_input_sha256": _digest(prior),
        "candidate_frontend_prior_sha256": _digest(prior),
        "direct_hypothesis_prior_sha256": _digest(prior),
        "executed_prior_sha256": _digest(executed),
        "candidate_frontend_receipt_sha256": _digest(f"frontend-{policy}-{chain}-{step}"),
        "direct_hypothesis_receipt_sha256": _digest(f"direct-{policy}-{chain}-{step}"),
    }


def _write(tmp_path, rows):
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_audit_accepts_policy_specific_carried_prior_trace(tmp_path):
    trace = _write(
        tmp_path,
        [
            _row("active", "a", 0, "prior-0", "prior-active-1"),
            _row("active", "a", 1, "prior-active-1", "prior-active-2"),
            _row("stop", "a", 0, "prior-0", "prior-0"),
            _row("stop", "a", 1, "prior-0", "prior-0"),
        ],
    )
    report = audit(trace)
    assert report["passed"] is True
    assert report["policy_count"] == 2
    assert report["transition_count"] == 2
    assert report["changed_prior_transition_count"] == 1


def test_audit_accepts_explicit_train_diagnostic(tmp_path):
    rows = [
        _row("active", "a", 0, "prior-0", "prior-1"),
        _row("active", "a", 1, "prior-1", "prior-2"),
        _row("stop", "a", 0, "prior-0", "prior-0"),
        _row("stop", "a", 1, "prior-0", "prior-0"),
    ]
    for row in rows:
        row["split"] = "train"

    report = audit(_write(tmp_path, rows), expected_split="train")

    assert report["passed"] is True
    assert report["split"] == "train"


def test_audit_rejects_test_split_request(tmp_path):
    with pytest.raises(ValueError, match="train/validation"):
        audit(_write(tmp_path, []), expected_split="test")


def test_audit_rejects_reused_frontend_on_changed_prior(tmp_path):
    rows = [
        _row("active", "a", 0, "prior-0", "prior-1"),
        _row("active", "a", 1, "prior-1", "prior-2"),
        _row("stop", "a", 0, "prior-0", "prior-0"),
        _row("stop", "a", 1, "prior-0", "prior-0"),
    ]
    rows[1]["candidate_frontend_prior_sha256"] = _digest("prior-0")
    with pytest.raises(ValueError, match="reused a frontend"):
        audit(_write(tmp_path, rows))


def test_audit_rejects_next_step_not_using_previous_executed_prior(tmp_path):
    rows = [
        _row("active", "a", 0, "prior-0", "prior-1"),
        _row("active", "a", 1, "wrong-prior", "prior-2"),
        _row("stop", "a", 0, "prior-0", "prior-0"),
        _row("stop", "a", 1, "prior-0", "prior-0"),
    ]
    with pytest.raises(ValueError, match="preceding executed prior"):
        audit(_write(tmp_path, rows))
