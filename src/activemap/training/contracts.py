"""Training-data contracts that must also be verifiable without PyTorch."""

from __future__ import annotations

from typing import Any

from activemap.features import ONLINE_OBSERVABLE_STATE_CONTRACT


def validate_selector_data_contract(
    samples: list[Any], expected: dict[str, Any] | None
) -> dict[str, str] | None:
    """Fail closed when an online selector manifest lacks its observable contract."""

    if expected is None:
        return None
    if expected != ONLINE_OBSERVABLE_STATE_CONTRACT:
        raise ValueError("unsupported selector data_contract")
    mismatches = [
        sample.sample_id
        for sample in samples
        if sample.metadata.get("online_state_contract") != expected
    ]
    if mismatches:
        preview = ", ".join(mismatches[:3])
        raise ValueError(
            "selector manifest violates the declared online state contract: " + preview
        )
    return dict(expected)
