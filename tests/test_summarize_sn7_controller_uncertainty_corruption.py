import json

import pytest

from scripts.summarize_sn7_controller_uncertainty_corruption import (
    CONTROLLER_KEYS,
    _controller_means,
)


def test_controller_means_reads_seed_summaries(tmp_path) -> None:
    for seed, value in ((1, 0.2), (2, 0.4)):
        path = (
            tmp_path
            / "severity4"
            / f"seed{seed}"
            / "q15"
            / "closed_loop"
        )
        path.mkdir(parents=True)
        metrics = dict.fromkeys(CONTROLLER_KEYS, value)
        (path / "summary.json").write_text(
            json.dumps(
                {
                    "policies": {
                        "edit_utility": {
                            "metrics": metrics,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

    result = _controller_means(tmp_path, 4, "15", seeds=(1, 2))

    assert result["tool_call_episode_rate"] == pytest.approx(0.3)
    assert result["false_edit_rate"] == pytest.approx(0.3)
