from argparse import Namespace
from pathlib import Path

from scripts.launch_gate_ranker_baseline_matrix import (
    DEPLOYMENT_POLICIES,
    WRITEBACK_POLICIES,
    build_stages,
    finalize_matrix,
)


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        python="python",
        states=Path("states.jsonl"),
        episodes=Path("episodes.jsonl"),
        ranker_checkpoint=Path("ranker.pt"),
        updater_checkpoint=Path("updater.pt"),
        candidate_closed_loop=Path("candidate_loop.jsonl"),
        candidate_writeback=Path("candidate_writeback.jsonl"),
        output_root=tmp_path,
        gpu=4,
        seed=7,
        max_candidates=16,
        max_acquisitions=2,
        bootstrap_repetitions=20,
        image_size=512,
        threshold=0.5,
        asset_root_map=["/old=/new"],
        learned_selector=[
            "generic_selector=generic.pt",
            "edit_conditioned_selector=edit.pt",
        ],
        limit=3,
    )


def test_matrix_stages_use_ranker_and_same_updater(tmp_path: Path):
    stages = build_stages(_args(tmp_path))
    baseline = stages[0].command
    assert baseline[baseline.index("--ranker-checkpoint") + 1] == "ranker.pt"
    assert baseline[baseline.index("--episodes") + 1] == "episodes.jsonl"
    writebacks = [stage for stage in stages if stage.name.startswith("writeback_")]
    assert len(writebacks) == len(WRITEBACK_POLICIES) + 2
    assert all(stage.command[2] == "updater.pt" for stage in writebacks)
    comparisons = [
        stage for stage in stages if stage.name.startswith("compare_writeback_")
    ]
    assert len(comparisons) == len(WRITEBACK_POLICIES) + 2
    assert all("aoi_id" in stage.command for stage in comparisons)
    assert baseline.count("--learned-selector") == 2


def _interval(low: float = 0.01, high: float = 0.02, delta: float = 0.02):
    return {"ci95_low": low, "ci95_high": high, "delta": delta}


def test_finalizer_selects_strongest_deployment_baseline(tmp_path: Path):
    comparisons = tmp_path / "comparisons"
    comparisons.mkdir()
    intervals = {
        "mean_quality_cost_utility": _interval(),
        "false_edit_rate": _interval(-0.01, 0.01),
        "terminal_accuracy": _interval(0.0, 0.1),
    }
    closed = {
        "paired_aoi_comparisons": {
            f"gate_ranker_minus_{policy}": {"intervals": intervals}
            for policy in (*DEPLOYMENT_POLICIES, "shortlist_oracle_upper_bound")
        }
    }
    (comparisons / "closed_loop.json").write_text(
        __import__("json").dumps(closed), encoding="utf-8"
    )
    for index, policy in enumerate(WRITEBACK_POLICIES):
        utility = 0.01 * index
        payload = {
            "group_key": "aoi_id",
            "baseline": {"episode_utility_v2_balanced_auc": utility},
            "candidate": {"episode_utility_v2_balanced_auc": 0.2},
            "paired_delta": {
                "raster_iou_gain_auc": _interval(),
                "vector_replay_iou_auc": _interval(delta=0.02),
                "vector_delta_topology_valid_auc": _interval(-0.005, 0.01),
                "episode_utility_v2_balanced_auc": _interval(),
                "episode_utility_v2_safety_auc": _interval(-0.005, 0.02),
                "episode_utility_v2_cost_aware_auc": _interval(-0.005, 0.02),
                "false_edit_auc": _interval(-0.01, 0.01),
            },
        }
        (comparisons / f"writeback_vs_{policy}.json").write_text(
            __import__("json").dumps(payload), encoding="utf-8"
        )

    result = finalize_matrix(tmp_path)

    assert result["strongest_deployment_baseline"] == DEPLOYMENT_POLICIES[-1]
    assert result["promotion"]["promote"] is True
    assert result["oracle_is_upper_bound_only"] is True
