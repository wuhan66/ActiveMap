import hashlib
import json

import pytest

from scripts.aggregate_sn7_learned_defer_extension import POLICY, SEEDS, aggregate


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(policy):
    value = 1.0 if policy == "activemap" else 0.0
    return "".join(
        json.dumps(
            {
                "source_episode": f"episode-{aoi}",
                "budget": 3.0,
                "aoi_id": f"aoi-{aoi}",
                "split": "val",
                "test_assets_read": False,
                "terminal_correct": value,
                "false_edit": 0.0,
                "missed_edit": 0.0,
                "spent_cost": 0.0,
                "quality_gain": value,
                "quality_cost_utility": value,
                "episode_utility_v2_proxy_balanced": value,
                "episode_utility_v2_proxy_safety": value,
                "episode_utility_v2_proxy_cost_aware": value,
            }
        )
        + "\n"
        for aoi in range(2)
    )


def _receipt(root, seed, *, condition_on_hypothesis=False):
    traces = {}
    for key in ("natural_validation_traces", "r2_filtered_traces"):
        traces[key] = {}
        for policy in ("activemap", POLICY):
            path = root / f"seed{seed}" / key / f"{policy}.jsonl"
            traces[key][policy] = {
                "path": str(path.resolve()),
                "sha256": _write(path, _rows(policy)),
            }
    component = root / f"seed{seed}" / "gate.bin"
    component_hash = _write(component, "gate")
    payload = {
        "schema_version": "activemap-sn7-learned-defer-extension-seed-v1",
        "split": "val",
        "test_assets_read": False,
        "controller_seed": seed,
        "comparison_contract": {
            "only_selector_conditioning_differs": True,
            "same_architecture_training_budget_and_seed": True,
            "same_recurrent_belief_tool_gate_safe_commit": True,
        },
        "inputs": {
            "val_states": {"sha256": "states"},
            "val_episodes": {"sha256": "episodes"},
            "edit_manifest": {"sha256": "manifest"},
        },
        "selectors": {
            POLICY: {
                "condition_on_hypothesis": condition_on_hypothesis,
                "stop_margin_source": "checkpoint",
            }
        },
        "shared_controller_components": {
            "tool_gate": {"path": str(component.resolve()), "sha256": component_hash}
        },
        **traces,
    }
    _write(root / f"seed{seed}" / "COMPLETE.json", json.dumps(payload))


def test_aggregate_reports_activemap_minus_learned_defer_for_both_slices(tmp_path):
    for seed in SEEDS:
        _receipt(tmp_path, seed)
    result = aggregate(tmp_path, repetitions=10, seed=1)
    assert set(result["results"]) == {"r1_natural", "r2_edit_only"}
    comparison = result["results"]["r1_natural"]["activemap_minus_learned_defer"]
    assert comparison["intervals"]["balanced_utility"]["observed_delta"] == 1.0


def test_aggregate_rejects_hypothesis_conditioned_reference(tmp_path):
    for seed in SEEDS:
        _receipt(tmp_path, seed, condition_on_hypothesis=seed == SEEDS[-1])
    with pytest.raises(ValueError, match="hypothesis-conditioned"):
        aggregate(tmp_path, repetitions=10, seed=1)
