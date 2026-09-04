from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "selector"


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))


def test_generic_selector_is_capacity_matched_to_edit_conditioned_v5() -> None:
    pairs = [
        ("muno21_evidence_conservative_v5_server.yaml", "muno21_evidence_generic_v5_server.yaml"),
        (
            "muno21_evidence_conservative_v5_seed2_server.yaml",
            "muno21_evidence_generic_v5_seed2_server.yaml",
        ),
        (
            "muno21_evidence_conservative_v5_seed3_server.yaml",
            "muno21_evidence_generic_v5_seed3_server.yaml",
        ),
    ]
    for conditioned_name, generic_name in pairs:
        conditioned = _load(conditioned_name)
        generic = _load(generic_name)
        for section in ("seed", "data", "model", "training", "monitoring"):
            assert generic[section] == conditioned[section]

        conditioned_ablation = conditioned["ablation"]
        generic_ablation = generic["ablation"]
        assert conditioned_ablation["condition_on_hypothesis"] is True
        assert generic_ablation["condition_on_hypothesis"] is False
        assert generic_ablation == {
            **conditioned_ablation,
            "name": "generic_v5",
            "condition_on_hypothesis": False,
        }
        assert generic["output_dir"] != conditioned["output_dir"]
