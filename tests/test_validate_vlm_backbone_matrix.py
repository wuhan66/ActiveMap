from pathlib import Path

import yaml

from scripts.validate_vlm_backbone_matrix import validate


def _matrix():
    path = Path("configs/experiments/vlm_backbone_matrix.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_repository_vlm_backbone_matrix_preserves_fairness_contract():
    result = validate(_matrix())
    assert result["valid"] is True
    assert result["backbone_count"] == 4
    assert result["primary"] == "qwen3_vl_4b"
    assert result["same_scale_ablation"] == "gemma3_4b"


def test_vlm_backbone_matrix_rejects_test_reference():
    payload = _matrix()
    payload["data"]["test_jsonl"] = "/forbidden/test.jsonl"
    try:
        validate(payload)
    except ValueError as error:
        assert "test assets" in str(error)
    else:
        raise AssertionError("test references must be rejected")
