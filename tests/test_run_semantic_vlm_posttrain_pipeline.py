import pytest

from scripts.run_semantic_vlm_posttrain_pipeline import validate_training_results


def _result(seed, returncode=0):
    return {"seed": seed, "returncode": returncode}


def test_validate_training_results_requires_final_adapters(tmp_path):
    for seed in (16, 19):
        final = tmp_path / f"seed{seed}" / "final"
        final.mkdir(parents=True)
        (final / "adapter_config.json").write_text("{}")
    validate_training_results([_result(16), _result(19)], [16, 19], tmp_path)


def test_validate_training_results_rejects_failed_seed(tmp_path):
    with pytest.raises(RuntimeError, match="19"):
        validate_training_results([_result(16), _result(19, 1)], [16, 19], tmp_path)
