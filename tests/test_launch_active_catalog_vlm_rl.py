from types import SimpleNamespace

from scripts.launch_active_catalog_vlm_rl import command


def test_rl_launcher_preserves_policy_optimization_contract(tmp_path):
    args = SimpleNamespace(
        python="python", sft_adapter=tmp_path / "adapter",
        train_jsonl=tmp_path / "train.jsonl", val_jsonl=tmp_path / "val.jsonl",
        epochs=1.0, learning_rate=1e-6, gradient_accumulation=16,
        max_length=2048, action_limit=4, kl_beta=0.05,
        entropy_weight=0.01, temperature=1.0, seed=7,
        logging_steps=5, eval_steps=100, save_steps=100,
        max_train_samples=None, max_eval_samples=None,
    )
    result = command(args, tmp_path / "output")
    assert result[1] == "scripts/train_active_catalog_vlm_rl.py"
    assert result[result.index("--action-limit") + 1] == "4"
    assert result[result.index("--kl-beta") + 1] == "0.05"
