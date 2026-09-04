from argparse import Namespace
from pathlib import Path

from scripts.launch_active_catalog_vlm_dpo import trainer_command


def test_trainer_command_preserves_smoke_limits():
    args = Namespace(
        python="python",
        sft_adapter=Path("adapter"),
        train_jsonl=Path("train.jsonl"),
        val_jsonl=Path("val.jsonl"),
        epochs=1.0,
        learning_rate=5e-6,
        batch_size=1,
        gradient_accumulation=1,
        max_length=2048,
        beta=0.1,
        family_balance="none",
        family_balance_power=1.0,
        safe_acquire_boost=2.0,
        unsafe_stop_scale=0.25,
        seed=7,
        logging_steps=1,
        eval_steps=1,
        save_steps=1,
        max_train_samples=2,
        max_eval_samples=2,
    )
    command = trainer_command(args, Path("out"))
    assert command[-4:] == [
        "--max-train-samples",
        "2",
        "--max-eval-samples",
        "2",
    ]
    assert command[1] == "scripts/train_active_catalog_vlm_dpo.py"
    assert command[command.index("--safe-acquire-boost") + 1] == "2.0"
    assert command[command.index("--unsafe-stop-scale") + 1] == "0.25"
