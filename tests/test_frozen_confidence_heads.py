from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import torch


def _load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "train_frozen_confidence_heads.py"
    spec = importlib.util.spec_from_file_location("train_frozen_confidence_heads", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_head_training_is_reproducible() -> None:
    module = _load_script()
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(32, 4, generator=generator)
    target = torch.sigmoid(features[:, 0] - 0.5 * features[:, 1])
    kwargs = {
        "device": torch.device("cpu"),
        "seed": 17,
        "batch_size": 8,
        "max_epochs": 8,
        "min_epochs": 2,
        "patience": 3,
        "learning_rate": 0.05,
        "weight_decay": 0.0,
    }
    first_state, first_history = module._train_head(
        features[:24], target[:24], features[24:], target[24:], **kwargs
    )
    second_state, second_history = module._train_head(
        features[:24], target[:24], features[24:], target[24:], **kwargs
    )
    assert first_history == second_history
    assert set(first_state) == {"weight", "bias"}
    assert torch.equal(first_state["weight"], second_state["weight"])
    assert torch.equal(first_state["bias"], second_state["bias"])
