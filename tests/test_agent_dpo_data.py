import pytest

from scripts.train_agent_dpo import (
    ControlledStop,
    _append_metric_event,
    _audit_reference_adapter,
    _audit_rows,
    _rows_digest,
    _tokenizer_source,
    _wait_for_control,
)


class _Tokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text.split())))}


def _row(chosen_utility=1.0, rejected_utility=0.0):
    return {
        "prompt": "one two three",
        "chosen": " four",
        "rejected": " five six",
        "chosen_utility": chosen_utility,
        "rejected_utility": rejected_utility,
        "preference_family": "safety",
    }


def test_dpo_audit_reports_families_lengths_and_truncation() -> None:
    summary = _audit_rows([_row()], _Tokenizer(), max_length=4)

    assert summary == {
        "samples": 1,
        "families": {"safety": 1},
        "nonpositive_margins": 0,
        "max_observed_length": 5,
        "truncated": 1,
    }


def test_dpo_audit_rejects_nonpositive_margin() -> None:
    with pytest.raises(ValueError, match="non-positive preference margins"):
        _audit_rows([_row(chosen_utility=0.0, rejected_utility=0.0)], _Tokenizer(), 10)


def test_tokenizer_source_uses_adapter_base(tmp_path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    class _Config:
        base_model_name_or_path = "/models/base"

    class _PeftConfig:
        @staticmethod
        def from_pretrained(path):
            assert path == str(adapter)
            return _Config()

    assert _tokenizer_source(str(adapter), _PeftConfig) == "/models/base"


def test_tokenizer_source_keeps_base_model_path(tmp_path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    assert _tokenizer_source(str(base), object) == str(base)


def test_rows_digest_is_stable_and_content_sensitive() -> None:
    first = [_row(), {**_row(), "preference_family": "update"}]
    reordered_keys = [{key: row[key] for key in reversed(row)} for row in first]
    assert _rows_digest(first) == _rows_digest(reordered_keys)
    changed = [*first[:-1], {**first[-1], "chosen_utility": 2.0}]
    assert _rows_digest(first) != _rows_digest(changed)


def test_reference_adapter_audit_requires_exact_frozen_sft_copy() -> None:
    torch = pytest.importorskip("torch")

    class AdapterPair(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.peft_config = {"default": object(), "ref": object()}
            self.default = torch.nn.Module()
            self.default.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
            self.ref = torch.nn.Module()
            self.ref.weight = torch.nn.Parameter(
                self.default.weight.detach().clone(), requires_grad=False
            )

        def named_parameters(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            yield "adapter.default.weight", self.default.weight
            yield "adapter.ref.weight", self.ref.weight

    model = AdapterPair()
    summary = _audit_reference_adapter(model)
    assert summary["source"] == "copied_sft_default_adapter"
    assert summary["parameter_tensors_compared"] == 1

    model.ref.weight.data[0] = 9.0
    with pytest.raises(RuntimeError, match="differs from SFT initialization"):
        _audit_reference_adapter(model)


def test_stop_control_prevents_training_from_being_published(tmp_path) -> None:
    control = tmp_path / "control"
    control.mkdir()
    (control / "STOP").touch()

    with pytest.raises(ControlledStop, match="control/STOP requested"):
        _wait_for_control(control, poll_seconds=0.0)


def test_metric_events_are_immediately_appended_as_jsonl(tmp_path) -> None:
    path = tmp_path / "history.jsonl"
    _append_metric_event(path, {"step": 5, "loss": 0.6})
    _append_metric_event(path, {"step": 10, "loss": 0.5})

    assert path.read_text(encoding="utf-8").splitlines() == [
        '{"step":5,"loss":0.6}',
        '{"step":10,"loss":0.5}',
    ]
