from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.render_sn7_v5_selector_config import render_config


def test_rendered_v5_selector_config_binds_all_five_registered_arguments(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        "seed: 1\ndata:\n  samples: old\ntraining:\n  device: cpu\noutput_dir: old\n",
        encoding="utf-8",
    )
    samples = tmp_path / "fit_tune.jsonl"
    samples.write_text('{"split":"train"}\n', encoding="utf-8")
    output = tmp_path / "configs" / "seed.yaml"
    run_dir = tmp_path / "selectors" / "seed"

    render_config(base, samples, output, run_dir, seed=20260817)

    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert payload["seed"] == 20260817
    assert payload["data"]["samples"] == str(samples.resolve())
    assert payload["training"]["device"] == "cuda:0"
    assert payload["output_dir"] == str(run_dir.resolve())
    with pytest.raises(FileExistsError):
        render_config(base, samples, output, run_dir, seed=20260818)
