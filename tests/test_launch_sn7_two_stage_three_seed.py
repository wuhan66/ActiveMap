import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.launch_sn7_two_stage_three_seed import jobs, prepare_job, validate_samples


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_jobs_pairs_three_distinct_seeds_and_gpus() -> None:
    assert jobs([21, 22, 23], [0, 2, 4]) == [
        {"seed": 21, "gpu": 0},
        {"seed": 22, "gpu": 2},
        {"seed": 23, "gpu": 4},
    ]


@pytest.mark.parametrize(
    ("seeds", "gpus"),
    [([21, 22], [0, 2]), ([21, 21, 23], [0, 2, 4]), ([21, 22, 23], [0, 0, 4])],
)
def test_jobs_rejects_incomplete_or_duplicate_matrix(seeds, gpus) -> None:
    with pytest.raises(ValueError):
        jobs(seeds, gpus)


def test_prepare_job_supports_promoted_variant_and_samples(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    samples = tmp_path / "states.jsonl"
    prepared = prepare_job(
        repo,
        tmp_path / "runs",
        {"seed": 21, "gpu": 0},
        base_config=repo / "configs/selector/sn7_evidence_two_stage_v3_server.yaml",
        variant_name="generic_utility",
        samples=samples,
    )
    payload = yaml.safe_load(Path(prepared["config"]).read_text(encoding="utf-8"))
    assert payload["data"]["samples"] == str(samples.resolve())
    assert payload["ablation"]["condition_on_hypothesis"] is False


def test_prepare_job_requires_online_observable_contract(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    prepared = prepare_job(
        repo,
        tmp_path / "runs",
        {"seed": 21, "gpu": 0},
        base_config=repo / "configs/selector/sn7_online_observable_two_stage_v1.yaml",
        require_online_observable=True,
    )
    payload = yaml.safe_load(Path(prepared["config"]).read_text(encoding="utf-8"))
    assert payload["data_contract"]["version"] == "online-observable-state-v1"

    with pytest.raises(ValueError, match="online-observable-state-v1"):
        prepare_job(
            repo,
            tmp_path / "legacy-runs",
            {"seed": 22, "gpu": 1},
            base_config=repo / "configs/selector/sn7_evidence_two_stage_v3_server.yaml",
            require_online_observable=True,
        )


def test_validate_samples_follows_online_sanitizer_provenance(tmp_path: Path) -> None:
    source = tmp_path / "states.jsonl"
    source.write_text('{"source": true}\n', encoding="utf-8")
    source.with_suffix(".audit.json").write_text(
        json.dumps(
            {
                "test_assets_read": False,
                "duplicate_source_budget_steps": 0,
                "states": 1,
                "targets": {"STOP": 1},
            }
        ),
        encoding="utf-8",
    )
    source.with_suffix(".utility_audit.json").write_text("{}\n", encoding="utf-8")
    samples = tmp_path / "online.jsonl"
    samples.write_text('{"online": true}\n', encoding="utf-8")
    summary = samples.with_suffix(samples.suffix + ".summary.json")
    summary.write_text(
        json.dumps(
            {
                "schema_version": "online-observable-selector-features-v1",
                "source": {"path": str(source.resolve()), "sha256": _sha256(source)},
                "output": {"path": str(samples.resolve()), "sha256": _sha256(samples)},
                "online_state_contract": {"version": "online-observable-state-v1"},
                "test_assets_read": False,
            }
        ),
        encoding="utf-8",
    )

    result = validate_samples(samples)

    assert result["states"] == 1
    assert result["sanitizer_summary"]["sha256"] == _sha256(summary)
    assert result["audits"]["audit"] == str(source.with_suffix(".audit.json").resolve())
