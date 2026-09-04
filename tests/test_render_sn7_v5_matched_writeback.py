from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.figures.render_sn7_v5_matched_writeback import render

POLICIES = (
    "direct_commit",
    "direct_safe_commit",
    "selected_commit",
    "selected_safe_commit",
)
FORCED_ACQUISITION_POLICY = "forced_safe_commit"
OPERATIONS = ("ADD", "DELETE", "RESHAPE")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _aggregate_summary(run_root: Path, *, eligible: bool) -> dict:
    policies = [*POLICIES, FORCED_ACQUISITION_POLICY]
    return {
        "schema_version": "sn7-v5-matched-nonkeep-factorial-v2",
        "split": "val",
        "test_assets_read": False,
        "model_seeds": [20260817, 20260818, 20260819],
        "policies": policies,
        "promotion": {"eligible_for_extension_claim": eligible},
        "inputs": {
            str(seed): {
                policy: str(
                    (
                        run_root
                        / "writebacks"
                        / f"seed{seed}"
                        / "val"
                        / policy
                        / "writeback.jsonl"
                    ).resolve()
                )
                for policy in policies
            }
            for seed in (20260817, 20260818, 20260819)
        },
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "source.png"
    Image.fromarray(
        np.dstack(np.meshgrid(np.arange(32), np.arange(32)))
        .astype(np.uint8)
        .repeat(2, axis=2)[..., :3]
    ).save(source)
    cases = []
    episodes = []
    policies: dict[str, list[dict]] = {policy: [] for policy in POLICIES}
    rollouts = {"direct": [], "selected": []}
    evidence_root = tmp_path / "evidence"
    for operation_index, operation in enumerate(OPERATIONS):
        for sample_index in range(3):
            number = operation_index * 3 + sample_index
            task_id = f"task-{number}"
            source_episode = f"episode-{number}"
            initial, extra = f"initial-{number}", f"extra-{number}"
            case = {
                "operation": operation,
                "source_episode": source_episode,
                "task_id": task_id,
                "aoi_id": f"aoi-{operation_index}",
                "budget": 1.5,
                "state_id": f"{source_episode}__b1p5__s0",
                "initial_evidence_ids": [initial],
            }
            cases.append(case)
            episodes.append(
                {
                    "episode_id": source_episode,
                    "aoi_id": case["aoi_id"],
                    "split": "val",
                    "test_assets_read": False,
                    "evidence_catalog": [
                        {
                            "evidence_id": initial,
                            "image_path": str(source),
                            "region": [0, 0, 32, 32],
                        },
                        {"evidence_id": extra, "image_path": str(source), "region": [0, 0, 32, 32]},
                    ],
                }
            )
            base = np.zeros((16, 16), dtype=bool)
            base[4:10, 4:10] = True
            target = base.copy()
            target[7:13, 7:13] = True
            direct = base.copy()
            selected = target.copy()
            for policy in POLICIES:
                rejected_by_safe_commit = policy == "selected_safe_commit" and sample_index == 0
                final = (
                    base
                    if rejected_by_safe_commit
                    else (selected if policy.startswith("selected") else direct)
                )
                artifact = tmp_path / "artifacts" / policy / f"{task_id}.npz"
                artifact.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    artifact,
                    committed_mask=final,
                    prior_mask=base,
                    target_mask=target,
                    valid_mask=np.ones_like(base),
                )
                policies[policy].append(
                    {
                        "task_id": task_id,
                        "aoi_id": case["aoi_id"],
                        "budget": 1.5,
                        "target": f"COMMIT:{operation}",
                        "split": "val",
                        "test_assets_read": False,
                        "mask_artifact": str(artifact),
                        "writeback_changed": not rejected_by_safe_commit,
                        "effective_operation": operation,
                    }
                )
            rollout = {
                "task_id": task_id,
                "source_episode": source_episode,
                "aoi_id": case["aoi_id"],
                "budget": 1.5,
                "split": "val",
                "test_assets_read": False,
                "initial_evidence_id": initial,
                "selected_extra_evidence_id": extra,
            }
            rollouts["selected"].append(rollout)
            rollouts["direct"].append({**rollout, "selected_extra_evidence_id": None})
            crop_dir = evidence_root / source_episode
            crop_dir.mkdir(parents=True)
            crop = crop_dir / "selected_extra.png"
            Image.open(source).convert("RGB").save(crop)
            _write_json(
                crop_dir / "manifest.json",
                {
                    "source_episode": source_episode,
                    "task_id": task_id,
                    "budget": case["budget"],
                    "split": "val",
                    "test_assets_read": False,
                    "evidence": [
                        {
                            "evidence_id": extra,
                            "output": crop.name,
                            "output_sha256": _sha256(crop),
                            "timestamp": "2019_01",
                            "scale": 2,
                        }
                    ],
                },
            )
    qualitative = tmp_path / "qualitative_manifest.json"
    _write_json(
        qualitative,
        {
            "schema_version": "sn7-v5-predeclared-qualitative-manifest-v1",
            "seed": 20260817,
            "split": "val",
            "test_assets_read": False,
            "controller_or_writeback_outputs_read": False,
            "cases": cases,
        },
    )
    episodes_path = tmp_path / "episodes.jsonl"
    _write_jsonl(episodes_path, episodes)
    run_root = tmp_path / "run"
    for policy, rows in policies.items():
        _write_jsonl(
            run_root / "writebacks" / "seed20260817" / "val" / policy / "writeback.jsonl", rows
        )
    _write_jsonl(
        run_root / "rollouts" / "seed20260817_val" / "direct_rollouts.jsonl", rollouts["direct"]
    )
    _write_jsonl(
        run_root / "rollouts" / "seed20260817_val" / "selected_rollouts.jsonl", rollouts["selected"]
    )
    _write_json(
        run_root / "authorization" / "checkpoint_receipts" / "seed20260817.json",
        {"test_assets_read": False, "checkpoint_sha256": "checkpoint-hash"},
    )
    _write_json(
        run_root / "train_calibration" / "seed20260817.json",
        {"split": "train", "test_assets_read": False},
    )
    return qualitative, episodes_path, run_root, evidence_root


def test_renders_predeclared_validation_cases_with_provenance(tmp_path: Path) -> None:
    qualitative, episodes, run_root, evidence_root = _fixture(tmp_path)
    output = tmp_path / "output"

    result = render(
        qualitative_manifest=qualitative,
        episodes_path=episodes,
        run_root=run_root,
        output_dir=output,
        scope="supplement",
        render_seed=None,
        evidence_root=evidence_root,
        aggregate_summary=None,
        asset_root_map=None,
        panel_size=64,
    )

    assert result["split"] == "val"
    assert result["test_assets_read"] is False
    assert len(result["cases"]) == 9
    first = result["cases"][0]
    assert first["selected_extra_evidence"] is True
    assert (output / first["directory"] / "plate.png").is_file()
    assert (output / first["directory"] / "selected_evidence.png").is_file()
    assert (output / first["directory"] / "prior_change_zoom.png").is_file()
    assert (output / first["directory"] / "prior_residual_zoom.png").is_file()
    assert (output / first["directory"] / "safe_commit_delta.png").is_file()
    assert set(first["zoom_bounds"]) == {"change", "residual"}
    assert set(first["panels"]["prior"]["zooms"]) == {"change", "residual"}
    safe_delta = np.asarray(Image.open(output / first["directory"] / "safe_commit_delta.png"))
    assert np.any(np.all(safe_delta == np.asarray((213, 94, 0), dtype=np.uint8), axis=-1))
    assert Image.open(output / first["directory"] / "plate.png").size == (64 * 6, 64 * 3)
    assert (output / "manifest.json").is_file()


def test_rejects_non_validation_episode(tmp_path: Path) -> None:
    qualitative, episodes, run_root, evidence_root = _fixture(tmp_path)
    rows = [json.loads(line) for line in episodes.read_text(encoding="utf-8").splitlines()]
    rows[0]["split"] = "test"
    _write_jsonl(episodes, rows)

    with pytest.raises(ValueError, match="validation-only"):
        render(
            qualitative_manifest=qualitative,
            episodes_path=episodes,
            run_root=run_root,
            output_dir=tmp_path / "invalid",
            scope="supplement",
            render_seed=None,
            evidence_root=evidence_root,
            aggregate_summary=None,
            asset_root_map=None,
            panel_size=64,
        )


def test_main_scope_requires_completed_factorial_aggregate(tmp_path: Path) -> None:
    qualitative, episodes, run_root, evidence_root = _fixture(tmp_path)

    with pytest.raises(ValueError, match="requires the completed factorial aggregate"):
        render(
            qualitative_manifest=qualitative,
            episodes_path=episodes,
            run_root=run_root,
            output_dir=tmp_path / "main-without-aggregate",
            scope="main",
            render_seed=None,
            evidence_root=evidence_root,
            aggregate_summary=None,
            asset_root_map=None,
            panel_size=64,
        )


def test_main_scope_exports_legible_three_row_assets(tmp_path: Path) -> None:
    qualitative, episodes, run_root, evidence_root = _fixture(tmp_path)
    aggregate = tmp_path / "three_seed_nonkeep_factorial_with_forced_summary.json"
    _write_json(
        aggregate,
        _aggregate_summary(run_root, eligible=True),
    )
    output = tmp_path / "main-output"

    result = render(
        qualitative_manifest=qualitative,
        episodes_path=episodes,
        run_root=run_root,
        output_dir=output,
        scope="main",
        render_seed=None,
        evidence_root=evidence_root,
        aggregate_summary=aggregate,
        asset_root_map=None,
        panel_size=64,
    )

    assert set(result["main_assets"]) == {
        "full_context",
        "change_zooms",
        "residual_zooms",
        "contact_sheet",
    }
    for filename in (
        "v5_main_context.png",
        "v5_main_change_zooms.png",
        "v5_main_residual_zooms.png",
    ):
        assert Image.open(output / filename).size == (64 * 6, 64 * 3)
    assert Image.open(output / "v5_main_contact_sheet.png").size == (64 * 6, 64 * 9)


def test_main_scope_rejects_a_completed_but_unpromoted_aggregate(tmp_path: Path) -> None:
    qualitative, episodes, run_root, evidence_root = _fixture(tmp_path)
    aggregate = tmp_path / "three_seed_nonkeep_factorial_with_forced_summary.json"
    _write_json(
        aggregate,
        _aggregate_summary(run_root, eligible=False),
    )

    with pytest.raises(ValueError, match="passed registered promotion gate"):
        render(
            qualitative_manifest=qualitative,
            episodes_path=episodes,
            run_root=run_root,
            output_dir=tmp_path / "unpromoted-main-output",
            scope="main",
            render_seed=None,
            evidence_root=evidence_root,
            aggregate_summary=aggregate,
            asset_root_map=None,
            panel_size=64,
        )


def test_main_scope_rejects_aggregate_from_different_writebacks(tmp_path: Path) -> None:
    qualitative, episodes, run_root, evidence_root = _fixture(tmp_path)
    aggregate = tmp_path / "three_seed_nonkeep_factorial_with_forced_summary.json"
    summary = _aggregate_summary(run_root, eligible=True)
    summary["inputs"]["20260817"]["selected_safe_commit"] = str(tmp_path / "wrong.jsonl")
    _write_json(aggregate, summary)

    with pytest.raises(ValueError, match="does not match renderer writeback"):
        render(
            qualitative_manifest=qualitative,
            episodes_path=episodes,
            run_root=run_root,
            output_dir=tmp_path / "mixed-input-main-output",
            scope="main",
            render_seed=None,
            evidence_root=evidence_root,
            aggregate_summary=aggregate,
            asset_root_map=None,
            panel_size=64,
        )
