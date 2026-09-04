from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

from activemap.features import ONLINE_OBSERVABLE_STATE_CONTRACT  # noqa: E402
from scripts.prepare_online_observable_selector_registry import build_registry  # noqa: E402


def write_base_registry(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "fixture-v1",
                "test_assets_read": False,
                "seed_artifacts": {
                    "1": {
                        "selector": {"path": "old.pt", "sha256": "old", "stop_margin": 0.0},
                        "tool_gate": {"path": "gate.joblib", "sha256": "gate"},
                        "tool_belief": {"path": "belief.pt", "sha256": "belief"},
                        "post_tool_adapter": {"path": "adapter.pt", "sha256": "adapter"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def write_selector(path: Path, contract=ONLINE_OBSERVABLE_STATE_CONTRACT) -> None:
    torch.save({"data_contract": contract, "stop_margin": 0.42}, path)


def test_registry_freezes_contract_hash_and_component_provenance(tmp_path: Path):
    base = tmp_path / "base.yaml"
    output = tmp_path / "online.yaml"
    selector = tmp_path / "selector.pt"
    write_base_registry(base)
    write_selector(selector)

    registry = build_registry(
        base,
        output,
        storage_root=tmp_path,
        selectors={7: selector},
        component_sources={7: 1},
    )

    entry = registry["seed_artifacts"]["7"]["selector"]
    assert registry["test_assets_read"] is False
    assert registry["test_policy"] == "not_authorized"
    assert entry["data_contract"] == ONLINE_OBSERVABLE_STATE_CONTRACT
    assert entry["stop_margin"] == 0.42
    assert entry["path"] == "${STORAGE_ROOT}/selector.pt"
    assert registry["provenance"]["component_reuse"]["7"]["component_source_seed"] == 1


def test_registry_rejects_legacy_selector_checkpoint(tmp_path: Path):
    base = tmp_path / "base.yaml"
    selector = tmp_path / "legacy.pt"
    write_base_registry(base)
    write_selector(selector, contract=None)

    with pytest.raises(ValueError, match="online-observable"):
        build_registry(
            base,
            tmp_path / "online.yaml",
            storage_root=tmp_path,
            selectors={7: selector},
            component_sources={7: 1},
        )
