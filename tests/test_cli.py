from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from activemap.cli import app

runner = CliRunner()


def _make_month(root: Path, aoi: str, month: str) -> None:
    base = root / "train" / aoi
    for folder in ("images_masked", "labels_match", "UDM_masks"):
        (base / folder).mkdir(parents=True, exist_ok=True)
    stem = f"global_monthly_2019_{month}_mosaic_{aoi}"
    (base / "images_masked" / f"{stem}.tif").touch()
    (base / "labels_match" / f"{stem}_Buildings.geojson").write_text("{}", encoding="utf-8")
    (base / "UDM_masks" / f"{stem}_UDM.tif").touch()


def test_gate_zero_cli_round_trip(tmp_path: Path) -> None:
    raw_root = tmp_path / "sn7"
    for index in range(4):
        aoi = f"L15-000{index}E-000{index}N_1_2_3"
        _make_month(raw_root, aoi, "01")
        _make_month(raw_root, aoi, "02")

    manifest = tmp_path / "sn7.parquet"
    result = runner.invoke(
        app,
        ["index-sn7", str(raw_root), str(manifest), "--no-metadata"],
    )
    assert result.exit_code == 0, result.output
    assert manifest.is_file()
    assert len(pd.read_parquet(manifest)) == 8

    split_dir = tmp_path / "splits"
    result = runner.invoke(app, ["make-splits", str(manifest), str(split_dir), "--seed", "7"])
    assert result.exit_code == 0, result.output
    split_manifest = tmp_path / "sn7_split.parquet"
    assert split_manifest.is_file()
    assert (split_dir / "sn7_v1_train.txt").is_file()

    result = runner.invoke(app, ["audit-manifest", str(split_manifest)])
    assert result.exit_code == 0, result.output


def test_validate_jsonl_cli() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = runner.invoke(
        app,
        [
            "validate-jsonl",
            str(project_root / "examples" / "episodes.jsonl"),
            str(project_root / "schemas" / "episode.schema.json"),
        ],
    )
    assert result.exit_code == 0, result.output
