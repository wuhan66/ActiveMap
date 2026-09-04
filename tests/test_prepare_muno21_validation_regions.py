import json

import pytest

from scripts.prepare_muno21_validation_regions import prepare


def test_prepare_validation_regions_is_test_disjoint(tmp_path):
    summary = tmp_path / "summary.json"
    output = tmp_path / "regions.json"
    summary.write_text(
        json.dumps(
            {
                "derived_validation_regions": ["houston", "chicago"],
                "official_test_regions": ["boston"],
                "seed": 20260710,
            }
        ),
        encoding="utf-8",
    )

    result = prepare(summary, output)

    assert json.loads(output.read_text(encoding="utf-8")) == ["chicago", "houston"]
    assert result["official_test_overlap"] == []


def test_prepare_validation_regions_rejects_test_overlap(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "derived_validation_regions": ["boston"],
                "official_test_regions": ["boston"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overlap official test"):
        prepare(summary, tmp_path / "regions.json")
