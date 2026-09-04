import pytest

from scripts.preflight_sn7_step0_frozen_test_v2 import _parse_gpu_rows


def test_parse_gpu_rows() -> None:
    rows = _parse_gpu_rows("0, 2, 24564, 0\n2, 410, 24564, 11\n")
    assert rows[0]["memory_used_mib"] == 2
    assert rows[2]["utilization_percent"] == 11


def test_parse_gpu_rows_rejects_malformed_output() -> None:
    with pytest.raises(ValueError, match="unexpected"):
        _parse_gpu_rows("0, 2, 24564\n")
