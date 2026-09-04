import pytest

from scripts.render_active_catalog_paired_examples import select_paired_examples


def _trace(target, prediction):
    return {"target_edit": target, "predicted_edit": prediction}


def _writeback(gain):
    return {"raster_iou_gain": gain}


def test_paired_selection_uses_identical_keys_and_finds_disagreement():
    keys = [("a", 3.0), ("b", 3.0), ("c", 3.0)]
    traces = {
        "sft": {
            keys[0]: _trace("ADD", "KEEP"),
            keys[1]: _trace("ADD", "ADD"),
            keys[2]: _trace("ADD", "ADD"),
        },
        "react": {
            keys[0]: _trace("ADD", "ADD"),
            keys[1]: _trace("ADD", "KEEP"),
            keys[2]: _trace("ADD", "ADD"),
        },
    }
    writebacks = {
        "sft": {
            keys[0]: _writeback(-0.2),
            keys[1]: _writeback(0.1),
            keys[2]: _writeback(0.0),
        },
        "react": {
            keys[0]: _writeback(0.3),
            keys[1]: _writeback(-0.1),
            keys[2]: _writeback(0.2),
        },
    }
    selected = select_paired_examples(
        traces,
        writebacks,
        candidate="react",
        per_edit_success=1,
        per_edit_failure=1,
        per_edit_disagreement=1,
        min_visual_fraction=0.0,
    )
    assert len({key for key, _ in selected}) == len(selected)
    assert (keys[0], "success") in selected
    assert (keys[1], "failure") in selected


def test_paired_selection_rejects_mismatched_support():
    traces = {
        "sft": {("a", 3.0): _trace("ADD", "ADD")},
        "react": {("b", 3.0): _trace("ADD", "ADD")},
    }
    writebacks = {
        "sft": {("a", 3.0): _writeback(0.0)},
        "react": {("b", 3.0): _writeback(0.1)},
    }
    with pytest.raises(ValueError, match="identical"):
        select_paired_examples(
            traces,
            writebacks,
            candidate="react",
            per_edit_success=1,
            per_edit_failure=1,
            per_edit_disagreement=1,
        )
