import numpy as np
from PIL import Image

from activemap.agent.identifiers import public_task_id
from scripts.render_active_catalog_closed_loop_examples import (
    belief_history,
    boundary,
    error_mask,
    error_overlay,
    event_action,
    render_panel_bundle,
    select_examples,
)


def test_boundary_and_error_overlay_have_stable_shape():
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    assert boundary(mask).sum() == 12
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    rendered = error_overlay(image, mask, np.roll(mask, 1, axis=0))
    assert rendered.shape == image.shape
    assert rendered.max() > 0
    mask_only = error_mask(mask, np.roll(mask, 1, axis=0))
    assert mask_only.shape == image.shape
    assert np.all(mask_only[~(mask | np.roll(mask, 1, axis=0))] == 0)


def test_belief_history_reads_only_observable_state():
    trace = {
        "events": [
            {
                "observable_state": {
                    "belief": {"edit_probabilities": [0.1, 0.2, 0.3, 0.4]}
                }
            }
        ]
    }
    assert np.allclose(belief_history(trace), [[0.1, 0.2, 0.3, 0.4]])


def test_missing_belief_history_is_explicitly_empty():
    values = belief_history({"events": []})
    assert values.shape == (0, 4)


def test_event_action_supports_step0_and_legacy_schemas():
    assert event_action({"executed_action": {"action": "ACQUIRE"}}) == "ACQUIRE"
    assert event_action({"action": "TOOL"}) == "TOOL"
    assert event_action({"event_type": "belief_update"}) == "belief_update"


def test_example_selection_joins_public_task_and_budget():
    trace = {
        "source_episode": "episode",
        "budget": 3.0,
        "target_edit": "RESHAPE",
        "predicted_edit": "RESHAPE",
        "spent_cost": 1.0,
    }
    writeback = {
        "task_id": public_task_id("episode"),
        "budget": 3.0,
        "raster_iou_gain": 0.2,
    }
    selected = select_examples([trace], [writeback], 1)
    assert selected[0][2] == "largest map-quality gain"


def test_panel_bundle_writes_composite_and_individual_files(tmp_path):
    y, x = np.mgrid[:32, :32]
    initial = np.stack((x * 8, y * 8, np.full_like(x, 90)), axis=-1).astype(np.uint8)
    acquired = np.roll(initial, 3, axis=1)
    files = render_panel_bundle(
        initial_rgb=initial,
        acquired_rgb=acquired,
        prior_panel=initial,
        committed_panel=acquired,
        target_panel=initial,
        errors=np.maximum(initial, acquired),
        beliefs=np.asarray(
            [[0.6, 0.1, 0.1, 0.2], [0.2, 0.1, 0.1, 0.6]], dtype=np.float32
        ),
        output_dir=tmp_path,
        header="LAYOUT TEST",
        acquired=True,
    )
    assert len(files) == 7
    assert "composite" not in files
    for key in (
        "initial_evidence",
        "acquired_evidence",
        "prior_mask",
        "committed_mask",
        "target_mask",
        "error_mask",
    ):
        loaded = np.asarray(Image.open(tmp_path / files[key]))
        assert loaded.shape[:2] == (32, 32)
        assert loaded.max() > loaded.min()
    belief = np.asarray(Image.open(tmp_path / files["belief_revision"]))
    assert belief.shape[0] > 100 and belief.shape[1] > 100
    assert belief[..., 3].max() > 0
