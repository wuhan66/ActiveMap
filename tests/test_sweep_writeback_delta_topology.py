import json

import numpy as np

from scripts.sweep_writeback_delta_topology import sweep


def test_sweep_selects_filter_without_raster_regression(tmp_path):
    prior = np.zeros((8, 8), dtype=np.float32)
    target = prior.copy()
    target[2:6, 2:6] = 1.0
    committed = target.copy()
    committed[0, 7] = 1.0
    artifact = tmp_path / "mask.npz"
    np.savez_compressed(
        artifact,
        committed_mask=committed,
        prior_mask=prior,
        target_mask=target,
        valid_mask=np.ones_like(prior),
        transform=np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    )
    source = tmp_path / "writeback.jsonl"
    source.write_text(
        json.dumps(
            {
                "task_id": "task-1",
                "budget": 1.0,
                "operation": "ADD",
                "mask_artifact": str(artifact),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = sweep(
        [source],
        [0, 2],
        raster_iou_tolerance=0.0,
        topology_floor=0.99,
        replay_floor=0.98,
    )

    assert result["selected_min_delta_component_pixels"] == 2
    assert result["candidates"][1]["mean_raster_iou"] == 1.0
    assert result["test_assets_read"] is False
