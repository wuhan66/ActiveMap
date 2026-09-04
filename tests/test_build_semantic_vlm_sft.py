from types import SimpleNamespace

import numpy as np
from PIL import Image

from activemap.agent.records import AgentBelief
from activemap.geo_tools.records import GeoToolName, GeoToolResult
from activemap.models import EditOperation
from scripts.build_semantic_vlm_sft import consensus_tool_opportunity, render_registered_state


def _row(operation, *, cost=0.75):
    return SimpleNamespace(
        task_id="task",
        evidence_id="evidence",
        gt_edit=EditOperation.ADD,
        split="train",
        post_acquisition_belief=AgentBelief(
            edit_probabilities=[0.8, 0.1, 0.05, 0.05],
            confidence=0.8,
            uncertainty=0.2,
        ),
        semantic_result=GeoToolResult(
            call_id="semantic",
            tool=GeoToolName.RASTER_SEGMENT,
            success=True,
            outputs={"gated_edit": operation.value},
            cost=cost,
        ),
        example_id="example",
    )


def test_consensus_requires_majority_and_positive_mean_utility():
    decision = consensus_tool_opportunity(
        [_row(EditOperation.ADD), _row(EditOperation.ADD), _row(EditOperation.KEEP)],
        minimum_votes=2,
    )
    assert decision["beneficial_votes"] == 2
    assert decision["use_tool"] is True


def test_registered_state_renderer_writes_two_aligned_panels(tmp_path):
    image = np.zeros((3, 8, 8), dtype=np.float32)
    image[0, 2:6, 2:6] = 1.0
    prior = np.zeros((8, 8), dtype=np.uint8)
    prior[3:5, 3:5] = 1
    image_path, prior_path = tmp_path / "image.npy", tmp_path / "prior.npy"
    output_path = tmp_path / "state.jpg"
    np.save(image_path, image)
    np.save(prior_path, prior)
    render_registered_state(image_path, prior_path, output_path, panel_size=32)
    assert Image.open(output_path).size == (64, 32)
