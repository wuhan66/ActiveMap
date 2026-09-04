import numpy as np

from activemap.agent.identifiers import public_evidence_id, public_task_id
from activemap.agent.records import AgentBelief
from activemap.agent.tool_belief_data import PostAcquisitionToolPairExample
from activemap.agent.tool_features import (
    SEMANTIC_TOOL_FEATURE_DIM,
    SEMANTIC_TOOL_FEATURE_NAMES,
    encode_semantic_tool_result,
)
from activemap.geo_tools.records import GeoToolName, GeoToolResult
from activemap.models import (
    CandidateHypothesis,
    EditOperation,
    EditRecord,
    EpisodeRecord,
    EvidenceItem,
)
from activemap.updater_records import UpdaterSample
from scripts.build_post_acquisition_semantic_tool_data import (
    augment_examples,
    default_threshold_grid,
    parse_asset_root_maps,
    remap_asset_path,
)


def _belief():
    return AgentBelief(
        edit_probabilities=[0.8, 0.1, 0.05, 0.05],
        confidence=0.8,
        uncertainty=0.4,
    )


def _result(tool, call_id, outputs=None):
    return GeoToolResult(
        call_id=call_id,
        tool=tool,
        success=True,
        outputs=outputs or {},
        cost=0.1,
    )


class _FakeSemanticTool:
    cost = 0.25

    def __init__(self):
        self.last_call = None

    def run(self, call):
        self.last_call = call
        return _result(
            GeoToolName.RASTER_SEGMENT,
            call.call_id,
            {
                "foreground_fraction": 0.4,
                "changed_fraction": 0.2,
                "add_fraction": 0.15,
                "remove_fraction": 0.05,
                "learned_add_fraction": 0.12,
                "learned_remove_fraction": 0.03,
                "mean_add_probability": 0.24,
                "mean_remove_probability": 0.08,
                "edit_probabilities": [0.1, 0.6, 0.2, 0.1],
                "update_probability": 0.6,
                "confidence": 0.6,
                "uncertainty": 0.4,
                "prior_iou": 0.7,
                "mean_probability": 0.6,
                "component_count": 3,
            },
        ).model_copy(update={"cost": self.cost})


def test_semantic_tool_encoding_preserves_directional_map_differences():
    result = _FakeSemanticTool().run(type("Call", (), {"call_id": "semantic"})())
    encoded = encode_semantic_tool_result(result)
    by_name = dict(zip(SEMANTIC_TOOL_FEATURE_NAMES, encoded, strict=True))
    assert len(encoded) == SEMANTIC_TOOL_FEATURE_DIM
    assert by_name["add_fraction"] == 0.15
    assert by_name["remove_fraction"] == 0.05
    assert by_name["learned_add_fraction"] == 0.12
    assert by_name["learned_remove_fraction"] == 0.03
    assert by_name["mean_add_probability"] == 0.24
    assert by_name["mean_remove_probability"] == 0.08
    assert by_name["edit_probability_keep"] == 0.1
    assert by_name["edit_probability_add"] == 0.6
    assert by_name["edit_probability_delete"] == 0.2
    assert by_name["edit_probability_reshape"] == 0.1
    assert by_name["update_probability"] == 0.6
    assert by_name["confidence"] == 0.6
    assert by_name["uncertainty"] == 0.4
    assert by_name["prior_iou"] == 0.7


def test_prior_backend_sweeps_only_the_frozen_current_threshold():
    assert default_threshold_grid("prior-sam-road", 0.8) == (0.8,)
    assert 0.341 in default_threshold_grid("sam-road", 0.6)
    assert default_threshold_grid("updater", 0.5) is None


def test_asset_root_remap_is_explicit_and_prefix_safe(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    mappings = parse_asset_root_maps([f"{source}={target}"])

    assert remap_asset_path(source / "images" / "tile.jpg", mappings) == (
        target / "images" / "tile.jpg"
    )
    assert remap_asset_path(tmp_path / "source-other" / "tile.jpg", mappings) == (
        tmp_path / "source-other" / "tile.jpg"
    )


def test_semantic_augmentation_matches_public_task_and_evidence(tmp_path):
    raw_task = "task-raw"
    raw_evidence = "candidate"
    task_id = public_task_id(raw_task)
    evidence_id = public_evidence_id(raw_evidence)
    prior_path = tmp_path / "prior.npy"
    target_path = tmp_path / "target.npy"
    np.save(prior_path, np.zeros((6, 8), dtype=np.uint8))
    np.save(target_path, np.zeros((6, 8), dtype=np.uint8))
    episode = EpisodeRecord(
        episode_id=raw_task,
        split="train",
        source_dataset="MUNO21",
        map_before="before",
        target_map="target",
        hypothesis=CandidateHypothesis(
            op=EditOperation.KEEP,
            object_id="road",
            source="test",
        ),
        evidence_catalog=[
            EvidenceItem(
                evidence_id=raw_evidence,
                timestamp="2020",
                    region=(0, 0, 8, 6),
                scale=1,
                image_path="image.tif",
                clear_fraction=1.0,
                cost=1.0,
            )
        ],
        gt_edit=EditRecord(op=EditOperation.KEEP, object_id="road"),
        is_synthetic=False,
        derivation_version="test",
    )
    sample = UpdaterSample(
        sample_id="task-raw".removesuffix("__temporal"),
        split="train",
        image_path="image.tif",
        prior_mask_path=str(prior_path),
        target_mask_path=str(target_path),
        edit_type=EditOperation.KEEP,
        geometry_delta=[0.0] * 8,
    )
    row = PostAcquisitionToolPairExample(
        example_id="example",
        task_id=task_id,
        split="train",
        evidence_id=evidence_id,
        post_acquisition_belief=_belief(),
        quality_result=_result(GeoToolName.IMAGE_QUALITY, "quality"),
        temporal_result=_result(GeoToolName.TEMPORAL_CHANGE, "temporal"),
        target_belief=_belief(),
        gt_edit=EditOperation.KEEP,
        evidence_cost=1.0,
        tool_cost=0.2,
        metadata={"test_assets_read": False},
    )
    tool = _FakeSemanticTool()
    augmented, summary = augment_examples(
        [row],
        {task_id: episode},
        {task_id: sample},
        tool,
        threshold=0.5,
        threshold_grid=(0.2, 0.5),
    )
    assert augmented[0].semantic_result is not None
    assert augmented[0].tool_cost == 0.45
    assert "out_size" not in tool.last_call.parameters
    assert "pixel_window" not in tool.last_call.parameters
    assert tool.last_call.inputs["image_path"] == sample.image_path
    assert tool.last_call.parameters["threshold_grid"] == [0.2, 0.5]
    assert summary["example_count"] == 1
    assert summary["test_assets_read"] is False
