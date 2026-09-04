import torch

from activemap.agent.post_tool_action_adapter import (
    PostToolActionAdapter,
    PostToolActionAdapterConfig,
    SEMANTIC_CONDITIONED_FEATURE_DIM,
    SEMANTIC_ONLY_FEATURE_DIM,
    TOOL_CONDITIONED_FEATURE_DIM,
)
from activemap.agent.tool_belief_model import BELIEF_FEATURE_DIM


def test_post_tool_adapter_forward_and_backward():
    model = PostToolActionAdapter(
        PostToolActionAdapterConfig(hidden_dim=16)
    )
    update, operation = model(torch.zeros(4, BELIEF_FEATURE_DIM))
    assert update.shape == (4,)
    assert operation.shape == (4, 3)
    loss = update.square().mean() + operation.square().mean()
    loss.backward()
    assert torch.isfinite(loss)


def test_tool_conditioned_adapter_forward_and_backward():
    model = PostToolActionAdapter(
        PostToolActionAdapterConfig(
            input_dim=TOOL_CONDITIONED_FEATURE_DIM,
            hidden_dim=16,
            include_tool_results=True,
        )
    )
    update, operation = model(torch.zeros(3, TOOL_CONDITIONED_FEATURE_DIM))
    assert update.shape == (3,)
    assert operation.shape == (3, 3)
    (update.sum() + operation.sum()).backward()


def test_semantic_conditioned_adapter_forward_and_backward():
    model = PostToolActionAdapter(
        PostToolActionAdapterConfig(
            input_dim=SEMANTIC_CONDITIONED_FEATURE_DIM,
            hidden_dim=16,
            include_semantic_result=True,
        )
    )
    update, operation = model(torch.zeros(3, SEMANTIC_CONDITIONED_FEATURE_DIM))
    assert update.shape == (3,)
    assert operation.shape == (3, 3)
    (update.sum() + operation.sum()).backward()


def test_semantic_only_adapter_forward_and_backward():
    model = PostToolActionAdapter(
        PostToolActionAdapterConfig(
            input_dim=SEMANTIC_ONLY_FEATURE_DIM,
            hidden_dim=16,
            semantic_only=True,
        )
    )
    update, operation = model(torch.zeros(3, SEMANTIC_ONLY_FEATURE_DIM))
    assert update.shape == (3,)
    assert operation.shape == (3, 3)
