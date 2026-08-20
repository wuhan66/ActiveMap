"""Capacity-matched generic and edit-conditioned evidence selectors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from activemap.features import EVIDENCE_DIM, HYPOTHESIS_DIM, STATE_DIM


@dataclass(frozen=True)
class SelectorConfig:
    evidence_dim: int = EVIDENCE_DIM
    hypothesis_dim: int = HYPOTHESIS_DIM
    state_dim: int = STATE_DIM
    hidden_dim: int = 128
    dropout: float = 0.10
    condition_on_hypothesis: bool = True
    allow_stop: bool = True
    decision_mode: str = "joint"
    candidate_value_head: bool = False
    candidate_decision_mode: str = "rank"
    terminal_gate_mode: str = "context"

    def __post_init__(self) -> None:
        if self.decision_mode not in {"joint", "two_stage"}:
            raise ValueError("decision_mode must be joint or two_stage")
        if self.candidate_decision_mode not in {"rank", "value", "hybrid"}:
            raise ValueError("candidate_decision_mode must be rank, value, or hybrid")
        if self.candidate_decision_mode != "rank" and not self.candidate_value_head:
            raise ValueError(
                "value-based candidate decisions require candidate_value_head"
            )
        if self.terminal_gate_mode not in {"context", "value", "hybrid"}:
            raise ValueError("terminal_gate_mode must be context, value, or hybrid")
        if self.terminal_gate_mode != "context" and not self.candidate_value_head:
            raise ValueError(
                "value-aware terminal gating requires candidate_value_head"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mlp(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
    )


class EvidenceSelector(nn.Module):
    """Score a variable candidate set and an optional STOP action."""

    def __init__(self, config: SelectorConfig) -> None:
        super().__init__()
        self.config = config
        self.evidence_encoder = _mlp(
            config.evidence_dim, config.hidden_dim, config.dropout
        )
        self.hypothesis_encoder = _mlp(
            config.hypothesis_dim, config.hidden_dim, config.dropout
        )
        self.state_encoder = _mlp(config.state_dim, config.hidden_dim, config.dropout)
        self.context_projection = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.candidate_value_head = (
            nn.Sequential(
                nn.Linear(config.hidden_dim * 4, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 1),
            )
            if config.candidate_value_head
            else None
        )
        self.stop_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward_training_components(
        self,
        evidence: Tensor,
        hypothesis: Tensor,
        state: Tensor,
        evidence_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        if evidence.ndim != 3:
            raise ValueError("evidence must have shape [batch, candidates, features]")
        encoded_evidence = self.evidence_encoder(evidence)
        encoded_state = self.state_encoder(state)
        encoded_hypothesis = self.hypothesis_encoder(hypothesis)
        if not self.config.condition_on_hypothesis:
            encoded_hypothesis = torch.zeros_like(encoded_hypothesis)
        context = self.context_projection(
            torch.cat([encoded_state, encoded_hypothesis], dim=-1)
        )
        expanded_context = context.unsqueeze(1).expand(-1, evidence.shape[1], -1)
        interactions = torch.cat(
            [
                encoded_evidence,
                expanded_context,
                encoded_evidence * expanded_context,
                torch.abs(encoded_evidence - expanded_context),
            ],
            dim=-1,
        )
        evidence_logits = self.score_head(interactions).squeeze(-1)
        candidate_values = (
            self.candidate_value_head(interactions).squeeze(-1)
            if self.candidate_value_head is not None
            else None
        )
        if evidence_mask is not None:
            evidence_logits = evidence_logits.masked_fill(~evidence_mask, -torch.inf)
            if candidate_values is not None:
                candidate_values = candidate_values.masked_fill(
                    ~evidence_mask, -torch.inf
                )
        gate_or_stop_logit = self.stop_head(context).squeeze(-1)
        return evidence_logits, candidate_values, gate_or_stop_logit

    def candidate_decision_scores(
        self, rank_scores: Tensor, candidate_values: Tensor | None
    ) -> Tensor:
        if self.config.candidate_decision_mode == "rank":
            return rank_scores
        if candidate_values is None:
            raise RuntimeError("candidate value scores are unavailable")
        if self.config.candidate_decision_mode == "value":
            return candidate_values
        return rank_scores + candidate_values

    def terminal_gate_score(
        self, candidate_values: Tensor | None, context_gate: Tensor
    ) -> Tensor:
        if self.config.terminal_gate_mode == "context":
            return context_gate
        if candidate_values is None:
            raise RuntimeError("candidate value scores are unavailable")
        value_gate = candidate_values.max(dim=-1).values
        if self.config.terminal_gate_mode == "value":
            return value_gate
        return context_gate + value_gate

    def forward_components(
        self,
        evidence: Tensor,
        hypothesis: Tensor,
        state: Tensor,
        evidence_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        (
            rank_scores,
            candidate_values,
            gate_or_stop_logit,
        ) = self.forward_training_components(evidence, hypothesis, state, evidence_mask)
        return (
            self.candidate_decision_scores(rank_scores, candidate_values),
            self.terminal_gate_score(candidate_values, gate_or_stop_logit),
        )

    def forward(
        self,
        evidence: Tensor,
        hypothesis: Tensor,
        state: Tensor,
        evidence_mask: Tensor | None = None,
    ) -> Tensor:
        evidence_logits, gate_or_stop_logit = self.forward_components(
            evidence, hypothesis, state, evidence_mask
        )
        if not self.config.allow_stop:
            return evidence_logits
        if self.config.decision_mode == "two_stage":
            best_evidence_logit = torch.max(evidence_logits, dim=-1).values
            stop_logit = best_evidence_logit - gate_or_stop_logit
        else:
            stop_logit = gate_or_stop_logit
        return torch.cat([evidence_logits, stop_logit[:, None]], dim=1)

    def parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
