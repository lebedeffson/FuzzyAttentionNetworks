from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from .aggregator import FuzzyConceptAggregator
from .memberships import MembershipLayer


@dataclass
class TemporalConceptFANOutput:
    logit: torch.Tensor
    probability: torch.Tensor
    latent: torch.Tensor
    latent_sequence: torch.Tensor
    concept_trajectories: torch.Tensor
    temporal_concept_weights: torch.Tensor
    concept_summaries: torch.Tensor
    memberships: torch.Tensor
    concept_weights: torch.Tensor
    concept_contributions: torch.Tensor

    @property
    def concepts(self) -> torch.Tensor:
        return self.concept_summaries

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "logit": self.logit,
            "probability": self.probability,
            "latent": self.latent,
            "latent_sequence": self.latent_sequence,
            "concepts": self.concept_summaries,
            "concept_trajectories": self.concept_trajectories,
            "temporal_concept_weights": self.temporal_concept_weights,
            "concept_summaries": self.concept_summaries,
            "memberships": self.memberships,
            "concept_weights": self.concept_weights,
            "concept_contributions": self.concept_contributions,
        }


class SequenceTemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        sequence_length: int,
        dropout: float = 0.1,
        num_layers: int = 2,
        nhead: int | None = None,
        dim_feedforward: int | None = None,
    ):
        super().__init__()
        if nhead is None:
            nhead = max(1, min(4, latent_dim // 16))
        if dim_feedforward is None:
            dim_feedforward = max(64, latent_dim * 2)
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.pos = nn.Parameter(torch.zeros(1, sequence_length, latent_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.pos[:, : x.shape[1]]
        return self.encoder(h)


class TokenConceptProjector(nn.Module):
    def __init__(self, latent_dim: int, n_concepts: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, n_concepts),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class TemporalConceptAggregator(nn.Module):
    """Per-concept temporal attention over concept trajectories only."""

    def __init__(self, n_concepts: int, sequence_length: int):
        super().__init__()
        self.n_concepts = int(n_concepts)
        self.sequence_length = int(sequence_length)
        self.value_weight = nn.Parameter(torch.ones(n_concepts))
        self.time_weight = nn.Parameter(torch.zeros(n_concepts))
        self.bias = nn.Parameter(torch.zeros(n_concepts))
        time = torch.linspace(0.0, 1.0, sequence_length)
        self.register_buffer("time", time.view(1, sequence_length, 1))

    def forward(self, concept_trajectories: torch.Tensor, mode: str = "attention") -> tuple[torch.Tensor, torch.Tensor]:
        if concept_trajectories.ndim != 3:
            raise ValueError("Expected concept trajectories [B,T,K]")
        if concept_trajectories.shape[-1] != self.n_concepts:
            raise ValueError(f"Expected {self.n_concepts} concepts, got {concept_trajectories.shape[-1]}")
        if mode == "static":
            weights = torch.zeros_like(concept_trajectories)
            weights[:, -1, :] = 1.0
            return concept_trajectories[:, -1, :], weights
        scores = concept_trajectories * self.value_weight.view(1, 1, -1)
        scores = scores + self.time * self.time_weight.view(1, 1, -1) + self.bias.view(1, 1, -1)
        weights = torch.softmax(scores, dim=1)
        summaries = (weights * concept_trajectories).sum(dim=1)
        return summaries, weights


class TemporalConceptFANModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        latent_dim: int,
        n_concepts: int,
        membership: str = "mixed",
        oracle: bool = False,
        temporal_mode: str = "attention",
        dropout: float = 0.1,
        encoder_layers: int = 2,
        encoder_heads: int | None = None,
        encoder_ffn: int | None = None,
    ):
        super().__init__()
        self.oracle = bool(oracle)
        self.temporal_mode = temporal_mode
        self.encoder = SequenceTemporalEncoder(
            input_dim,
            latent_dim,
            sequence_length,
            dropout,
            num_layers=encoder_layers,
            nhead=encoder_heads,
            dim_feedforward=encoder_ffn,
        )
        self.projector = TokenConceptProjector(latent_dim, n_concepts)
        self.temporal_aggregator = TemporalConceptAggregator(n_concepts, sequence_length)
        self.membership = MembershipLayer(n_concepts, membership)
        self.aggregator = FuzzyConceptAggregator(n_concepts)
        self.decision_head = nn.Linear(n_concepts, 1)

    def freeze_concept_path(self) -> None:
        for module in [self.encoder, self.projector]:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_concept_path(self) -> None:
        for module in [self.encoder, self.projector]:
            for param in module.parameters():
                param.requires_grad = True

    def decision_from_contributions(self, contributions: torch.Tensor) -> torch.Tensor:
        return self.decision_head(contributions).squeeze(-1)

    def forward(self, x: torch.Tensor, concept_sequence_override: torch.Tensor | None = None) -> TemporalConceptFANOutput:
        h = self.encoder(x)
        if self.oracle:
            if concept_sequence_override is None:
                raise ValueError("Oracle temporal FAN requires concept_sequence_override")
            trajectories = concept_sequence_override.float()
        else:
            trajectories = self.projector(h)
        summaries, temporal_weights = self.temporal_aggregator(trajectories, self.temporal_mode)
        memberships = self.membership(summaries)
        _, concept_weights, contributions = self.aggregator(summaries, memberships)
        logit = self.decision_from_contributions(contributions)
        return TemporalConceptFANOutput(
            logit=logit,
            probability=torch.sigmoid(logit),
            latent=h.mean(dim=1),
            latent_sequence=h,
            concept_trajectories=trajectories,
            temporal_concept_weights=temporal_weights,
            concept_summaries=summaries,
            memberships=memberships,
            concept_weights=concept_weights,
            concept_contributions=contributions,
        )
