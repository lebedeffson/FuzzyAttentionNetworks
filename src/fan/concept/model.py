from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .aggregator import FuzzyConceptAggregator
from .memberships import MembershipLayer
from .outputs import ConceptFANOutput
from .projector import ConceptProjector


@dataclass(frozen=True)
class ConceptFANConfig:
    input_dim: int
    sequence_length: int = 36
    latent_dim: int = 128
    n_concepts: int = 5
    membership: str = "mixed"
    dropout: float = 0.1


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, sequence_length: int, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.pos = nn.Parameter(torch.zeros(1, sequence_length, latent_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=max(1, min(4, latent_dim // 16)),
            dim_feedforward=max(64, latent_dim * 2),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.pos[:, : x.shape[1]]
        h = self.encoder(h)
        return h.mean(dim=1)


class ConceptFANModel(nn.Module):
    def __init__(self, cfg: ConceptFANConfig, oracle: bool = False):
        super().__init__()
        self.cfg = cfg
        self.oracle = bool(oracle)
        self.encoder = TemporalEncoder(cfg.input_dim, cfg.latent_dim, cfg.sequence_length, cfg.dropout)
        self.projector = ConceptProjector(cfg.latent_dim, cfg.n_concepts)
        self.membership = MembershipLayer(cfg.n_concepts, cfg.membership)
        self.aggregator = FuzzyConceptAggregator(cfg.n_concepts)
        self.decision_head = nn.Linear(cfg.n_concepts, 1)

    def decision_from_contributions(self, contributions: torch.Tensor) -> torch.Tensor:
        return self.decision_head(contributions).squeeze(-1)

    def forward(self, x: torch.Tensor, concept_override: torch.Tensor | None = None) -> ConceptFANOutput:
        latent = self.encoder(x)
        if self.oracle:
            if concept_override is None:
                raise ValueError("Oracle ConceptFAN requires concept_override")
            concepts = concept_override.float()
        else:
            concepts = self.projector(latent)
        memberships = self.membership(concepts)
        _, weights, contributions = self.aggregator(concepts, memberships)
        logit = self.decision_from_contributions(contributions)
        return ConceptFANOutput(
            logit=logit,
            probability=torch.sigmoid(logit),
            latent=latent,
            concepts=concepts,
            memberships=memberships,
            concept_weights=weights,
            concept_contributions=contributions,
        )


class OracleConceptFAN(ConceptFANModel):
    def __init__(self, cfg: ConceptFANConfig):
        super().__init__(cfg, oracle=True)


class PredictedConceptFAN(ConceptFANModel):
    def __init__(self, cfg: ConceptFANConfig):
        super().__init__(cfg, oracle=False)

