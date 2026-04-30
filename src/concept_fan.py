"""Concept-level FAN models used by the ICP safety-critical IDSS paper.

This module is intentionally separate from the repository's earlier multimodal
FAN code.  The paper studies concept-mediated time-series decision support:
latent state -> grounded concepts -> fuzzy concept attention -> decision.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesEncoder(nn.Module):
    """Compact 1D CNN encoder for multivariate temporal windows."""

    def __init__(self, in_channels: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x).squeeze(-1))


class GaussianConceptMembership(nn.Module):
    """Learnable Gaussian membership for concept activations in [0, 1]."""

    def __init__(self, num_concepts: int, center: float = 0.5, width: float = 0.25):
        super().__init__()
        self.centers = nn.Parameter(torch.full((num_concepts,), center))
        self.widths = nn.Parameter(torch.full((num_concepts,), width))

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        widths = self.widths.abs().clamp_min(1e-3)
        return torch.exp(-((concepts - self.centers) ** 2) / (2.0 * widths ** 2))


class ConceptFANBlock(nn.Module):
    """Fuzzy attention over grounded concept memberships."""

    def __init__(self, num_concepts: int, hidden_dim: int = 64, temperature: float = 1.0):
        super().__init__()
        self.membership = GaussianConceptMembership(num_concepts)
        self.temperature = temperature
        self.attn = nn.Sequential(
            nn.Linear(num_concepts * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_concepts),
        )

    def forward(self, concepts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        memberships = self.membership(concepts)
        logits = self.attn(torch.cat([concepts, memberships], dim=1))
        alpha = torch.softmax(logits / self.temperature, dim=1)
        evidence = alpha * memberships
        return evidence, alpha, memberships


class SafetyCriticalConceptFAN(nn.Module):
    """Concept-oriented FAN classifier used as the proposed model."""

    def __init__(self, in_channels: int, num_concepts: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = TimeSeriesEncoder(in_channels, hidden_dim, latent_dim)
        self.concepts = nn.Sequential(nn.Linear(latent_dim, num_concepts), nn.Sigmoid())
        self.fan = ConceptFANBlock(num_concepts, hidden_dim=hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(num_concepts, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_from_concepts(self, concepts: torch.Tensor):
        evidence, alpha, memberships = self.fan(concepts)
        logit = self.head(evidence).squeeze(1)
        return logit, alpha, memberships

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encoder(x)
        concepts = self.concepts(z)
        logit, alpha, memberships = self.forward_from_concepts(concepts)
        return {
            "logit": logit,
            "z": z,
            "concepts": concepts,
            "alpha": alpha,
            "membership": memberships,
        }


class ConceptBottleneckBaseline(nn.Module):
    """CBM baseline: same encoder and concept head, no FAN aggregation."""

    def __init__(self, in_channels: int, num_concepts: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = TimeSeriesEncoder(in_channels, hidden_dim, latent_dim)
        self.concepts = nn.Sequential(nn.Linear(latent_dim, num_concepts), nn.Sigmoid())
        self.head = nn.Linear(num_concepts, 1)

    def forward_from_concepts(self, concepts: torch.Tensor):
        logit = self.head(concepts).squeeze(1)
        alpha = torch.softmax(concepts.abs(), dim=1)
        return logit, alpha, concepts

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encoder(x)
        concepts = self.concepts(z)
        logit, alpha, memberships = self.forward_from_concepts(concepts)
        return {
            "logit": logit,
            "z": z,
            "concepts": concepts,
            "alpha": alpha,
            "membership": memberships,
        }


class CNNBaseline(nn.Module):
    """Black-box 1D CNN baseline without concept mediation."""

    def __init__(self, in_channels: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = TimeSeriesEncoder(in_channels, hidden_dim, latent_dim)
        self.head = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"logit": self.head(self.encoder(x)).squeeze(1)}


class TransformerBaseline(nn.Module):
    """Lightweight temporal Transformer baseline."""

    def __init__(self, in_channels: int, hidden_dim: int, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.input_proj(x.transpose(1, 2))
        h = self.encoder(h).mean(dim=1)
        return {"logit": self.head(h).squeeze(1)}


def structural_alignment_loss(z: torch.Tensor, concepts: torch.Tensor) -> torch.Tensor:
    """MSE between mini-batch cosine similarity matrices."""

    z_norm = F.normalize(z, dim=1)
    c_norm = F.normalize(concepts, dim=1)
    return F.mse_loss(z_norm @ z_norm.T, c_norm @ c_norm.T)
