from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FuzzyConceptAggregator(nn.Module):
    """FAN concept weighting block.

    The decision evidence is strictly ``alpha * membership``. Scores may use
    concepts and memberships, but the final decision head receives only
    contributions.
    """

    def __init__(self, n_concepts: int, hidden_dim: int = 32, temperature: float = 1.0):
        super().__init__()
        self.n_concepts = int(n_concepts)
        self.score_net = nn.Sequential(
            nn.Linear(n_concepts + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.raw_temperature = nn.Parameter(torch.tensor(float(temperature)).log())

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.raw_temperature) + 1e-6

    def forward(self, concepts: torch.Tensor, memberships: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if concepts.shape != memberships.shape:
            raise ValueError("Concept and membership tensors must have identical shapes")
        bsz, n_concepts = concepts.shape
        if n_concepts != self.n_concepts:
            raise ValueError(f"Expected {self.n_concepts} concepts, got {n_concepts}")
        expanded_concepts = concepts.unsqueeze(1).expand(bsz, n_concepts, n_concepts)
        score_input = torch.cat([expanded_concepts, memberships.unsqueeze(-1)], dim=-1)
        scores = self.score_net(score_input).squeeze(-1)
        weights = torch.softmax(scores / self.temperature, dim=-1)
        contributions = weights * memberships
        return scores, weights, contributions

