from __future__ import annotations

import torch
import torch.nn as nn


class ConceptProjector(nn.Module):
    def __init__(self, latent_dim: int, n_concepts: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, n_concepts),
            nn.Sigmoid(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)

