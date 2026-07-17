from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GaussianMembership(nn.Module):
    def __init__(self, n_heads: int, n_memberships: int, head_dim: int):
        super().__init__()
        self.n_heads = int(n_heads)
        self.n_memberships = int(n_memberships)
        self.head_dim = int(head_dim)
        centers = torch.linspace(-1.0, 1.0, n_memberships).view(1, n_memberships, 1).expand(n_heads, n_memberships, head_dim)
        self.centers = nn.Parameter(centers.clone())
        self.raw_width = nn.Parameter(torch.full((n_heads, n_memberships, head_dim), -0.4))

    @property
    def widths(self) -> torch.Tensor:
        return F.softplus(self.raw_width) + 1e-4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,H,T,D] -> [B,H,T,M,D]
        return torch.exp(-0.5 * ((x.unsqueeze(3) - self.centers.view(1, self.n_heads, 1, self.n_memberships, self.head_dim)) / self.widths.view(1, self.n_heads, 1, self.n_memberships, self.head_dim)) ** 2)
