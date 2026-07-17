from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MembershipLayer(nn.Module):
    """Concept-level fuzzy membership functions.

    Input and output shape is ``[batch, n_concepts]``. Width parameters are
    positive by construction through softplus.
    """

    def __init__(self, n_concepts: int, family: str = "gaussian", epsilon: float = 1e-6):
        super().__init__()
        valid = {"gaussian", "bell", "sigmoid", "mixed"}
        if family not in valid:
            raise ValueError(f"Unknown membership family {family!r}; expected one of {sorted(valid)}")
        self.n_concepts = int(n_concepts)
        self.family = family
        self.epsilon = float(epsilon)
        self.center = nn.Parameter(torch.full((n_concepts,), 0.5))
        self.raw_delta = nn.Parameter(torch.full((n_concepts,), -0.7))
        self.raw_slope = nn.Parameter(torch.zeros(n_concepts))
        self.raw_bell_power = nn.Parameter(torch.full((n_concepts,), 0.7))
        self.mixture_logits = nn.Parameter(torch.zeros(3))

    @property
    def delta(self) -> torch.Tensor:
        return F.softplus(self.raw_delta) + self.epsilon

    def gaussian(self, c: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * ((c - self.center) / self.delta) ** 2)

    def bell(self, c: torch.Tensor) -> torch.Tensor:
        power = F.softplus(self.raw_bell_power) + self.epsilon
        return 1.0 / (1.0 + torch.abs((c - self.center) / self.delta) ** (2.0 * power))

    def sigmoid(self, c: torch.Tensor) -> torch.Tensor:
        slope = F.softplus(self.raw_slope) + self.epsilon
        return torch.sigmoid((c - self.center) / self.delta * slope)

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        if concepts.shape[-1] != self.n_concepts:
            raise ValueError(f"Expected {self.n_concepts} concepts, got {concepts.shape[-1]}")
        if self.family == "gaussian":
            mu = self.gaussian(concepts)
        elif self.family == "bell":
            mu = self.bell(concepts)
        elif self.family == "sigmoid":
            mu = self.sigmoid(concepts)
        else:
            weights = torch.softmax(self.mixture_logits, dim=0)
            mu = weights[0] * self.gaussian(concepts) + weights[1] * self.bell(concepts) + weights[2] * self.sigmoid(concepts)
        return torch.clamp(mu, 0.0, 1.0)

