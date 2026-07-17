from __future__ import annotations

import torch


def product_tnorm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * b


def softmin_tnorm(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    stacked = torch.stack([a, b], dim=0)
    weights = torch.softmax(-stacked / float(temperature), dim=0)
    return (weights * stacked).sum(dim=0)
