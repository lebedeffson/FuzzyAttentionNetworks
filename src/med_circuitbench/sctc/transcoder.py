from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseClinicalTranscoder(nn.Module):
    def __init__(self, d_model: int = 128, n_features: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d_model, n_features), nn.ReLU())
        self.decoder = nn.Linear(n_features, d_model)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encoder(h)
        a_hat = self.decoder(z)
        return {"z": z, "a_hat": a_hat}


@dataclass(frozen=True)
class SCTCLossParts:
    total: torch.Tensor
    reconstruction: torch.Tensor
    sparse: torch.Tensor
    behavior: torch.Tensor
    temporal: torch.Tensor


def temporal_huber_loss(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if z.shape[1] < 2:
        return torch.zeros((), device=z.device, dtype=z.dtype)
    dz = z[:, 1:] - z[:, :-1]
    dx = x[:, 1:] - x[:, :-1]
    weights = torch.exp(-torch.linalg.vector_norm(dx, dim=-1) / (x.shape[-1] ** 0.5))
    huber = F.huber_loss(dz, torch.zeros_like(dz), delta=1.0, reduction="none").mean(dim=-1)
    return (weights * huber).mean()


def sctc_loss(
    a_hat: torch.Tensor,
    a: torch.Tensor,
    z: torch.Tensor,
    logit_hat: torch.Tensor,
    logit: torch.Tensor,
    x: torch.Tensor | None = None,
    lambda_1: float = 1e-4,
    lambda_b: float = 0.1,
    lambda_t: float = 0.0,
    temporal_override: torch.Tensor | None = None,
) -> SCTCLossParts:
    rec = F.mse_loss(a_hat, a)
    sparse = z.mean()
    behavior = F.l1_loss(logit_hat, logit)
    if temporal_override is not None:
        temporal = temporal_override.to(device=a.device, dtype=a.dtype)
    elif x is not None:
        temporal = temporal_huber_loss(z, x)
    else:
        temporal = torch.zeros((), device=a.device, dtype=a.dtype)
    total = rec + lambda_1 * sparse + lambda_b * behavior + lambda_t * temporal
    return SCTCLossParts(total, rec, sparse, behavior, temporal)


def support_fraction(z: torch.Tensor) -> float:
    return float((z > 0).float().mean().item())
