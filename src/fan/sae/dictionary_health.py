from __future__ import annotations

import torch
import torch.nn.functional as F


def dictionary_health(z: torch.Tensor, x: torch.Tensor, x_hat: torch.Tensor, decoder_weight: torch.Tensor, dead_threshold: float = 1e-5) -> dict[str, float]:
    freq = (z > 0).float().mean(dim=0)
    explained = 1.0 - torch.sum((x - x_hat) ** 2) / torch.sum((x - x.mean(dim=0, keepdim=True)) ** 2).clamp_min(1e-8)
    dirs = F.normalize(decoder_weight.T, dim=-1)
    gram = dirs @ dirs.T
    off = gram - torch.diag_embed(torch.diagonal(gram))
    return {
        "L0": float((z > 0).float().sum(dim=-1).mean().detach().cpu()),
        "dead_feature_fraction": float((freq < dead_threshold).float().mean().detach().cpu()),
        "explained_variance": float(explained.detach().cpu()),
        "decoder_coherence": float(off.abs().max().detach().cpu()),
    }
