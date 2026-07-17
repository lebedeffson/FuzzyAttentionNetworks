from __future__ import annotations

import torch


def ranked_concepts(contributions: torch.Tensor, descending: bool = True) -> torch.Tensor:
    return torch.argsort(contributions, dim=-1, descending=descending)


def remove_contributions(contributions: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    out = contributions.clone()
    out.scatter_(1, indices, 0.0)
    return out


def insert_contributions(contributions: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(contributions)
    out.scatter_(1, indices, contributions.gather(1, indices))
    return out


def random_indices(batch: int, n_concepts: int, k: int, generator: torch.Generator | None = None, device=None) -> torch.Tensor:
    rows = [torch.randperm(n_concepts, generator=generator, device=device)[:k] for _ in range(batch)]
    return torch.stack(rows, dim=0)
