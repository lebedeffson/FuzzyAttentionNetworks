from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from .outputs import ConceptFANOutput


@dataclass(frozen=True)
class ConceptFANLossConfig:
    lambda_c: float = 1.0
    lambda_a: float = 0.05
    lambda_s: float = 0.01


def _cosine_similarity_matrix(x: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=-1)
    return x @ x.transpose(0, 1)


def entropy(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -(weights * torch.log(weights + eps)).sum(dim=-1).mean()


def concept_fan_loss(
    output: ConceptFANOutput,
    target: torch.Tensor,
    concept_targets: torch.Tensor,
    cfg: ConceptFANLossConfig | None = None,
) -> Dict[str, torch.Tensor]:
    cfg = cfg or ConceptFANLossConfig()
    task_loss = F.binary_cross_entropy_with_logits(output.logit, target.float())
    concept_loss = F.mse_loss(output.concepts, concept_targets.float())
    align_loss = F.mse_loss(_cosine_similarity_matrix(output.latent), _cosine_similarity_matrix(output.concepts))
    sparsity_loss = entropy(output.concept_weights)
    total = task_loss + cfg.lambda_c * concept_loss + cfg.lambda_a * align_loss + cfg.lambda_s * sparsity_loss
    return {
        "task_loss": task_loss,
        "concept_loss": concept_loss,
        "alignment_loss": align_loss,
        "sparsity_loss": sparsity_loss,
        "total_loss": total,
    }

