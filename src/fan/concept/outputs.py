from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class ConceptFANOutput:
    logit: torch.Tensor
    probability: torch.Tensor
    latent: torch.Tensor
    concepts: torch.Tensor
    memberships: torch.Tensor
    concept_weights: torch.Tensor
    concept_contributions: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "logit": self.logit,
            "probability": self.probability,
            "latent": self.latent,
            "concepts": self.concepts,
            "memberships": self.memberships,
            "concept_weights": self.concept_weights,
            "concept_contributions": self.concept_contributions,
        }

