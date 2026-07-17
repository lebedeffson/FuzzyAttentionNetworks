from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from fan.concept.temporal import SequenceTemporalEncoder, TemporalConceptAggregator, TokenConceptProjector


MODEL_ARMS = [
    "ConceptFAN-NoAlpha",
    "PureNoFuzzy",
    "PlainTransformer",
    "ConceptFAN-StabilityReg",
    "TemporalCEM",
]
CONCEPT_ARMS = ["ConceptFAN-NoAlpha", "PureNoFuzzy", "ConceptFAN-StabilityReg", "TemporalCEM"]
STABILITY_ANALYSIS_ARMS = ["ConceptFAN-NoAlpha", "PureNoFuzzy", "ConceptFAN-StabilityReg"]


@dataclass
class RealModelOutput:
    logit: torch.Tensor
    probability: torch.Tensor
    concept_trajectories: torch.Tensor | None
    concept_summaries: torch.Tensor | None
    contributions: torch.Tensor | None
    bias: torch.Tensor
    latent_sequence: torch.Tensor
    evidence: torch.Tensor | None = None

    def as_dict(self) -> dict[str, torch.Tensor | None]:
        return {
            "logit": self.logit,
            "probability": self.probability,
            "concept_trajectories": self.concept_trajectories,
            "concept_summaries": self.concept_summaries,
            "contributions": self.contributions,
            "bias": self.bias,
            "latent_sequence": self.latent_sequence,
            "evidence": self.evidence,
        }


class MonotoneFuzzyEvidence(nn.Module):
    """Concept-wise monotone fuzzy evidence with an exact identity bypass."""

    def __init__(self, n_concepts: int):
        super().__init__()
        self.center = nn.Parameter(torch.full((n_concepts,), 0.5))
        self.raw_width = nn.Parameter(torch.full((n_concepts,), -1.5))

    @property
    def width(self) -> torch.Tensor:
        return F.softplus(self.raw_width) + 1e-3

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((concepts - self.center) / self.width)


class AdditiveConceptModel(nn.Module):
    """Shared architecture for ConceptFAN and the pure no-fuzzy ablation."""

    def __init__(self, config: dict, input_dim: int, *, fuzzy: bool):
        super().__init__()
        model = config["model"]
        n_concepts = int(model["n_concepts"])
        self.fuzzy = bool(fuzzy)
        self.encoder = SequenceTemporalEncoder(
            input_dim=input_dim,
            latent_dim=int(model["latent_dim"]),
            sequence_length=int(model["sequence_length"]),
            dropout=float(model["dropout"]),
            num_layers=int(model["layers"]),
            nhead=int(model["heads"]),
            dim_feedforward=int(model["ffn"]),
        )
        self.projector = TokenConceptProjector(int(model["latent_dim"]), n_concepts)
        self.temporal_aggregator = TemporalConceptAggregator(n_concepts, int(model["sequence_length"]))
        self.fuzzy_evidence = MonotoneFuzzyEvidence(n_concepts)
        self.decision_weight = nn.Parameter(torch.empty(n_concepts))
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.xavier_uniform_(self.decision_weight.view(1, -1))

    def initialize_prevalence(self, prevalence: float) -> None:
        value = min(max(float(prevalence), 1e-4), 1.0 - 1e-4)
        with torch.no_grad():
            self.bias.copy_(torch.logit(torch.tensor(value, device=self.bias.device)))

    def forward_from_concepts(
        self,
        latent_sequence: torch.Tensor,
        trajectories: torch.Tensor,
        summaries: torch.Tensor,
        concept_mask: torch.Tensor | None = None,
    ) -> RealModelOutput:
        evidence = self.fuzzy_evidence(summaries) if self.fuzzy else summaries
        if concept_mask is not None:
            evidence = evidence * concept_mask.to(dtype=evidence.dtype)
        contributions = evidence * self.decision_weight.view(1, -1)
        logit = self.bias + contributions.sum(dim=-1)
        return RealModelOutput(
            logit=logit,
            probability=torch.sigmoid(logit),
            concept_trajectories=trajectories,
            concept_summaries=summaries,
            contributions=contributions,
            bias=self.bias,
            latent_sequence=latent_sequence,
            evidence=evidence,
        )

    def forward(self, x: torch.Tensor, concept_mask: torch.Tensor | None = None) -> RealModelOutput:
        latent = self.encoder(x)
        trajectories = self.projector(latent)
        summaries, _ = self.temporal_aggregator(trajectories, "attention")
        return self.forward_from_concepts(latent, trajectories, summaries, concept_mask)


class PlainTransformerModel(nn.Module):
    def __init__(self, config: dict, input_dim: int):
        super().__init__()
        model = config["model"]
        self.encoder = SequenceTemporalEncoder(
            input_dim=input_dim,
            latent_dim=int(model["latent_dim"]),
            sequence_length=int(model["sequence_length"]),
            dropout=float(model["dropout"]),
            num_layers=int(model["layers"]),
            nhead=int(model["heads"]),
            dim_feedforward=int(model["ffn"]),
        )
        self.temporal_score = nn.Linear(int(model["latent_dim"]), 1)
        self.head = nn.Linear(int(model["latent_dim"]), 1)

    @property
    def bias(self) -> torch.Tensor:
        return self.head.bias.squeeze(0)

    def forward(self, x: torch.Tensor, concept_mask: torch.Tensor | None = None) -> RealModelOutput:
        del concept_mask
        latent = self.encoder(x)
        weights = torch.softmax(self.temporal_score(latent).squeeze(-1), dim=1)
        pooled = (weights.unsqueeze(-1) * latent).sum(dim=1)
        logit = self.head(pooled).squeeze(-1)
        return RealModelOutput(
            logit=logit,
            probability=torch.sigmoid(logit),
            concept_trajectories=None,
            concept_summaries=None,
            contributions=None,
            bias=self.bias,
            latent_sequence=latent,
            evidence=None,
        )


class TemporalCEMModel(nn.Module):
    def __init__(self, config: dict, input_dim: int):
        super().__init__()
        model = config["model"]
        latent_dim = int(model["latent_dim"])
        n_concepts = int(model["n_concepts"])
        self.encoder = SequenceTemporalEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            sequence_length=int(model["sequence_length"]),
            dropout=float(model["dropout"]),
            num_layers=int(model["layers"]),
            nhead=int(model["heads"]),
            dim_feedforward=int(model["ffn"]),
        )
        self.projector = TokenConceptProjector(latent_dim, n_concepts)
        self.temporal_aggregator = TemporalConceptAggregator(n_concepts, int(model["sequence_length"]))
        embedding_dim = max(16, latent_dim // n_concepts)
        self.positive_embedding = nn.Parameter(torch.empty(n_concepts, embedding_dim))
        self.negative_embedding = nn.Parameter(torch.empty(n_concepts, embedding_dim))
        self.classifier_weight = nn.Parameter(torch.empty(n_concepts, embedding_dim))
        self.gate = nn.Sequential(nn.Linear(1, 8), nn.GELU(), nn.Linear(8, 1), nn.Sigmoid())
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.xavier_uniform_(self.positive_embedding)
        nn.init.xavier_uniform_(self.negative_embedding)
        nn.init.xavier_uniform_(self.classifier_weight)

    def initialize_prevalence(self, prevalence: float) -> None:
        value = min(max(float(prevalence), 1e-4), 1.0 - 1e-4)
        with torch.no_grad():
            self.bias.copy_(torch.logit(torch.tensor(value, device=self.bias.device)))

    def forward(self, x: torch.Tensor, concept_mask: torch.Tensor | None = None) -> RealModelOutput:
        latent = self.encoder(x)
        trajectories = self.projector(latent)
        summaries, _ = self.temporal_aggregator(trajectories, "attention")
        probabilities = summaries.unsqueeze(-1)
        embeddings = probabilities * self.positive_embedding.unsqueeze(0) + (1.0 - probabilities) * self.negative_embedding.unsqueeze(0)
        gates = self.gate(probabilities)
        contributions = (gates * embeddings * self.classifier_weight.unsqueeze(0)).sum(dim=-1)
        if concept_mask is not None:
            contributions = contributions * concept_mask.to(dtype=contributions.dtype)
        logit = self.bias + contributions.sum(dim=-1)
        return RealModelOutput(
            logit=logit,
            probability=torch.sigmoid(logit),
            concept_trajectories=trajectories,
            concept_summaries=summaries,
            contributions=contributions,
            bias=self.bias,
            latent_sequence=latent,
            evidence=gates.squeeze(-1),
        )


def build_model(arm: str, config: dict, input_dim: int, prevalence: float) -> nn.Module:
    if arm == "ConceptFAN-NoAlpha" or arm == "ConceptFAN-StabilityReg":
        model: nn.Module = AdditiveConceptModel(config, input_dim, fuzzy=True)
    elif arm == "PureNoFuzzy":
        model = AdditiveConceptModel(config, input_dim, fuzzy=False)
    elif arm == "PlainTransformer":
        model = PlainTransformerModel(config, input_dim)
    elif arm == "TemporalCEM":
        model = TemporalCEMModel(config, input_dim)
    else:
        raise ValueError(f"Unknown model arm: {arm}")
    if hasattr(model, "initialize_prevalence"):
        model.initialize_prevalence(prevalence)
    return model


def masked_concept_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("Concept prediction, target, and mask shapes must match")
    element = F.smooth_l1_loss(prediction, target, reduction="none")
    denominator = mask.sum().clamp_min(1.0)
    return (element * mask).sum() / denominator


def exact_decomposition_error(output: RealModelOutput) -> torch.Tensor:
    if output.contributions is None:
        return output.logit.new_zeros(output.logit.shape)
    reconstructed = output.bias + output.contributions.sum(dim=-1)
    return torch.abs(output.logit.float() - reconstructed.float())


def normalized_contributions(contributions: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    return contributions / (contributions.abs().sum(dim=-1, keepdim=True) + epsilon)


def parameter_sha256(model: nn.Module) -> str:
    payload = io.BytesIO()
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in sorted(model.state_dict().items())}
    torch.save(state, payload)
    return hashlib.sha256(payload.getvalue()).hexdigest()
