from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregator import FuzzyConceptAggregator
from .memberships import MembershipLayer


@dataclass
class TemporalConceptFANOutput:
    logit: torch.Tensor
    probability: torch.Tensor
    latent: torch.Tensor
    latent_sequence: torch.Tensor
    concept_trajectories: torch.Tensor
    temporal_concept_weights: torch.Tensor
    concept_summaries: torch.Tensor
    memberships: torch.Tensor
    concept_weights: torch.Tensor
    concept_contributions: torch.Tensor

    @property
    def concepts(self) -> torch.Tensor:
        return self.concept_summaries

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "logit": self.logit,
            "probability": self.probability,
            "latent": self.latent,
            "latent_sequence": self.latent_sequence,
            "concepts": self.concept_summaries,
            "concept_trajectories": self.concept_trajectories,
            "temporal_concept_weights": self.temporal_concept_weights,
            "concept_summaries": self.concept_summaries,
            "memberships": self.memberships,
            "concept_weights": self.concept_weights,
            "concept_contributions": self.concept_contributions,
        }


class SequenceTemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        sequence_length: int,
        dropout: float = 0.1,
        num_layers: int = 2,
        nhead: int | None = None,
        dim_feedforward: int | None = None,
    ):
        super().__init__()
        if nhead is None:
            nhead = max(1, min(4, latent_dim // 16))
        if dim_feedforward is None:
            dim_feedforward = max(64, latent_dim * 2)
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.pos = nn.Parameter(torch.zeros(1, sequence_length, latent_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.pos[:, : x.shape[1]]
        return self.encoder(h)


class TokenConceptProjector(nn.Module):
    def __init__(self, latent_dim: int, n_concepts: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, n_concepts),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class TemporalConceptAggregator(nn.Module):
    """Per-concept temporal attention over concept trajectories only."""

    def __init__(self, n_concepts: int, sequence_length: int):
        super().__init__()
        self.n_concepts = int(n_concepts)
        self.sequence_length = int(sequence_length)
        self.value_weight = nn.Parameter(torch.ones(n_concepts))
        self.time_weight = nn.Parameter(torch.zeros(n_concepts))
        self.bias = nn.Parameter(torch.zeros(n_concepts))
        time = torch.linspace(0.0, 1.0, sequence_length)
        self.register_buffer("time", time.view(1, sequence_length, 1))

    def forward(self, concept_trajectories: torch.Tensor, mode: str = "attention") -> tuple[torch.Tensor, torch.Tensor]:
        if concept_trajectories.ndim != 3:
            raise ValueError("Expected concept trajectories [B,T,K]")
        if concept_trajectories.shape[-1] != self.n_concepts:
            raise ValueError(f"Expected {self.n_concepts} concepts, got {concept_trajectories.shape[-1]}")
        if mode == "static":
            weights = torch.zeros_like(concept_trajectories)
            weights[:, -1, :] = 1.0
            return concept_trajectories[:, -1, :], weights
        scores = concept_trajectories * self.value_weight.view(1, 1, -1)
        scores = scores + self.time * self.time_weight.view(1, 1, -1) + self.bias.view(1, 1, -1)
        weights = torch.softmax(scores, dim=1)
        summaries = (weights * concept_trajectories).sum(dim=1)
        return summaries, weights


class TemporalConceptFANModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        latent_dim: int,
        n_concepts: int,
        membership: str = "mixed",
        oracle: bool = False,
        temporal_mode: str = "attention",
        dropout: float = 0.1,
        encoder_layers: int = 2,
        encoder_heads: int | None = None,
        encoder_ffn: int | None = None,
    ):
        super().__init__()
        self.oracle = bool(oracle)
        self.temporal_mode = temporal_mode
        self.encoder = SequenceTemporalEncoder(
            input_dim,
            latent_dim,
            sequence_length,
            dropout,
            num_layers=encoder_layers,
            nhead=encoder_heads,
            dim_feedforward=encoder_ffn,
        )
        self.projector = TokenConceptProjector(latent_dim, n_concepts)
        self.temporal_aggregator = TemporalConceptAggregator(n_concepts, sequence_length)
        self.membership = MembershipLayer(n_concepts, membership)
        self.aggregator = FuzzyConceptAggregator(n_concepts)
        self.decision_head = nn.Linear(n_concepts, 1)

    def freeze_concept_path(self) -> None:
        for module in [self.encoder, self.projector]:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_concept_path(self) -> None:
        for module in [self.encoder, self.projector]:
            for param in module.parameters():
                param.requires_grad = True

    def decision_from_contributions(self, contributions: torch.Tensor) -> torch.Tensor:
        return self.decision_head(contributions).squeeze(-1)

    def forward(self, x: torch.Tensor, concept_sequence_override: torch.Tensor | None = None) -> TemporalConceptFANOutput:
        h = self.encoder(x)
        if self.oracle:
            if concept_sequence_override is None:
                raise ValueError("Oracle temporal FAN requires concept_sequence_override")
            trajectories = concept_sequence_override.float()
        else:
            trajectories = self.projector(h)
        summaries, temporal_weights = self.temporal_aggregator(trajectories, self.temporal_mode)
        memberships = self.membership(summaries)
        _, concept_weights, contributions = self.aggregator(summaries, memberships)
        logit = self.decision_from_contributions(contributions)
        return TemporalConceptFANOutput(
            logit=logit,
            probability=torch.sigmoid(logit),
            latent=h.mean(dim=1),
            latent_sequence=h,
            concept_trajectories=trajectories,
            temporal_concept_weights=temporal_weights,
            concept_summaries=summaries,
            memberships=memberships,
            concept_weights=concept_weights,
            concept_contributions=contributions,
        )


@dataclass
class MultiSetAdditiveFANOutput:
    logit: torch.Tensor
    probability: torch.Tensor
    latent: torch.Tensor
    latent_sequence: torch.Tensor
    concept_trajectories: torch.Tensor
    temporal_concept_weights: torch.Tensor
    concept_summaries: torch.Tensor
    memberships: torch.Tensor
    membership_local_weights: torch.Tensor
    fuzzy_values: torch.Tensor
    concept_weights: torch.Tensor
    concept_evidence: torch.Tensor
    signed_decision_contributions: torch.Tensor

    @property
    def concepts(self) -> torch.Tensor:
        return self.concept_summaries

    @property
    def concept_contributions(self) -> torch.Tensor:
        return self.signed_decision_contributions

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "logit": self.logit,
            "probability": self.probability,
            "latent": self.latent,
            "latent_sequence": self.latent_sequence,
            "concepts": self.concept_summaries,
            "concept_trajectories": self.concept_trajectories,
            "temporal_concept_weights": self.temporal_concept_weights,
            "concept_summaries": self.concept_summaries,
            "memberships": self.memberships,
            "membership_local_weights": self.membership_local_weights,
            "fuzzy_values": self.fuzzy_values,
            "concept_weights": self.concept_weights,
            "concept_evidence": self.concept_evidence,
            "signed_decision_contributions": self.signed_decision_contributions,
        }


class MultiSetMembershipLayer(nn.Module):
    """Train-quantile initialized fuzzy basis with M memberships per concept."""

    def __init__(self, n_concepts: int, n_memberships: int = 3, family: str = "gaussian", epsilon: float = 1e-6):
        super().__init__()
        if n_memberships not in {3, 5}:
            raise ValueError("n_memberships must be 3 or 5")
        valid = {"gaussian", "bell", "sigmoid", "mixed"}
        if family not in valid:
            raise ValueError(f"Unknown membership family {family!r}")
        self.n_concepts = int(n_concepts)
        self.n_memberships = int(n_memberships)
        self.family = family
        self.epsilon = float(epsilon)
        centers = torch.linspace(0.2, 0.8, n_memberships).repeat(n_concepts, 1)
        self.centers = nn.Parameter(centers)
        self.raw_widths = nn.Parameter(torch.full((n_concepts, n_memberships), -1.4))
        self.raw_slope = nn.Parameter(torch.ones(n_concepts, n_memberships))
        self.raw_bell_power = nn.Parameter(torch.full((n_concepts, n_memberships), 0.7))
        self.mixture_logits = nn.Parameter(torch.zeros(3))

    @property
    def widths(self) -> torch.Tensor:
        return F.softplus(self.raw_widths) + self.epsilon

    def initialize_from_quantiles(self, train_concepts: torch.Tensor) -> None:
        if train_concepts.ndim != 2 or train_concepts.shape[1] != self.n_concepts:
            raise ValueError("Expected train concepts [N,K]")
        qs = torch.tensor([0.2, 0.5, 0.8] if self.n_memberships == 3 else [0.1, 0.3, 0.5, 0.7, 0.9], device=train_concepts.device)
        centers = torch.quantile(train_concepts.float(), qs, dim=0).T
        centers = torch.clamp(centers, 0.0, 1.0)
        diffs = torch.diff(centers, dim=1).abs()
        left = diffs[:, :1]
        right = diffs[:, -1:]
        middle = 0.5 * (diffs[:, :-1] + diffs[:, 1:]) if self.n_memberships > 2 else diffs
        widths = torch.cat([left, middle, right], dim=1)
        widths = torch.clamp(widths, min=0.05)
        with torch.no_grad():
            self.centers.copy_(centers.cpu())
            self.raw_widths.copy_(torch.log(torch.expm1(widths.cpu() - self.epsilon).clamp_min(1e-6)))

    def gaussian(self, c: torch.Tensor) -> torch.Tensor:
        x = c.unsqueeze(-1)
        return torch.exp(-0.5 * ((x - self.centers.unsqueeze(0)) / self.widths.unsqueeze(0)) ** 2)

    def bell(self, c: torch.Tensor) -> torch.Tensor:
        x = c.unsqueeze(-1)
        power = F.softplus(self.raw_bell_power).unsqueeze(0) + self.epsilon
        return 1.0 / (1.0 + torch.abs((x - self.centers.unsqueeze(0)) / self.widths.unsqueeze(0)) ** (2.0 * power))

    def sigmoid(self, c: torch.Tensor) -> torch.Tensor:
        x = c.unsqueeze(-1)
        slope = F.softplus(self.raw_slope).unsqueeze(0) + self.epsilon
        return torch.sigmoid((x - self.centers.unsqueeze(0)) / self.widths.unsqueeze(0) * slope)

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
            w = torch.softmax(self.mixture_logits, dim=0)
            mu = w[0] * self.gaussian(concepts) + w[1] * self.bell(concepts) + w[2] * self.sigmoid(concepts)
        return torch.clamp(mu, 0.0, 1.0)


class MultiSetFuzzyAggregator(nn.Module):
    def __init__(
        self,
        n_concepts: int,
        n_memberships: int,
        temperature: float = 1.0,
        alpha_mode: str = "softmax_alpha",
        gamma_init: float = 0.5,
        gamma_max: float = 0.9,
    ):
        super().__init__()
        self.n_concepts = int(n_concepts)
        self.n_memberships = int(n_memberships)
        valid = {"no_alpha", "uniform_alpha", "softmax_alpha", "residual_floor_alpha"}
        if alpha_mode not in valid:
            raise ValueError(f"Unknown alpha_mode {alpha_mode!r}")
        self.alpha_mode = alpha_mode
        self.gamma_max = float(gamma_max)
        self.local_score = nn.Sequential(nn.Linear(2, 8), nn.GELU(), nn.Linear(8, 1))
        self.concept_score = nn.Sequential(nn.Linear(n_concepts + 1, 16), nn.GELU(), nn.Linear(16, 1))
        self.raw_temperature = nn.Parameter(torch.tensor(float(temperature)).log())
        gamma = min(max(float(gamma_init), 1e-4), self.gamma_max - 1e-4)
        self.raw_gamma = nn.Parameter(torch.logit(torch.tensor(gamma / self.gamma_max)))

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.raw_temperature) + 1e-6

    @property
    def gamma(self) -> torch.Tensor:
        return self.gamma_max * torch.sigmoid(self.raw_gamma)

    def forward(
        self,
        concepts: torch.Tensor,
        memberships: torch.Tensor,
        concept_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if memberships.ndim != 3:
            raise ValueError("Expected memberships [B,K,M]")
        bsz, n_concepts, n_memberships = memberships.shape
        if n_concepts != self.n_concepts or n_memberships != self.n_memberships:
            raise ValueError("Membership shape does not match aggregator")
        local_input = torch.stack([concepts.unsqueeze(-1).expand_as(memberships), memberships], dim=-1)
        local_scores = self.local_score(local_input).squeeze(-1)
        local_weights = torch.softmax(local_scores, dim=-1)
        fuzzy_values = (local_weights * memberships).sum(dim=-1)
        expanded = concepts.unsqueeze(1).expand(bsz, n_concepts, n_concepts)
        concept_input = torch.cat([expanded, fuzzy_values.unsqueeze(-1)], dim=-1)
        concept_scores = self.concept_score(concept_input).squeeze(-1)
        mask_float = None
        if concept_mask is not None:
            if concept_mask.shape != (bsz, n_concepts):
                raise ValueError(f"Expected concept_mask {(bsz, n_concepts)}, got {tuple(concept_mask.shape)}")
            mask = concept_mask.to(device=concept_scores.device, dtype=torch.bool)
            concept_scores = concept_scores.masked_fill(~mask, -1.0e9)
            mask_float = mask.to(dtype=fuzzy_values.dtype)
        if mask_float is None:
            active = torch.ones_like(fuzzy_values)
        else:
            active = mask_float
        active_count = active.sum(dim=-1, keepdim=True).clamp_min(1.0)
        if self.alpha_mode == "no_alpha":
            concept_weights = active
        elif self.alpha_mode == "uniform_alpha":
            concept_weights = active / active_count
        else:
            softmax_weights = torch.softmax(concept_scores / self.temperature, dim=-1)
            if mask_float is not None:
                softmax_weights = softmax_weights * active
            if self.alpha_mode == "softmax_alpha":
                concept_weights = softmax_weights
            else:
                concept_weights = (1.0 - self.gamma) * (active / active_count) + self.gamma * softmax_weights
        concept_evidence = concept_weights * fuzzy_values
        return local_weights, fuzzy_values, concept_weights, concept_evidence


class AdditiveDecisionHead(nn.Module):
    def __init__(self, n_concepts: int, positive_weights: bool = False):
        super().__init__()
        self.positive_weights = bool(positive_weights)
        self.raw_weight = nn.Parameter(torch.empty(n_concepts))
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.xavier_uniform_(self.raw_weight.view(1, -1))

    @property
    def weight(self) -> torch.Tensor:
        if self.positive_weights:
            return F.softplus(self.raw_weight)
        return self.raw_weight

    def initialize_bias_from_prevalence(self, prevalence: float) -> None:
        prevalence = float(min(max(prevalence, 1e-4), 1.0 - 1e-4))
        with torch.no_grad():
            self.bias.copy_(torch.logit(torch.tensor(prevalence, device=self.bias.device)))

    def contributions(self, evidence: torch.Tensor) -> torch.Tensor:
        return evidence * self.weight.view(1, -1)

    def forward(self, evidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        contributions = self.contributions(evidence)
        return self.bias + contributions.sum(dim=-1), contributions


class MultiSetAdditiveTemporalConceptFANModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        latent_dim: int,
        n_concepts: int,
        n_memberships: int = 3,
        membership: str = "gaussian",
        oracle: bool = False,
        temporal_mode: str = "attention",
        dropout: float = 0.1,
        encoder_layers: int = 2,
        encoder_heads: int | None = None,
        encoder_ffn: int | None = None,
        temperature: float = 1.0,
        positive_decision_weights: bool = False,
        alpha_mode: str = "softmax_alpha",
        gamma_init: float = 0.5,
        gamma_max: float = 0.9,
    ):
        super().__init__()
        self.oracle = bool(oracle)
        self.temporal_mode = temporal_mode
        self.latent_dim = int(latent_dim)
        self.encoder = SequenceTemporalEncoder(
            input_dim,
            latent_dim,
            sequence_length,
            dropout,
            num_layers=encoder_layers,
            nhead=encoder_heads,
            dim_feedforward=encoder_ffn,
        )
        self.projector = TokenConceptProjector(latent_dim, n_concepts)
        self.temporal_aggregator = TemporalConceptAggregator(n_concepts, sequence_length)
        self.membership = MultiSetMembershipLayer(n_concepts, n_memberships, membership)
        self.aggregator = MultiSetFuzzyAggregator(
            n_concepts,
            n_memberships,
            temperature,
            alpha_mode=alpha_mode,
            gamma_init=gamma_init,
            gamma_max=gamma_max,
        )
        self.decision_head = AdditiveDecisionHead(n_concepts, positive_weights=positive_decision_weights)

    def freeze_concept_path(self) -> None:
        for module in [self.encoder, self.projector]:
            for param in module.parameters():
                param.requires_grad = False

    def freeze_membership(self, frozen: bool = True) -> None:
        for param in self.membership.parameters():
            param.requires_grad = not frozen

    def freeze_alpha(self, frozen: bool = True) -> None:
        for param in self.aggregator.parameters():
            param.requires_grad = not frozen

    def decision_from_evidence(self, evidence: torch.Tensor) -> torch.Tensor:
        logit, _ = self.decision_head(evidence)
        return logit

    def forward_from_summaries(
        self,
        h: torch.Tensor,
        trajectories: torch.Tensor,
        summaries: torch.Tensor,
        temporal_weights: torch.Tensor,
        concept_mask: torch.Tensor | None = None,
    ) -> MultiSetAdditiveFANOutput:
        memberships = self.membership(summaries)
        local_weights, fuzzy_values, concept_weights, evidence = self.aggregator(summaries, memberships, concept_mask=concept_mask)
        logit, signed = self.decision_head(evidence)
        return MultiSetAdditiveFANOutput(
            logit=logit,
            probability=torch.sigmoid(logit),
            latent=h.mean(dim=1),
            latent_sequence=h,
            concept_trajectories=trajectories,
            temporal_concept_weights=temporal_weights,
            concept_summaries=summaries,
            memberships=memberships,
            membership_local_weights=local_weights,
            fuzzy_values=fuzzy_values,
            concept_weights=concept_weights,
            concept_evidence=evidence,
            signed_decision_contributions=signed,
        )

    def forward(
        self,
        x: torch.Tensor,
        concept_sequence_override: torch.Tensor | None = None,
        concept_mask: torch.Tensor | None = None,
    ) -> MultiSetAdditiveFANOutput:
        if self.oracle:
            if concept_sequence_override is None:
                raise ValueError("Oracle temporal FAN requires concept_sequence_override")
            trajectories = concept_sequence_override.float()
            h = x.new_zeros((x.shape[0], trajectories.shape[1], self.latent_dim))
        else:
            h = self.encoder(x)
            trajectories = self.projector(h)
        summaries, temporal_weights = self.temporal_aggregator(trajectories, self.temporal_mode)
        return self.forward_from_summaries(h, trajectories, summaries, temporal_weights, concept_mask=concept_mask)
