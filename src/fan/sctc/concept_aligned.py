from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fan.sctc.adaptive import TopKAnnealingSchedule
from fan.sctc.joint_causal import (
    CompactCausalTranscoder,
    JointCausalSCTC,
    compact_capacity,
    compact_top_k,
    decoder_incoherence,
    feature_decorrelation,
    tied_penalty,
    transition_loss,
    transition_sparsity,
    usage_dominance,
)


LAYER_CONCEPT_INDICES: dict[int, tuple[int, ...]] = {
    0: (0,),
    1: (1,),
    2: (2,),
    3: (3, 4),
}


def sinkhorn(logits: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """Doubly-normalize feature-to-concept scores for soft matching."""
    if logits.numel() == 0:
        return torch.zeros_like(logits)
    mat = torch.exp(logits - logits.max()).clamp_min(1e-12)
    for _ in range(n_iters):
        mat = mat / mat.sum(dim=1, keepdim=True).clamp_min(1e-12)
        mat = mat / mat.sum(dim=0, keepdim=True).clamp_min(1e-12)
    return mat


def correlation_matrix(features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return feature x target Pearson correlations for flattened sequences."""
    feat = features.reshape(-1, features.shape[-1]).float()
    tgt = targets.reshape(-1, targets.shape[-1]).float()
    feat = feat - feat.mean(dim=0, keepdim=True)
    tgt = tgt - tgt.mean(dim=0, keepdim=True)
    feat_std = feat.square().sum(dim=0, keepdim=True).sqrt().clamp_min(1e-8)
    tgt_std = tgt.square().sum(dim=0, keepdim=True).sqrt().clamp_min(1e-8)
    return (feat / feat_std).T @ (tgt / tgt_std)


@dataclass(frozen=True)
class ConceptAlignedInterventionalTrainConfig:
    stage: str = "correct_concepts"
    capacity_multiplier: float = 2.0
    lambda_behavior: float = 0.10
    lambda_sparse: float = 1e-5
    lambda_transition: float = 0.03
    lambda_edge_sparse: float = 0.001
    lambda_interventional: float = 0.03
    lambda_concept: float = 0.10
    lambda_matching: float = 0.03
    lambda_incoherence: float = 0.01
    lambda_decorrelation: float = 0.001
    lambda_tied: float = 0.001
    lambda_usage: float = 0.001
    max_feature_frequency: float = 0.70
    matching_temperature: float = 0.10
    intervention_features_per_batch: int = 2
    epochs: int = 25
    batch_size: int = 128
    learning_rate: float = 1e-3


class ConceptAlignedInterventionalSCTC(JointCausalSCTC):
    def __init__(self, transcoders: list[CompactCausalTranscoder], layer_concept_indices: dict[int, tuple[int, ...]] | None = None):
        super().__init__(transcoders)
        self.layer_concept_indices = layer_concept_indices or LAYER_CONCEPT_INDICES
        self.concept_readouts = nn.ModuleList(
            [nn.Linear(transcoder.n_features, len(self.layer_concept_indices[layer])) for layer, transcoder in enumerate(transcoders)]
        )

    def concept_predictions(self, zs: list[torch.Tensor]) -> list[torch.Tensor]:
        return [readout(z) for readout, z in zip(self.concept_readouts, zs)]


def build_concept_aligned_model(
    train_activations: list[torch.Tensor],
    cfg: ConceptAlignedInterventionalTrainConfig,
) -> tuple[ConceptAlignedInterventionalSCTC, list[int], list[int]]:
    transcoders: list[CompactCausalTranscoder] = []
    capacities: list[int] = []
    topks: list[int] = []
    for activation in train_activations:
        cap = compact_capacity(activation, cfg.capacity_multiplier)
        top_k = compact_top_k(cap)
        capacities.append(cap)
        topks.append(top_k)
        transcoders.append(CompactCausalTranscoder.from_train_activation(activation, cap, top_k))
    return ConceptAlignedInterventionalSCTC(transcoders), capacities, topks


def concept_losses(
    model: ConceptAlignedInterventionalSCTC,
    zs: list[torch.Tensor],
    concept_targets: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    concept_total = zs[0].new_zeros(())
    matching_total = zs[0].new_zeros(())
    rows: dict[str, float] = {}
    preds = model.concept_predictions(zs)
    for layer, pred in enumerate(preds):
        indices = model.layer_concept_indices[layer]
        target = concept_targets[..., list(indices)]
        concept = F.huber_loss(pred, target)
        corr = correlation_matrix(zs[layer], target).abs()
        assignment = sinkhorn(corr / max(float(temperature), 1e-6))
        matching = -((assignment * corr).sum() / assignment.sum().clamp_min(1e-8))
        concept_total = concept_total + concept
        matching_total = matching_total + matching
        rows[f"layer{layer}_concept_loss"] = float(concept.detach().cpu())
        rows[f"layer{layer}_matching_score"] = float((-matching).detach().cpu())
    return concept_total / len(zs), matching_total / len(zs), rows


def real_interventional_consistency(
    model: ConceptAlignedInterventionalSCTC,
    activations: list[torch.Tensor],
    zs: list[torch.Tensor],
    downstream_forward: Callable[[int, torch.Tensor], torch.Tensor],
    features_per_batch: int = 2,
) -> torch.Tensor:
    """Self-supervise sparse transitions using decoded feature ablations and true downstream forward."""
    losses: list[torch.Tensor] = []
    for layer, matrix in enumerate(model.transitions):
        source_z = zs[layer]
        target_z = zs[layer + 1].detach()
        activity = source_z.mean(dim=(0, 1))
        active = torch.topk(activity, k=min(int(features_per_batch), activity.numel())).indices
        for feature_id in active:
            feature_id_int = int(feature_id.detach().cpu().item())
            z_ablated = source_z.clone()
            z_ablated[..., feature_id_int] = 0.0
            h_ablated = model.transcoders[layer].decode(z_ablated)
            after_activation = downstream_forward(layer, h_ablated)
            after_z = model.transcoders[layer + 1](after_activation)["z"]
            observed_delta = after_z - target_z
            predicted_delta = -source_z[..., feature_id_int].unsqueeze(-1) * matrix[:, feature_id_int].view(1, 1, -1)
            losses.append(F.huber_loss(predicted_delta, observed_delta))

            feature_std = source_z[..., feature_id_int].std().clamp_min(1e-6)
            z_pushed = source_z.clone()
            z_pushed[..., feature_id_int] = z_pushed[..., feature_id_int] + feature_std
            h_pushed = model.transcoders[layer].decode(z_pushed)
            pushed_activation = downstream_forward(layer, h_pushed)
            pushed_z = model.transcoders[layer + 1](pushed_activation)["z"]
            observed_push = pushed_z - target_z
            predicted_push = feature_std * matrix[:, feature_id_int].view(1, 1, -1)
            losses.append(F.huber_loss(predicted_push.expand_as(observed_push), observed_push))
    if not losses:
        return zs[0].new_zeros(())
    return sum(losses) / len(losses)


def make_control_targets(targets: torch.Tensor, mode: str, seed: int) -> torch.Tensor:
    if mode.startswith("correct_concepts"):
        return targets
    generator = torch.Generator(device=targets.device)
    generator.manual_seed(int(seed))
    if mode == "permuted_concepts":
        flat = targets.reshape(-1, targets.shape[-1])
        perm = torch.randperm(flat.shape[0], generator=generator, device=targets.device)
        return flat[perm].reshape_as(targets)
    if mode == "random_targets":
        mean = targets.mean(dim=(0, 1), keepdim=True)
        std = targets.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
        return torch.randn(targets.shape, generator=generator, device=targets.device, dtype=targets.dtype) * std + mean
    raise ValueError(f"unknown concept-control mode: {mode}")


def train_concept_aligned_interventional_sctc(
    train_activations: list[torch.Tensor],
    train_concepts: torch.Tensor,
    behavior_forward: Callable[[int, torch.Tensor], torch.Tensor],
    downstream_activation_forward: Callable[[int, torch.Tensor], torch.Tensor],
    train_logit: torch.Tensor,
    cfg: ConceptAlignedInterventionalTrainConfig,
    device: torch.device,
    control_seed: int,
) -> tuple[ConceptAlignedInterventionalSCTC, pd.DataFrame]:
    model, capacities, topks = build_concept_aligned_model(train_activations, cfg)
    model = model.to(device)
    schedules = [TopKAnnealingSchedule(target_top_k=top_k) for top_k in topks]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    targets = make_control_targets(train_concepts.float().to(device), cfg.stage, control_seed)
    dataset = TensorDataset(*[x.float() for x in train_activations], train_logit.float(), targets.detach().cpu().float())
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    rows: list[dict[str, float | int | str]] = []
    for epoch in range(1, cfg.epochs + 1):
        model.set_active_topks([sched.top_k_for_epoch(epoch, cap) for sched, cap in zip(schedules, capacities)])
        epoch_rows = []
        for batch in loader:
            activations = [x.to(device) for x in batch[:-2]]
            logit = batch[-2].to(device)
            concepts = batch[-1].to(device)
            opt.zero_grad(set_to_none=True)
            out = model(activations)
            rec = sum(F.mse_loss(r, a) for r, a in zip(out["reconstructed"], activations)) / len(activations)
            sparse = sum(z.mean() for z in out["z"]) / len(out["z"])
            behavior = F.l1_loss(behavior_forward(3, out["reconstructed"][3]), logit)
            trans = transition_loss(out["z"], model.transitions)
            edge = transition_sparsity(model.transitions)
            incoh = sum(decoder_incoherence(t) for t in model.transcoders) / len(model.transcoders)
            decor = sum(feature_decorrelation(z) for z in out["z"]) / len(out["z"])
            tied = sum(tied_penalty(t) for t in model.transcoders) / len(model.transcoders)
            usage = sum(usage_dominance(z, cfg.max_feature_frequency) for z in out["z"]) / len(out["z"])
            concept, matching, concept_row = concept_losses(model, out["z"], concepts, cfg.matching_temperature)
            inter = real_interventional_consistency(
                model,
                activations,
                out["z"],
                downstream_activation_forward,
                cfg.intervention_features_per_batch,
            )
            loss = (
                rec
                + cfg.lambda_behavior * behavior
                + cfg.lambda_sparse * sparse
                + cfg.lambda_transition * trans
                + cfg.lambda_edge_sparse * edge
                + cfg.lambda_interventional * inter
                + cfg.lambda_concept * concept
                + cfg.lambda_matching * matching
                + cfg.lambda_incoherence * incoh
                + cfg.lambda_decorrelation * decor
                + cfg.lambda_tied * tied
                + cfg.lambda_usage * usage
            )
            loss.backward()
            opt.step()
            for transcoder in model.transcoders:
                transcoder.normalize_decoder_()
            epoch_rows.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "reconstruction_loss": float(rec.detach().cpu()),
                    "behavior_loss": float(behavior.detach().cpu()),
                    "sparsity_loss": float(sparse.detach().cpu()),
                    "transition_loss": float(trans.detach().cpu()),
                    "edge_sparse_loss": float(edge.detach().cpu()),
                    "interventional_loss": float(inter.detach().cpu()),
                    "concept_loss": float(concept.detach().cpu()),
                    "matching_loss": float(matching.detach().cpu()),
                    "incoherence_loss": float(incoh.detach().cpu()),
                    "decorrelation_loss": float(decor.detach().cpu()),
                    "tied_loss": float(tied.detach().cpu()),
                    "usage_loss": float(usage.detach().cpu()),
                    **concept_row,
                }
            )
        frame = pd.DataFrame(epoch_rows)
        row = {"epoch": epoch, "stage": cfg.stage, "active_topks": "|".join(str(t.active_top_k) for t in model.transcoders)}
        for col in frame.columns:
            row[col] = float(frame[col].mean())
        rows.append(row)
    model.set_active_topks(topks)
    return model, pd.DataFrame(rows)
