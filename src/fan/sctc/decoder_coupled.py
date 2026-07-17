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
from fan.sctc.concept_aligned import LAYER_CONCEPT_INDICES, make_control_targets, sinkhorn
from fan.sctc.joint_causal import (
    WhiteningTransform,
    compact_capacity,
    compact_top_k,
    feature_decorrelation,
    transition_loss,
    transition_sparsity,
    usage_dominance,
)


@dataclass(frozen=True)
class DecoderCoupledTrainConfig:
    condition: str = "correct_concepts"
    capacity_multiplier: float = 2.0
    exact_tied_weights: bool = True
    epochs: int = 40
    min_epochs: int = 25
    patience: int = 6
    batch_size: int = 128
    learning_rate: float = 1e-4
    lambda_behavior: float = 0.10
    lambda_sparse: float = 1e-5
    lambda_transition: float = 0.03
    lambda_interventional: float = 0.03
    lambda_concept_decoder: float = 0.10
    lambda_decoder_alignment: float = 0.03
    lambda_assignment_entropy: float = 0.01
    lambda_incoherence: float = 0.01
    lambda_decorrelation: float = 0.001
    lambda_usage: float = 0.001
    assignment_temperature: float = 0.10
    intervention_features_per_batch: int = 2
    max_feature_frequency: float = 0.70


class TiedWhitenedTranscoder(nn.Module):
    def __init__(self, whitening: WhiteningTransform, n_features: int, top_k: int):
        super().__init__()
        self.n_features = int(n_features)
        self.target_top_k = int(top_k)
        self.active_top_k = int(top_k)
        self.register_buffer("white_mean", whitening.mean.float())
        self.register_buffer("white_components", whitening.components.float())
        self.register_buffer("white_eigenvalues", whitening.eigenvalues.float())
        self.white_eps = float(whitening.eps)
        self.encoder_weight = nn.Parameter(torch.empty(n_features, whitening.white_dim))
        self.encoder_bias = nn.Parameter(torch.full((n_features,), 0.01))
        self.reconstruction_bias = nn.Parameter(torch.zeros(whitening.white_dim))
        nn.init.xavier_uniform_(self.encoder_weight)
        self.normalize_decoder_()

    @classmethod
    def from_train_activation(cls, activation: torch.Tensor, n_features: int, top_k: int, variance: float = 0.995) -> "TiedWhitenedTranscoder":
        return cls(WhiteningTransform.fit(activation, variance=variance), n_features, top_k)

    @property
    def whitening(self) -> WhiteningTransform:
        return WhiteningTransform(self.white_mean, self.white_components, self.white_eigenvalues, self.white_eps)

    @property
    def decoder_weight(self) -> torch.Tensor:
        return self.encoder_weight.T

    def exact_tying_error(self) -> torch.Tensor:
        return (self.encoder_weight - self.decoder_weight.T).abs().max()

    def set_active_top_k(self, top_k: int) -> None:
        self.active_top_k = int(max(1, min(self.n_features, top_k)))

    def encode_white(self, white: torch.Tensor) -> torch.Tensor:
        z = torch.relu(F.linear(white, self.encoder_weight, self.encoder_bias))
        if 0 < self.active_top_k < z.shape[-1]:
            values, indices = torch.topk(z, k=self.active_top_k, dim=-1)
            sparse = torch.zeros_like(z)
            z = sparse.scatter(-1, indices, values)
        return z

    def encode(self, activation: torch.Tensor) -> torch.Tensor:
        return self.encode_white(self.whitening.whiten(activation))

    def decode_white(self, z: torch.Tensor) -> torch.Tensor:
        return F.linear(z, self.decoder_weight, self.reconstruction_bias)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.whitening.unwhiten(self.decode_white(z))

    def forward(self, activation: torch.Tensor) -> dict[str, torch.Tensor]:
        white = self.whitening.whiten(activation)
        z = self.encode_white(white)
        reconstructed_white = self.decode_white(z)
        return {
            "z": z,
            "white": white,
            "reconstructed_white": reconstructed_white,
            "reconstructed": self.whitening.unwhiten(reconstructed_white),
        }

    def original_decoder_directions(self) -> torch.Tensor:
        white_dirs = self.encoder_weight
        return (white_dirs * torch.sqrt(self.white_eigenvalues + self.white_eps).view(1, -1)) @ self.white_components.T

    def normalize_decoder_(self) -> None:
        with torch.no_grad():
            norms = self.encoder_weight.norm(dim=1).clamp_min(1e-8)
            self.encoder_weight.div_(norms.view(-1, 1))
            self.encoder_bias.mul_(norms)


class FrozenConceptProbe(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", weight.float())
        self.register_buffer("bias", bias.float())
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, white_activation: torch.Tensor) -> torch.Tensor:
        return F.linear(white_activation, self.weight, self.bias)

    def sha256(self) -> str:
        import hashlib

        payload = torch.cat([self.weight.detach().cpu().reshape(-1), self.bias.detach().cpu().reshape(-1)]).numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()


def fit_frozen_probe(white_activation: torch.Tensor, targets: torch.Tensor, ridge: float = 1e-4) -> FrozenConceptProbe:
    x = white_activation.reshape(-1, white_activation.shape[-1]).float()
    y = targets.reshape(-1, targets.shape[-1]).float()
    ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
    xb = torch.cat([x, ones], dim=1)
    eye = torch.eye(xb.shape[1], device=x.device, dtype=x.dtype)
    eye[-1, -1] = 0.0
    beta = torch.linalg.solve(xb.T @ xb + ridge * eye, xb.T @ y)
    weight = beta[:-1].T
    bias = beta[-1]
    return FrozenConceptProbe(weight.detach(), bias.detach())


class DecoderCoupledConceptSCTC(nn.Module):
    def __init__(self, transcoders: list[TiedWhitenedTranscoder], probes: list[FrozenConceptProbe]):
        super().__init__()
        self.transcoders = nn.ModuleList(transcoders)
        self.probes = nn.ModuleList(probes)
        self.transitions = nn.ParameterList(
            [nn.Parameter(torch.zeros(transcoders[i + 1].n_features, transcoders[i].n_features)) for i in range(len(transcoders) - 1)]
        )
        for param in self.transitions:
            nn.init.normal_(param, mean=0.0, std=0.01)

    def set_active_topks(self, topks: list[int]) -> None:
        for transcoder, top_k in zip(self.transcoders, topks):
            transcoder.set_active_top_k(top_k)

    def forward(self, activations: list[torch.Tensor]) -> dict[str, list[torch.Tensor]]:
        outs = [transcoder(act) for transcoder, act in zip(self.transcoders, activations)]
        concept_predictions = [probe(out["reconstructed_white"]) for probe, out in zip(self.probes, outs)]
        return {
            "outputs": outs,
            "z": [out["z"] for out in outs],
            "white": [out["white"] for out in outs],
            "reconstructed_white": [out["reconstructed_white"] for out in outs],
            "reconstructed": [out["reconstructed"] for out in outs],
            "concept_predictions": concept_predictions,
        }

    def exact_tying_error(self) -> torch.Tensor:
        return torch.stack([t.exact_tying_error() for t in self.transcoders]).max()

    def probe_hashes(self) -> list[str]:
        return [probe.sha256() for probe in self.probes]


def build_decoder_coupled_model(
    train_activations: list[torch.Tensor],
    train_concepts: torch.Tensor,
    cfg: DecoderCoupledTrainConfig,
    device: torch.device,
) -> tuple[DecoderCoupledConceptSCTC, list[int], list[int], dict]:
    transcoders: list[TiedWhitenedTranscoder] = []
    probes: list[FrozenConceptProbe] = []
    capacities: list[int] = []
    topks: list[int] = []
    probe_manifest = {"layers": []}
    for layer, activation in enumerate(train_activations):
        cap = compact_capacity(activation, cfg.capacity_multiplier)
        top_k = compact_top_k(cap)
        transcoder = TiedWhitenedTranscoder.from_train_activation(activation, cap, top_k).to(device)
        white = transcoder.whitening.whiten(activation.to(device))
        targets = train_concepts.to(device)[..., list(LAYER_CONCEPT_INDICES[layer])]
        probe = fit_frozen_probe(white, targets).to(device)
        for buffer in probe.buffers():
            buffer.requires_grad = False
        transcoders.append(transcoder)
        probes.append(probe)
        capacities.append(cap)
        topks.append(top_k)
        probe_manifest["layers"].append(
            {
                "layer": layer,
                "concept_indices": list(LAYER_CONCEPT_INDICES[layer]),
                "n_features": cap,
                "top_k": top_k,
                "white_dim": transcoder.whitening.white_dim,
                "probe_sha256": probe.sha256(),
            }
        )
    return DecoderCoupledConceptSCTC(transcoders, probes).to(device), capacities, topks, probe_manifest


def decoder_incoherence(model: TiedWhitenedTranscoder) -> torch.Tensor:
    directions = F.normalize(model.original_decoder_directions(), dim=-1)
    gram = directions @ directions.T
    off = gram - torch.diag_embed(torch.diagonal(gram))
    return off.square().mean()


def decoder_alignment_losses(
    model: DecoderCoupledConceptSCTC,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    alignment = model.transitions[0].new_zeros(()) if len(model.transitions) else model.transcoders[0].encoder_weight.new_zeros(())
    entropy = alignment.clone()
    assignments: list[torch.Tensor] = []
    for transcoder, probe in zip(model.transcoders, model.probes):
        decoder_dirs = F.normalize(transcoder.encoder_weight, dim=-1)
        probe_dirs = F.normalize(probe.weight, dim=-1)
        similarity = torch.abs(decoder_dirs @ probe_dirs.T)
        logits = similarity / max(float(temperature), 1e-6)
        if similarity.shape[1] == 1:
            assignment = torch.softmax(logits, dim=0)
        else:
            assignment = sinkhorn(logits)
        assignments.append(assignment.detach())
        alignment = alignment + (1.0 - (assignment * similarity).sum() / assignment.sum().clamp_min(1e-8))
        entropy = entropy + (-(assignment * torch.log(assignment.clamp_min(1e-8))).sum() / assignment.numel())
    return alignment / len(model.transcoders), entropy / len(model.transcoders), assignments


def decoder_concept_loss(
    model: DecoderCoupledConceptSCTC,
    out: dict[str, list[torch.Tensor]],
    concepts: torch.Tensor,
) -> torch.Tensor:
    loss = out["z"][0].new_zeros(())
    for layer, pred in enumerate(out["concept_predictions"]):
        target = concepts[..., list(LAYER_CONCEPT_INDICES[layer])]
        loss = loss + F.huber_loss(pred, target)
    return loss / len(out["concept_predictions"])


def real_interventional_consistency(
    model: DecoderCoupledConceptSCTC,
    zs: list[torch.Tensor],
    downstream_forward: Callable[[int, torch.Tensor], torch.Tensor],
    features_per_batch: int = 2,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for layer, matrix in enumerate(model.transitions):
        source_z = zs[layer]
        target_z = zs[layer + 1].detach()
        activity = source_z.mean(dim=(0, 1))
        active = torch.topk(activity, k=min(int(features_per_batch), activity.numel())).indices
        for feature_id in active:
            fid = int(feature_id.detach().cpu())
            z_ablated = source_z.clone()
            z_ablated[..., fid] = 0.0
            after_activation = downstream_forward(layer, model.transcoders[layer].decode(z_ablated))
            after_z = model.transcoders[layer + 1](after_activation)["z"]
            observed = after_z - target_z
            predicted = -source_z[..., fid].unsqueeze(-1) * matrix[:, fid].view(1, 1, -1)
            losses.append(F.huber_loss(predicted, observed))
    if not losses:
        return zs[0].new_zeros(())
    return sum(losses) / len(losses)


def make_random_matched_targets(targets: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=targets.device)
    generator.manual_seed(int(seed))
    flat = targets.reshape(-1, targets.shape[-1]).float()
    mean = flat.mean(dim=0)
    cov = torch.cov(flat.T) + 1e-5 * torch.eye(flat.shape[-1], device=targets.device)
    chol = torch.linalg.cholesky(cov)
    noise = torch.randn(flat.shape, generator=generator, device=targets.device, dtype=targets.dtype) @ chol.T + mean
    return noise.reshape_as(targets)


def make_decoder_coupled_targets(targets: torch.Tensor, condition: str, seed: int) -> tuple[torch.Tensor, dict]:
    if condition == "CORRECT_CONCEPTS":
        return targets, {"condition": condition}
    if condition == "PERMUTED_CONCEPTS":
        flat = targets.reshape(targets.shape[0], -1)
        generator = torch.Generator(device=targets.device)
        generator.manual_seed(int(seed))
        perm = torch.randperm(flat.shape[0], generator=generator, device=targets.device)
        permuted = flat[perm].reshape_as(targets)
        import hashlib

        return permuted, {"condition": condition, "permutation_seed": int(seed), "permutation_sha256": hashlib.sha256(perm.detach().cpu().numpy().tobytes()).hexdigest()}
    if condition == "RANDOM_MATCHED_CONCEPTS":
        matched = make_random_matched_targets(targets, seed)
        return matched, {"condition": condition, "random_seed": int(seed)}
    raise ValueError(f"unknown C2 condition {condition}")


def train_decoder_coupled_sctc(
    train_activations: list[torch.Tensor],
    train_concepts: torch.Tensor,
    behavior_forward: Callable[[int, torch.Tensor], torch.Tensor],
    downstream_forward: Callable[[int, torch.Tensor], torch.Tensor],
    train_logit: torch.Tensor,
    cfg: DecoderCoupledTrainConfig,
    device: torch.device,
    control_seed: int,
) -> tuple[DecoderCoupledConceptSCTC, pd.DataFrame, pd.DataFrame, dict]:
    model, capacities, topks, probe_manifest = build_decoder_coupled_model(train_activations, train_concepts, cfg, device)
    initial_probe_hashes = model.probe_hashes()
    schedules = [TopKAnnealingSchedule(target_top_k=top_k) for top_k in topks]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    targets, target_manifest = make_decoder_coupled_targets(train_concepts.float().to(device), cfg.condition, control_seed)
    dataset = TensorDataset(*[x.float() for x in train_activations], train_logit.float(), targets.detach().cpu().float())
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    rows = []
    best_state = None
    best_selection = float("inf")
    stale = 0
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
            inter = real_interventional_consistency(model, out["z"], downstream_forward, cfg.intervention_features_per_batch)
            concept = decoder_concept_loss(model, out, concepts)
            align, entropy, _ = decoder_alignment_losses(model, cfg.assignment_temperature)
            incoh = sum(decoder_incoherence(t) for t in model.transcoders) / len(model.transcoders)
            decor = sum(feature_decorrelation(z) for z in out["z"]) / len(out["z"])
            usage = sum(usage_dominance(z, cfg.max_feature_frequency) for z in out["z"]) / len(out["z"])
            loss = (
                rec
                + cfg.lambda_behavior * behavior
                + cfg.lambda_sparse * sparse
                + cfg.lambda_transition * trans
                + cfg.lambda_interventional * inter
                + cfg.lambda_concept_decoder * concept
                + cfg.lambda_decoder_alignment * align
                + cfg.lambda_assignment_entropy * entropy
                + cfg.lambda_incoherence * incoh
                + cfg.lambda_decorrelation * decor
                + cfg.lambda_usage * usage
            )
            loss.backward()
            opt.step()
            for transcoder in model.transcoders:
                transcoder.normalize_decoder_()
            selection = rec + cfg.lambda_behavior * behavior + cfg.lambda_concept_decoder * concept + cfg.lambda_decoder_alignment * align
            epoch_rows.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "selection_loss": float(selection.detach().cpu()),
                    "reconstruction_loss": float(rec.detach().cpu()),
                    "behavior_loss": float(behavior.detach().cpu()),
                    "sparsity_loss": float(sparse.detach().cpu()),
                    "transition_loss": float(trans.detach().cpu()),
                    "edge_sparse_loss": float(edge.detach().cpu()),
                    "interventional_loss": float(inter.detach().cpu()),
                    "decoder_concept_loss": float(concept.detach().cpu()),
                    "decoder_alignment_loss": float(align.detach().cpu()),
                    "assignment_entropy_loss": float(entropy.detach().cpu()),
                    "incoherence_loss": float(incoh.detach().cpu()),
                    "decorrelation_loss": float(decor.detach().cpu()),
                    "usage_loss": float(usage.detach().cpu()),
                    "exact_tying_error": float(model.exact_tying_error().detach().cpu()),
                }
            )
        frame = pd.DataFrame(epoch_rows)
        row = {"epoch": epoch, "condition": cfg.condition, "active_topks": "|".join(str(t.active_top_k) for t in model.transcoders)}
        for col in frame.columns:
            row[col] = float(frame[col].mean())
        rows.append(row)
        current = row["selection_loss"]
        if current < best_selection:
            best_selection = current
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= cfg.min_epochs and stale >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.set_active_topks(topks)
    final_probe_hashes = model.probe_hashes()
    assignment_rows = []
    _, _, assignments = decoder_alignment_losses(model, cfg.assignment_temperature)
    for layer, assignment in enumerate(assignments):
        for feature_id in range(assignment.shape[0]):
            for concept_local in range(assignment.shape[1]):
                assignment_rows.append(
                    {
                        "layer": layer,
                        "feature_id": feature_id,
                        "concept_local": concept_local,
                        "assignment": float(assignment[feature_id, concept_local].cpu()),
                    }
                )
    manifest = {
        **probe_manifest,
        "initial_probe_hashes": initial_probe_hashes,
        "final_probe_hashes": final_probe_hashes,
        "probe_hashes_unchanged": initial_probe_hashes == final_probe_hashes,
        "target_manifest": target_manifest,
        "selection_uses_recovery_metrics": False,
    }
    return model, pd.DataFrame(rows), pd.DataFrame(assignment_rows), manifest
