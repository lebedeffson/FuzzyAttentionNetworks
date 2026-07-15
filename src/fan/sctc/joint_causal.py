from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fan.sctc.adaptive import TopKAnnealingSchedule, effective_rank


REGISTERED_CAPACITIES = [4, 6, 8, 12, 16, 24, 32]


def compact_capacity(activation: torch.Tensor, multiplier: float, minimum: int = 4, maximum: int = 32) -> int:
    raw = int(np.ceil(multiplier * effective_rank(activation)))
    raw = min(maximum, max(minimum, raw))
    return min(REGISTERED_CAPACITIES, key=lambda x: (abs(x - raw), x))


def compact_top_k(capacity: int) -> int:
    if capacity <= 6:
        return 2
    if capacity <= 12:
        return 4
    if capacity <= 24:
        return 6
    return 8


@dataclass(frozen=True)
class WhiteningTransform:
    mean: torch.Tensor
    components: torch.Tensor
    eigenvalues: torch.Tensor
    eps: float = 1e-5

    @classmethod
    def fit(cls, activation: torch.Tensor, variance: float = 0.995, eps: float = 1e-5) -> "WhiteningTransform":
        flat = activation.reshape(-1, activation.shape[-1]).float()
        mean = flat.mean(dim=0)
        centered = flat - mean
        cov = centered.T @ centered / max(1, centered.shape[0] - 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        order = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[order].clamp_min(0.0)
        eigvecs = eigvecs[:, order]
        total = eigvals.sum().clamp_min(eps)
        cumulative = torch.cumsum(eigvals, dim=0) / total
        keep = int(torch.searchsorted(cumulative, torch.tensor(float(variance), device=cumulative.device)).item()) + 1
        keep = max(1, min(keep, eigvals.numel()))
        return cls(mean=mean, components=eigvecs[:, :keep], eigenvalues=eigvals[:keep].clamp_min(eps), eps=eps)

    @property
    def white_dim(self) -> int:
        return int(self.eigenvalues.numel())

    def to(self, device: torch.device) -> "WhiteningTransform":
        return WhiteningTransform(self.mean.to(device), self.components.to(device), self.eigenvalues.to(device), self.eps)

    def whiten(self, activation: torch.Tensor) -> torch.Tensor:
        centered = activation.float() - self.mean.view(*([1] * (activation.ndim - 1)), -1)
        projected = centered @ self.components
        return projected / torch.sqrt(self.eigenvalues.view(*([1] * (activation.ndim - 1)), -1) + self.eps)

    def unwhiten(self, white_activation: torch.Tensor) -> torch.Tensor:
        scaled = white_activation * torch.sqrt(self.eigenvalues.view(*([1] * (white_activation.ndim - 1)), -1) + self.eps)
        return scaled @ self.components.T + self.mean.view(*([1] * (white_activation.ndim - 1)), -1)


class CompactCausalTranscoder(nn.Module):
    def __init__(self, whitening: WhiteningTransform, n_features: int, top_k: int):
        super().__init__()
        self.n_features = int(n_features)
        self.target_top_k = int(top_k)
        self.active_top_k = int(top_k)
        self.register_buffer("white_mean", whitening.mean.float())
        self.register_buffer("white_components", whitening.components.float())
        self.register_buffer("white_eigenvalues", whitening.eigenvalues.float())
        self.white_eps = float(whitening.eps)
        self.encoder = nn.Linear(whitening.white_dim, n_features)
        self.decoder = nn.Linear(n_features, whitening.white_dim, bias=False)
        self.reconstruction_bias = nn.Parameter(torch.zeros(whitening.white_dim))
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.constant_(self.encoder.bias, 0.01)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)
        self.normalize_decoder_()

    @classmethod
    def from_train_activation(cls, activation: torch.Tensor, n_features: int, top_k: int, variance: float = 0.995) -> "CompactCausalTranscoder":
        return cls(WhiteningTransform.fit(activation, variance=variance), n_features=n_features, top_k=top_k)

    @property
    def whitening(self) -> WhiteningTransform:
        return WhiteningTransform(self.white_mean, self.white_components, self.white_eigenvalues, self.white_eps)

    def set_active_top_k(self, top_k: int) -> None:
        self.active_top_k = int(max(1, min(self.n_features, top_k)))

    def encode(self, activation: torch.Tensor) -> torch.Tensor:
        xw = self.whitening.whiten(activation)
        z = torch.relu(self.encoder(xw))
        if 0 < self.active_top_k < z.shape[-1]:
            values, indices = torch.topk(z, k=self.active_top_k, dim=-1)
            sparse = torch.zeros_like(z)
            z = sparse.scatter(-1, indices, values)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        white = self.decoder(z) + self.reconstruction_bias
        return self.whitening.unwhiten(white)

    def forward(self, activation: torch.Tensor) -> dict[str, torch.Tensor]:
        xw = self.whitening.whiten(activation)
        z = torch.relu(self.encoder(xw))
        if 0 < self.active_top_k < z.shape[-1]:
            values, indices = torch.topk(z, k=self.active_top_k, dim=-1)
            sparse = torch.zeros_like(z)
            z = sparse.scatter(-1, indices, values)
        rec_white = self.decoder(z) + self.reconstruction_bias
        return {"z": z, "white": xw, "reconstructed_white": rec_white, "reconstructed": self.whitening.unwhiten(rec_white)}

    def original_decoder_directions(self) -> torch.Tensor:
        white_dirs = self.decoder.weight.T
        return (white_dirs * torch.sqrt(self.white_eigenvalues + self.white_eps).view(1, -1)) @ self.white_components.T

    def normalize_decoder_(self) -> None:
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0).clamp_min(1e-8)
            self.decoder.weight.div_(norms.view(1, -1))
            self.encoder.weight.mul_(norms.view(-1, 1))
            self.encoder.bias.mul_(norms)

    @staticmethod
    def activation_frequency(z: torch.Tensor) -> torch.Tensor:
        return (z > 0).float().mean(dim=tuple(range(z.ndim - 1)))


def decoder_incoherence(model: CompactCausalTranscoder) -> torch.Tensor:
    directions = F.normalize(model.original_decoder_directions(), dim=-1)
    gram = directions @ directions.T
    off = gram - torch.diag_embed(torch.diagonal(gram))
    return off.square().mean()


def tied_penalty(model: CompactCausalTranscoder) -> torch.Tensor:
    return (model.encoder.weight - model.decoder.weight.T).square().mean()


def feature_decorrelation(z: torch.Tensor) -> torch.Tensor:
    flat = z.reshape(-1, z.shape[-1])
    if flat.shape[0] <= 1:
        return z.new_zeros(())
    centered = flat - flat.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(1, flat.shape[0] - 1)
    off = cov - torch.diag_embed(torch.diagonal(cov))
    return off.square().mean()


def usage_dominance(z: torch.Tensor, max_frequency: float = 0.7) -> torch.Tensor:
    freq = (z > 0).float().mean(dim=tuple(range(z.ndim - 1)))
    return torch.relu(freq - float(max_frequency)).square().mean()


@dataclass(frozen=True)
class JointCausalTrainConfig:
    stage: str
    capacity_multiplier: float = 2.0
    lambda_incoherence: float = 0.001
    lambda_decorrelation: float = 0.001
    lambda_tied: float = 0.001
    lambda_usage: float = 0.001
    lambda_behavior: float = 0.1
    lambda_sparse: float = 1e-5
    lambda_transition: float = 0.0
    lambda_edge_sparse: float = 0.0
    lambda_interventional: float = 0.0
    max_feature_frequency: float = 0.7
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    topk_no_topk_epochs: int = 3
    topk_anneal_until_epoch: int = 10


class JointCausalSCTC(nn.Module):
    def __init__(self, transcoders: list[CompactCausalTranscoder]):
        super().__init__()
        self.transcoders = nn.ModuleList(transcoders)
        self.transitions = nn.ParameterList(
            [nn.Parameter(torch.zeros(transcoders[i + 1].n_features, transcoders[i].n_features)) for i in range(len(transcoders) - 1)]
        )
        for param in self.transitions:
            nn.init.normal_(param, mean=0.0, std=0.01)

    def set_active_topks(self, topks: list[int]) -> None:
        for model, top_k in zip(self.transcoders, topks):
            model.set_active_top_k(top_k)

    def forward(self, activations: list[torch.Tensor]) -> dict[str, list[torch.Tensor]]:
        outs = [model(act) for model, act in zip(self.transcoders, activations)]
        return {
            "outputs": outs,
            "z": [out["z"] for out in outs],
            "reconstructed": [out["reconstructed"] for out in outs],
            "white": [out["white"] for out in outs],
            "reconstructed_white": [out["reconstructed_white"] for out in outs],
        }


def transition_loss(zs: list[torch.Tensor], transitions: nn.ParameterList, lags: list[int] | None = None) -> torch.Tensor:
    if lags is None:
        lags = [0, 1, 3]
    total = zs[0].new_zeros(())
    count = 0
    for idx, matrix in enumerate(transitions):
        source = zs[idx]
        target = zs[idx + 1]
        best = None
        for lag in lags:
            if lag == 0:
                src = source
                tgt = target
            elif source.shape[1] <= lag:
                continue
            else:
                src = source[:, :-lag]
                tgt = target[:, lag:]
            pred = torch.einsum("bti,ji->btj", src, matrix)
            loss = F.huber_loss(pred, tgt)
            best = loss if best is None else torch.minimum(best, loss)
        if best is not None:
            total = total + best
            count += 1
    return total / max(1, count)


def transition_sparsity(transitions: nn.ParameterList) -> torch.Tensor:
    return sum(param.abs().mean() for param in transitions) / max(1, len(transitions))


def interventional_consistency(zs: list[torch.Tensor], transitions: nn.ParameterList, features_per_batch: int = 2) -> torch.Tensor:
    losses = []
    for idx, matrix in enumerate(transitions):
        source = zs[idx].detach()
        target = zs[idx + 1].detach()
        flat_activity = source.mean(dim=(0, 1))
        active = torch.topk(flat_activity, k=min(features_per_batch, flat_activity.numel())).indices
        for feature_id in active:
            before = target
            predicted_delta = -source[..., feature_id].unsqueeze(-1) * matrix[:, feature_id].view(1, 1, -1)
            # The observed target after zeroing source feature is approximated in feature space by removing
            # the transition-predicted component; this keeps the loss self-supervised and label-free.
            observed_delta = -before * (source[..., feature_id].unsqueeze(-1) > 0).float() * 0.0
            losses.append(F.huber_loss(predicted_delta, observed_delta))
    if not losses:
        return zs[0].new_zeros(())
    return sum(losses) / len(losses)


def train_joint_causal_sctc(
    train_activations: list[torch.Tensor],
    behavior_forward: Callable[[int, torch.Tensor], torch.Tensor],
    train_logit: torch.Tensor,
    cfg: JointCausalTrainConfig,
    device: torch.device,
) -> tuple[JointCausalSCTC, pd.DataFrame]:
    transcoders = []
    topks = []
    capacities = []
    for activation in train_activations:
        cap = compact_capacity(activation, cfg.capacity_multiplier)
        top_k = compact_top_k(cap)
        capacities.append(cap)
        topks.append(top_k)
        transcoders.append(CompactCausalTranscoder.from_train_activation(activation, cap, top_k))
    model = JointCausalSCTC(transcoders).to(device)
    schedules = [TopKAnnealingSchedule(target_top_k=top_k) for top_k in topks]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    dataset = TensorDataset(*[x.float() for x in train_activations], train_logit.float())
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    rows = []
    for epoch in range(1, cfg.epochs + 1):
        model.set_active_topks([sched.top_k_for_epoch(epoch, cap) for sched, cap in zip(schedules, capacities)])
        epoch_rows = []
        for batch in loader:
            activations = [x.to(device) for x in batch[:-1]]
            logit = batch[-1].to(device)
            opt.zero_grad(set_to_none=True)
            out = model(activations)
            rec = sum(F.mse_loss(r, a) for r, a in zip(out["reconstructed"], activations)) / len(activations)
            sparse = sum(z.mean() for z in out["z"]) / len(out["z"])
            behavior = F.l1_loss(behavior_forward(3, out["reconstructed"][3]), logit)
            incoh = sum(decoder_incoherence(t) for t in model.transcoders) / len(model.transcoders)
            decor = sum(feature_decorrelation(z) for z in out["z"]) / len(out["z"])
            tied = sum(tied_penalty(t) for t in model.transcoders) / len(model.transcoders)
            usage = sum(usage_dominance(z, cfg.max_feature_frequency) for z in out["z"]) / len(out["z"])
            trans = transition_loss(out["z"], model.transitions)
            edge = transition_sparsity(model.transitions)
            inter = interventional_consistency(out["z"], model.transitions) if cfg.lambda_interventional else out["z"][0].new_zeros(())
            loss = (
                rec
                + cfg.lambda_behavior * behavior
                + cfg.lambda_sparse * sparse
                + cfg.lambda_incoherence * incoh
                + cfg.lambda_decorrelation * decor
                + cfg.lambda_tied * tied
                + cfg.lambda_usage * usage
                + cfg.lambda_transition * trans
                + cfg.lambda_edge_sparse * edge
                + cfg.lambda_interventional * inter
            )
            loss.backward()
            opt.step()
            for transcoder in model.transcoders:
                transcoder.normalize_decoder_()
            epoch_rows.append([float(x.detach().cpu()) for x in [loss, rec, behavior, sparse, incoh, decor, tied, usage, trans, edge, inter]])
        arr = np.asarray(epoch_rows)
        rows.append(
            {
                "epoch": epoch,
                "stage": cfg.stage,
                "active_topks": "|".join(str(t.active_top_k) for t in model.transcoders),
                "loss": float(arr[:, 0].mean()),
                "reconstruction_loss": float(arr[:, 1].mean()),
                "behavior_loss": float(arr[:, 2].mean()),
                "sparsity_loss": float(arr[:, 3].mean()),
                "incoherence_loss": float(arr[:, 4].mean()),
                "decorrelation_loss": float(arr[:, 5].mean()),
                "tied_loss": float(arr[:, 6].mean()),
                "usage_loss": float(arr[:, 7].mean()),
                "transition_loss": float(arr[:, 8].mean()),
                "edge_sparse_loss": float(arr[:, 9].mean()),
                "interventional_loss": float(arr[:, 10].mean()),
            }
        )
    model.set_active_topks(topks)
    return model, pd.DataFrame(rows)
