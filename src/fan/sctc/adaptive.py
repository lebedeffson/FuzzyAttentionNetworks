from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def effective_rank(activation: torch.Tensor, eps: float = 1e-12) -> float:
    """Participation-ratio effective rank of flattened activations."""
    x = activation.reshape(-1, activation.shape[-1]).float()
    x = x - x.mean(dim=0, keepdim=True)
    cov = x.T @ x / max(1, x.shape[0] - 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    return float((eigvals.sum().square() / eigvals.square().sum().clamp_min(eps)).item())


def layer_specific_capacity(
    activation: torch.Tensor,
    multiplier: float = 6.0,
    minimum: int = 16,
    maximum: int = 96,
) -> int:
    rank = effective_rank(activation)
    return int(min(maximum, max(minimum, round(multiplier * rank))))


@dataclass(frozen=True)
class TopKAnnealingSchedule:
    target_top_k: int
    no_topk_epochs: int = 3
    anneal_until_epoch: int = 10
    initial_top_k: int = 32

    def top_k_for_epoch(self, epoch: int, n_features: int) -> int:
        if epoch <= self.no_topk_epochs:
            return min(n_features, self.initial_top_k)
        if epoch >= self.anneal_until_epoch:
            return min(n_features, self.target_top_k)
        span = max(1, self.anneal_until_epoch - self.no_topk_epochs)
        frac = (epoch - self.no_topk_epochs) / span
        value = round((1.0 - frac) * self.initial_top_k + frac * self.target_top_k)
        return int(min(n_features, max(self.target_top_k, value)))


class AdaptiveSparseTranscoder(nn.Module):
    """SCTC variant for V3.1 utilization experiments.

    The module normalizes input activations with train statistics, reconstructs
    through a sparse dictionary with an explicit reconstruction bias, supports
    top-k annealing, and can reinitialize dead features from large residuals.
    """

    def __init__(
        self,
        d_model: int,
        n_features: int,
        top_k: int = 12,
        train_mean: torch.Tensor | None = None,
        train_std: torch.Tensor | None = None,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_features = int(n_features)
        self.target_top_k = int(top_k)
        self.active_top_k = int(top_k)
        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model, bias=False)
        self.reconstruction_bias = nn.Parameter(torch.zeros(d_model))
        mean = torch.zeros(d_model) if train_mean is None else train_mean.detach().float().reshape(-1)
        std = torch.ones(d_model) if train_std is None else train_std.detach().float().reshape(-1)
        self.register_buffer("train_mean", mean)
        self.register_buffer("train_std", std.clamp_min(1e-5))
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.constant_(self.encoder.bias, 0.01)
        self.normalize_decoder_()

    @classmethod
    def from_train_activation(cls, activation: torch.Tensor, n_features: int, top_k: int) -> "AdaptiveSparseTranscoder":
        flat = activation.reshape(-1, activation.shape[-1]).float()
        return cls(
            d_model=activation.shape[-1],
            n_features=n_features,
            top_k=top_k,
            train_mean=flat.mean(dim=0),
            train_std=flat.std(dim=0),
        )

    def set_active_top_k(self, top_k: int) -> None:
        self.active_top_k = int(max(1, min(self.n_features, top_k)))

    def normalize(self, activation: torch.Tensor) -> torch.Tensor:
        return (activation - self.train_mean.view(*([1] * (activation.ndim - 1)), -1)) / self.train_std.view(*([1] * (activation.ndim - 1)), -1)

    def denormalize(self, activation: torch.Tensor) -> torch.Tensor:
        return activation * self.train_std.view(*([1] * (activation.ndim - 1)), -1) + self.train_mean.view(*([1] * (activation.ndim - 1)), -1)

    def forward(self, activation: torch.Tensor) -> dict[str, torch.Tensor]:
        x_norm = self.normalize(activation.float())
        z = torch.relu(self.encoder(x_norm))
        if 0 < self.active_top_k < z.shape[-1]:
            values, indices = torch.topk(z, k=self.active_top_k, dim=-1)
            sparse = torch.zeros_like(z)
            z = sparse.scatter(-1, indices, values)
        rec_norm = self.decoder(z) + self.reconstruction_bias
        reconstructed = self.denormalize(rec_norm)
        return {
            "z": z,
            "normalized": x_norm,
            "reconstructed_normalized": rec_norm,
            "reconstructed": reconstructed,
        }

    def normalize_decoder_(self) -> None:
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0).clamp_min(1e-8)
            self.decoder.weight.div_(norms.view(1, -1))
            self.encoder.weight.mul_(norms.view(-1, 1))
            self.encoder.bias.mul_(norms)

    @staticmethod
    def activation_frequency(z: torch.Tensor) -> torch.Tensor:
        return (z > 0).float().mean(dim=tuple(range(z.ndim - 1)))

    def resample_dead_features(
        self,
        activation: torch.Tensor,
        optimizer: torch.optim.Optimizer | None = None,
        frequency_threshold: float = 1e-5,
        max_resamples: int | None = None,
    ) -> int:
        with torch.no_grad():
            out = self(activation)
            freq = self.activation_frequency(out["z"])
            dead = torch.nonzero(freq < frequency_threshold, as_tuple=False).flatten()
            if max_resamples is not None:
                dead = dead[: int(max_resamples)]
            if dead.numel() == 0:
                return 0
            residual = (out["normalized"] - out["reconstructed_normalized"]).reshape(-1, self.d_model)
            hard = residual.norm(dim=-1).topk(k=min(dead.numel(), residual.shape[0])).indices
            directions = F.normalize(residual[hard], dim=-1)
            if directions.shape[0] < dead.numel():
                repeats = int(np.ceil(dead.numel() / directions.shape[0]))
                directions = directions.repeat(repeats, 1)[: dead.numel()]
            self.decoder.weight[:, dead] = directions.T
            self.encoder.weight[dead] = directions
            self.encoder.bias[dead] = 0.01
            self.normalize_decoder_()
            if optimizer is not None:
                self._reset_optimizer_units(optimizer, dead)
            return int(dead.numel())

    def _reset_optimizer_units(self, optimizer: torch.optim.Optimizer, feature_ids: torch.Tensor) -> None:
        for param in [self.encoder.weight, self.encoder.bias, self.decoder.weight]:
            state = optimizer.state.get(param)
            if not state:
                continue
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if key not in state:
                    continue
                buf = state[key]
                if param is self.encoder.weight:
                    buf[feature_ids] = 0
                elif param is self.encoder.bias:
                    buf[feature_ids] = 0
                elif param is self.decoder.weight:
                    buf[:, feature_ids] = 0


@dataclass(frozen=True)
class AdaptiveSCTCTrainConfig:
    n_features: int
    target_top_k: int
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    lambda_l1: float = 1e-5
    lambda_behavior: float = 0.1
    lambda_temporal: float = 0.0
    resample_every_steps: int = 500
    dead_threshold: float = 1e-5
    patience: int = 6


def temporal_consistency(z: torch.Tensor) -> torch.Tensor:
    if z.ndim < 3 or z.shape[1] < 2:
        return z.new_zeros(())
    return F.huber_loss(z[:, 1:] - z[:, :-1], torch.zeros_like(z[:, 1:] - z[:, :-1]))


def train_adaptive_sctc(
    train_activation: torch.Tensor,
    train_logit: torch.Tensor,
    behavior_forward: Callable[[torch.Tensor], torch.Tensor],
    cfg: AdaptiveSCTCTrainConfig,
    device: torch.device,
) -> tuple[AdaptiveSparseTranscoder, pd.DataFrame]:
    model = AdaptiveSparseTranscoder.from_train_activation(
        train_activation,
        n_features=cfg.n_features,
        top_k=cfg.target_top_k,
    ).to(device)
    schedule = TopKAnnealingSchedule(target_top_k=cfg.target_top_k)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    data = TensorDataset(train_activation.float(), train_logit.float())
    loader = DataLoader(data, batch_size=cfg.batch_size, shuffle=True)
    rows: list[dict] = []
    best_loss = float("inf")
    best_state = None
    stale = 0
    step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.set_active_top_k(schedule.top_k_for_epoch(epoch, cfg.n_features))
        losses = []
        resampled = 0
        for activation, logit in loader:
            step += 1
            activation = activation.to(device)
            logit = logit.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(activation)
            rec = F.mse_loss(out["reconstructed"], activation)
            sparse = out["z"].mean()
            behavior = F.l1_loss(behavior_forward(out["reconstructed"]), logit)
            temporal = temporal_consistency(out["z"])
            loss = rec + cfg.lambda_l1 * sparse + cfg.lambda_behavior * behavior + cfg.lambda_temporal * temporal
            loss.backward()
            opt.step()
            model.normalize_decoder_()
            if cfg.resample_every_steps > 0 and step % cfg.resample_every_steps == 0:
                resampled += model.resample_dead_features(activation, opt, cfg.dead_threshold)
            losses.append([float(loss.item()), float(rec.item()), float(sparse.item()), float(behavior.item()), float(temporal.item())])
        with torch.no_grad():
            z = model(train_activation.to(device))["z"]
            dead = float((model.activation_frequency(z) < cfg.dead_threshold).float().mean().item())
            l0 = float((z > 0).float().sum(dim=-1).mean().item())
        arr = np.asarray(losses)
        epoch_loss = float(arr[:, 0].mean())
        rows.append(
            {
                "epoch": epoch,
                "active_top_k": model.active_top_k,
                "loss": epoch_loss,
                "reconstruction_loss": float(arr[:, 1].mean()),
                "sparsity_loss": float(arr[:, 2].mean()),
                "behavior_loss": float(arr[:, 3].mean()),
                "temporal_loss": float(arr[:, 4].mean()),
                "dead_feature_fraction": dead,
                "L0_per_token": l0,
                "resampled_features": resampled,
            }
        )
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)
