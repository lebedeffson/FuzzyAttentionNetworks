from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .model import SparseTranscoder


@dataclass(frozen=True)
class SCTCTrainConfig:
    n_features: int
    lambda_l1: float = 1e-5
    lambda_behavior: float = 0.1
    lambda_temporal: float = 0.0
    learning_rate: float = 1e-3
    epochs: int = 4
    batch_size: int = 128


def temporal_consistency(z: torch.Tensor) -> torch.Tensor:
    if z.ndim < 3 or z.shape[1] < 2:
        return z.new_zeros(())
    return F.huber_loss(z[:, 1:] - z[:, :-1], torch.zeros_like(z[:, 1:] - z[:, :-1]))


def train_sctc(
    train_activation: torch.Tensor,
    train_logit: torch.Tensor,
    behavior_forward,
    cfg: SCTCTrainConfig,
    device: torch.device,
) -> tuple[SparseTranscoder, pd.DataFrame]:
    model = SparseTranscoder(train_activation.shape[-1], cfg.n_features).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    dataset = TensorDataset(train_activation.float(), train_logit.float())
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    rows = []
    for epoch in range(cfg.epochs):
        losses = []
        for activation, logit in loader:
            activation = activation.to(device)
            logit = logit.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(activation)
            rec = F.mse_loss(out["reconstructed"], activation)
            sparse = out["z"].mean()
            recon_logit = behavior_forward(out["reconstructed"])
            behavior = F.l1_loss(recon_logit, logit)
            temporal = temporal_consistency(out["z"])
            loss = rec + cfg.lambda_l1 * sparse + cfg.lambda_behavior * behavior + cfg.lambda_temporal * temporal
            loss.backward()
            opt.step()
            model.normalize_decoder_()
            losses.append([float(loss.item()), float(rec.item()), float(sparse.item()), float(behavior.item()), float(temporal.item())])
        arr = np.asarray(losses)
        rows.append(
            {
                "epoch": epoch + 1,
                "loss": float(arr[:, 0].mean()),
                "reconstruction_loss": float(arr[:, 1].mean()),
                "sparsity_loss": float(arr[:, 2].mean()),
                "behavior_loss": float(arr[:, 3].mean()),
                "temporal_loss": float(arr[:, 4].mean()),
            }
        )
    return model, pd.DataFrame(rows)
