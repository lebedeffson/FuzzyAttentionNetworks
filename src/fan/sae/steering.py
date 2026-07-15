from __future__ import annotations

import torch


def steer_feature(h: torch.Tensor, decoder_direction: torch.Tensor, activation_std: float, alpha: float) -> torch.Tensor:
    return h + float(alpha) * float(activation_std) * decoder_direction.view(1, -1)


def ablate_feature(h: torch.Tensor, z_j: torch.Tensor, decoder_direction: torch.Tensor) -> torch.Tensor:
    return h - z_j.view(-1, 1) * decoder_direction.view(1, -1)
