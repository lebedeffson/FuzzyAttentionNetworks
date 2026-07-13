from __future__ import annotations

import torch


def remove_feature(a_l: torch.Tensor, z_i: torch.Tensor, decoder_direction: torch.Tensor) -> torch.Tensor:
    while z_i.ndim < a_l.ndim:
        z_i = z_i.unsqueeze(-1)
    return a_l - z_i * decoder_direction


def amplify_feature(a_l: torch.Tensor, eta: float, decoder_direction: torch.Tensor) -> torch.Tensor:
    return a_l + float(eta) * decoder_direction


def mean_abs_probability_effect(base_prob: torch.Tensor, intervened_prob: torch.Tensor) -> float:
    return float((intervened_prob - base_prob).abs().mean().item())
