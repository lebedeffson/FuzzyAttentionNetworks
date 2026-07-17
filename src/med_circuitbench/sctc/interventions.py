from __future__ import annotations

import torch
import numpy as np


def remove_feature(a_l: torch.Tensor, z_i: torch.Tensor, decoder_direction: torch.Tensor) -> torch.Tensor:
    while z_i.ndim < a_l.ndim:
        z_i = z_i.unsqueeze(-1)
    return a_l - z_i * decoder_direction


def amplify_feature(a_l: torch.Tensor, eta: float, decoder_direction: torch.Tensor) -> torch.Tensor:
    return a_l + float(eta) * decoder_direction


def mean_abs_probability_effect(base_prob: torch.Tensor, intervened_prob: torch.Tensor) -> float:
    return float((intervened_prob - base_prob).abs().mean().item())


def downstream_response(
    q_base: np.ndarray,
    q_ablate: np.ndarray,
    q_push: np.ndarray,
    sigma: float,
) -> dict[str, float | bool]:
    sigma = float(sigma) + 1e-8
    dr_ablate = float(np.mean((np.asarray(q_base) - np.asarray(q_ablate)) / sigma))
    dr_push = float(np.mean((np.asarray(q_push) - np.asarray(q_base)) / sigma))
    inconsistent = bool(np.sign(dr_ablate) != 0 and np.sign(dr_push) != 0 and np.sign(dr_ablate) != np.sign(dr_push))
    return {
        "DR_ablate": dr_ablate,
        "DR_push": dr_push,
        "DR": float(0.5 * (dr_ablate + dr_push)),
        "inconsistent": inconsistent,
    }
