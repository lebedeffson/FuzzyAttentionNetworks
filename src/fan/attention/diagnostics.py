from __future__ import annotations

import torch


def attention_diagnostics(attention: torch.Tensor, gate: torch.Tensor, memberships: torch.Tensor) -> dict[str, float]:
    row_sum = attention.sum(dim=-1)
    return {
        "attention_row_sum_error": float(torch.max(torch.abs(row_sum - 1.0)).detach().cpu()),
        "nan_count": float(torch.isnan(attention).sum().detach().cpu()),
        "inf_count": float(torch.isinf(attention).sum().detach().cpu()),
        "mean_gate": float(gate.detach().mean().cpu()),
        "membership_saturation_fraction": float(((memberships < 1e-4) | (memberships > 1.0 - 1e-4)).float().mean().detach().cpu()),
        "mean_attention_entropy": float((-(attention.clamp_min(1e-8) * torch.log(attention.clamp_min(1e-8))).sum(dim=-1)).mean().detach().cpu()),
    }
