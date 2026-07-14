from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize_signed_contributions(contributions: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return contributions / contributions.abs().sum(dim=-1, keepdim=True).clamp_min(eps)


def contribution_consistency_loss(contrib_a: torch.Tensor, contrib_b: torch.Tensor) -> torch.Tensor:
    """Cosine consistency loss for signed FAN contribution vectors."""
    a = normalize_signed_contributions(contrib_a)
    b = normalize_signed_contributions(contrib_b)
    return (1.0 - F.cosine_similarity(a, b, dim=-1)).mean()


def make_masked_view(
    x: torch.Tensor,
    mask_probability: float = 0.05,
    noise_std: float = 0.01,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Soft augmentation view for contribution consistency training."""
    out = x
    if noise_std > 0:
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator) * float(noise_std)
        out = out + noise
    if mask_probability > 0:
        keep = torch.rand(x.shape, device=x.device, generator=generator) >= float(mask_probability)
        out = out * keep.to(dtype=x.dtype)
    return out


def center_anchor_loss(learned_centers: torch.Tensor, initial_centers: torch.Tensor) -> torch.Tensor:
    return (learned_centers - initial_centers.to(device=learned_centers.device, dtype=learned_centers.dtype)).square().mean()


def ordered_centers_penalty(centers: torch.Tensor, margin: float = 1e-4) -> torch.Tensor:
    if centers.shape[-1] < 2:
        return centers.new_zeros(())
    diffs = centers[..., 1:] - centers[..., :-1]
    return F.relu(float(margin) - diffs).mean()


def minimum_width_penalty(widths: torch.Tensor, min_width: float = 1e-3) -> torch.Tensor:
    return F.relu(float(min_width) - widths).mean()


def fan_stability_regularizer(
    output_a,
    output_b,
    *,
    learned_centers: torch.Tensor | None = None,
    initial_centers: torch.Tensor | None = None,
    widths: torch.Tensor | None = None,
    lambda_center_anchor: float = 0.0,
    lambda_ordered_centers: float = 0.0,
    lambda_min_width: float = 0.0,
) -> dict[str, torch.Tensor]:
    consistency = contribution_consistency_loss(output_a.signed_decision_contributions, output_b.signed_decision_contributions)
    total = consistency
    center_anchor = consistency.new_zeros(())
    ordered = consistency.new_zeros(())
    min_width = consistency.new_zeros(())
    if learned_centers is not None and initial_centers is not None and lambda_center_anchor:
        center_anchor = center_anchor_loss(learned_centers, initial_centers)
        total = total + float(lambda_center_anchor) * center_anchor
    if learned_centers is not None and lambda_ordered_centers:
        ordered = ordered_centers_penalty(learned_centers)
        total = total + float(lambda_ordered_centers) * ordered
    if widths is not None and lambda_min_width:
        min_width = minimum_width_penalty(widths)
        total = total + float(lambda_min_width) * min_width
    return {
        "total": total,
        "contribution_consistency": consistency,
        "center_anchor": center_anchor,
        "ordered_centers": ordered,
        "minimum_width": min_width,
    }
